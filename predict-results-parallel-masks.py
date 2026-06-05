#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import concurrent.futures
import glob
import hashlib
import multiprocessing as mp
import os
import threading
import time

import cv2
import numpy as np
import torch
import tqdm
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from train_net import register_dataset
from modules import add_deepwindows_network_config


# constants
WINDOW_NAME = "COCO detections"
_IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
_thread_local = threading.local()


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deepwindows_network_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def parse_prediction_color(value):
    """
    Parse an RGB color supplied as one of:
      --prediction-color white
      --prediction-color 255,255,255
      --prediction-color '#ffffff'

    Returns None when random per-instance colors should be used.
    """
    if value is None or value == "":
        return None

    value = value.strip().lower()
    named_colors = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
    }
    if value in named_colors:
        return named_colors[value]

    if value.startswith("#"):
        value = value[1:]
    if len(value) == 6 and all(c in "0123456789abcdef" for c in value):
        return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))

    parts = value.replace(";", ",").split(",")
    if len(parts) == 3:
        try:
            color = tuple(int(p.strip()) for p in parts)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid RGB color: {value!r}") from exc
        if all(0 <= c <= 255 for c in color):
            return color

    raise argparse.ArgumentTypeError(
        "--prediction-color must be a name, '#RRGGBB', or 'R,G,B', e.g. white, #ffffff, or 255,255,255"
    )


def get_demo(cfg):
    """
    Keep one VisualizationDemo per worker thread.

    This allows up to max_threads calls to demo.run_on_image() to be active at the
    same time. It also avoids sharing one predictor/model object across threads.
    Note that this can use roughly max_threads times the model memory.
    """
    if not hasattr(_thread_local, "demo"):
        _thread_local.demo = VisualizationDemo(cfg)
    return _thread_local.demo


def list_input_images(input_path):
    if os.path.isdir(input_path):
        paths = [
            os.path.join(input_path, name)
            for name in sorted(os.listdir(input_path))
            if os.path.splitext(name)[1].lower() in _IMAGE_EXTENSIONS
        ]
    else:
        paths = sorted(glob.glob(os.path.expanduser(input_path)))
        if not paths and os.path.isfile(input_path):
            paths = [input_path]

    assert paths, f"No input images found for: {input_path}"
    return paths


def random_color_for_instance(image_path, instance_index):
    """Return a stable random-looking RGB color for this image/instance."""
    digest = hashlib.sha1(f"{image_path}:{instance_index}".encode("utf-8")).digest()
    return tuple(int(x) for x in digest[:3])


def instances_to_mask_image(predictions, image_shape, prediction_color=None, image_path=""):
    """
    Convert Detectron2 instance predictions into a colored instance segmentation mask.

    The output is RGB, uint8, shape H x W x 3. Background is black. If
    prediction_color is None, every instance receives a stable random color. If
    prediction_color is an RGB tuple, all predicted instance pixels receive that
    same color, e.g. (255, 255, 255) for a binary white mask.
    """
    height, width = image_shape[:2]
    mask_image = np.zeros((height, width, 3), dtype=np.uint8)

    if "instances" not in predictions:
        return mask_image

    instances = predictions["instances"].to("cpu")
    if not instances.has("pred_masks") or len(instances) == 0:
        return mask_image

    masks = instances.pred_masks.numpy().astype(bool)
    for i, mask in enumerate(masks):
        if prediction_color is None:
            color = random_color_for_instance(image_path, i)
        else:
            color = prediction_color
        mask_image[mask] = color

    return mask_image


def output_filename_for(input_path, output_path, multiple_inputs):
    if not output_path:
        return None

    if os.path.isdir(output_path) or multiple_inputs:
        os.makedirs(output_path, exist_ok=True)
        return os.path.join(output_path, os.path.basename(input_path))

    return output_path


def process_image(path, cfg, output_path, multiple_inputs, prediction_color):
    img = read_image(path, format="BGR")
    start_time = time.time()

    demo = get_demo(cfg)
    predictions, _visualized_output = demo.run_on_image(img)

    mask_rgb = instances_to_mask_image(
        predictions, img.shape, prediction_color=prediction_color, image_path=path
    )
    out_filename = output_filename_for(path, output_path, multiple_inputs)

    if out_filename:
        # cv2.imwrite expects BGR; mask_rgb is easier for users to reason about.
        cv2.imwrite(out_filename, mask_rgb[:, :, ::-1])

    elapsed = time.time() - start_time
    num_instances = len(predictions["instances"]) if "instances" in predictions else 0
    return path, out_filename, num_instances, elapsed


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        help="Input image directory, image file, or glob pattern such as 'directory/*.jpg'.",
    )
    parser.add_argument(
        "--output",
        help="Directory or file path where instance segmentation masks are saved. For multiple inputs, use a directory.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=1,
        help="Maximum number of images to process in parallel with demo.run_on_image().",
    )
    parser.add_argument(
        "--prediction-color",
        type=parse_prediction_color,
        default=None,
        help=(
            "Optional RGB color for all predicted masks. Omit for random per-instance colors. "
            "Examples: white, '#ffffff', '255,255,255'."
        ),
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    assert args.max_threads >= 1, "--max-threads must be >= 1"

    cfg = setup_cfg(args)
    register_dataset()

    if args.input:
        input_paths = list_input_images(args.input)
        multiple_inputs = len(input_paths) > 1
        if args.output and multiple_inputs:
            os.makedirs(args.output, exist_ok=True)

        if args.output:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_threads) as executor:
                futures = [
                    executor.submit(
                        process_image,
                        path,
                        cfg,
                        args.output,
                        multiple_inputs,
                        args.prediction_color,
                    )
                    for path in input_paths
                ]
                for future in tqdm.tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    disable=not args.output,
                ):
                    path, out_filename, num_instances, elapsed = future.result()
                    logger.info(
                        "{}: detected {} instances in {:.2f}s -> {}".format(
                            os.path.basename(path), num_instances, elapsed, out_filename
                        )
                    )
        else:
            # Interactive display is intentionally kept sequential.
            for path in input_paths:
                path, _out_filename, num_instances, elapsed = process_image(
                    path,
                    cfg,
                    output_path=None,
                    multiple_inputs=multiple_inputs,
                    prediction_color=args.prediction_color,
                )
                logger.info(
                    "{}: detected {} instances in {:.2f}s".format(
                        os.path.basename(path), num_instances, elapsed
                    )
                )
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                # process_image does not return the image in display mode, so rerun only for the display use case.
                img = read_image(path, format="BGR")
                predictions, _ = get_demo(cfg).run_on_image(img)
                mask_rgb = instances_to_mask_image(
                    predictions, img.shape, args.prediction_color, image_path=path
                )
                cv2.imshow(WINDOW_NAME, mask_rgb[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
