import argparse
import os

import numpy as np
from torchvision import transforms

_TRANSFORMS = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=10)

    return parser.parse_args()


def set_cuda_device(device: str) -> None:
    device_ids = device.split(",")

    if all(dev.isdigit() for dev in device_ids):
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        print(f"Using CUDA device:{device}")
    else:
        raise ValueError("Invalid device index. Must be a numeric value.")