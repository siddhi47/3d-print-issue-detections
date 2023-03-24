import os
import sys
import csv
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import train_val_split, calc_metrics, CDataset, load_config
from src.models import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_args(parser):
    # parser = argparse.ArgumentParser(description="Test a model")
    test_config = load_config("config.json", train=False)

    parser.add_argument(
        "--model",
        type=str,
        default=test_config["model"],
        help="Model to use for testing",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=test_config["batch_size"],
        help="Batch size to use for testing",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=test_config["num_classes"],
        help="Number of classes to use for testing",
    )
    parser.add_argument(
        "--reference_file",
        type=str,
        default=test_config["reference_file"],
        help="Reference file to use for testing",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=test_config["save_dir"],
        help="Path of saved model",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=test_config["image_size"],
        help="Image size to use for testing",
    )

    return parser



def test(test_args):
    pass


