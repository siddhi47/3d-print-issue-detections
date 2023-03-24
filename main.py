import os
import csv
import sys
import torch
import json
import argparse


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.train import train, train_args

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--type",
        type=str,
        default="train",
        help="train or test",
    )
    return parser



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


def main(parser):
    args = parser.parse_args()
    if args.type == "train":
        train(parser)
    elif args.type == "test":
        pass
        # test(args)
    else:
        print("Invalid type")


if __name__ == "__main__":
    parser = main_args()
    args = parser.parse_args()
    if args.type == "train":
        parser = train_args(parser)
    elif args.type == "test":
        parser = test_args(parser)
    else:
        print("Invalid type")
    main(parser)
