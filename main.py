import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn

import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.utils import train_val_split, CDataset, calc_metrics
from src.architectures import Net_VGG
import json
import argparse
import csv

def load_config(config_path, train=True):
    with open(config_path) as f:
        config = json.load(f)
    if train:
        config = config["train"]
    else:
        config = config["test"]
    return config


def main_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--type",
        type=str,
        default="train",
        help="train or test",
    )
    return parser


def train_args(parser):
    # parser = argparse.ArgumentParser(description="Train a model")
    train_config = load_config("config.json", train=True)

    parser.add_argument(
        "--model",
        type=str,
        default=train_config["model"],
        help="Model to use for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=train_config["epochs"],
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=train_config["batch_size"],
        help="Batch size to use for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=train_config["lr"],
        help="Learning rate to use for training",
    )
    parser.add_argument(
        "--train_val_split",
        type=float,
        default=train_config["train_val_split"],
        help="Percentage of data to use for validation",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=train_config["num_classes"],
        help="Number of classes to use for training",
    )
    parser.add_argument(
        "--reference_file",
        type=str,
        default=train_config["reference_file"],
        help="Reference file to use for training",
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        default=train_config["shuffle"],
        help="Shuffle data",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=train_config["image_size"],
        help="Image size to use for training",
    )

    parser.add_argument(
        "--image_dir",
        type=str,
        default=train_config["image_dir"],
        help="Image directory to use for training",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=train_config["save_dir"],
        help="Directory to save model",
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


def train(train_args):
    train_args = train_args.parse_args()
    IMAGE_SIZE = (train_args.image_size, train_args.image_size)
    BATCH_SIZE = train_args.batch_size
    validation_split = train_args.train_val_split
    shuffle_dataset = train_args.shuffle
    NUM_CLASSES = train_args.num_classes
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize(IMAGE_SIZE),
        ]
    )

    dataset = CDataset(train_args.reference_file, train_args.image_dir, transform)
    train_loader, validation_loader = train_val_split(
        dataset, BATCH_SIZE, val_percent=validation_split, shuffle=shuffle_dataset
    )
    model_path = os.path.join(PROJECT_DIR, train_args.save_dir, train_args.model)

    if os.path.exists(model_path):
        print("Saved model found, loading...")
        net = Net_VGG(NUM_CLASSES)
        net.load_state_dict(torch.load(model_path))
        print("Model loaded")
    else:
        print("No saved model found, training...")
        net = Net_VGG(NUM_CLASSES)
        print("Model created")

    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    os.mkdir('results',)
    for epoch in range(train_args.epochs):
        train_acc = []
        train_precisions = []
        train_recalls = []
        train_f1s = []

        for i, data in enumerate(train_loader, 0):
            net.train()
            net.to(device)
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            acc, precisions, recalls, f1s, cm = calc_metrics(
                outputs.argmax(dim=1).cpu(), labels.cpu()
            )

            train_acc.append(acc)
            train_precisions.append(precisions)
            train_recalls.append(recalls)
            train_f1s.append(f1s)
            print(
                "\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t accuracy: {:.6f}".format(
                    epoch,
                    i * len(data),
                    len(train_loader),
                    100.0 * i / len(train_loader),
                    loss.item(),
                    acc,
                    end="",
                )
            )
        with open(f"results/{train_args.model}_train_metrics.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    train_args.model,
                    epoch,
                    np.mean(train_acc),
                    np.mean(train_precisions),
                    np.mean(train_recalls),
                    np.mean(train_f1s),
                ]
            )
        

        # save model
        os.makedirs(os.path.join(PROJECT_DIR, train_args.save_dir), exist_ok=True)
        torch.save(net.state_dict(), model_path)

        if epoch % 1 == 0:
            print("validation")
            net.eval()
            net.to(device)
            val_loss = 0
            val_acc = []
            val_precisions = []
            val_recalls = []
            val_f1s = []

            with torch.no_grad():
                val_acc = []
                for i, data in enumerate(validation_loader):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    output = net(inputs)
                    val_loss += criterion(output, labels).item()
                    acc = (output.argmax(dim=1) == labels).float().mean()
                    val_acc.append(acc.item())
                    val_loss /= len(validation_loader)
                    acc, precisions, recalls, f1s, cm = calc_metrics(
                        output.argmax(dim=1).cpu(), labels.cpu()
                    )
                    val_acc.append(acc)
                    val_precisions.append(precisions)
                    val_recalls.append(recalls)
                    val_f1s.append(f1s)

                with open(f"results/{train_args.model}_val_metrics.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            train_args.model,
                            epoch,
                            np.mean(val_acc),
                            np.mean(val_precisions),
                            np.mean(val_recalls),
                            np.mean(val_f1s)
                        ]
                    )

                print(
                    "Val set: Average loss: {:.4f}, Accuracy: {}\n".format(
                        val_loss,
                        np.mean(val_acc),
                    )
                )
    print("Finished Training")


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
