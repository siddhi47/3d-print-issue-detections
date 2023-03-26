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


def train_args(parser):
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
    
    parser.add_argument(
            "--validation",
            type=bool,
            default=train_config["validation"],
            help="Whether to use validation or not",
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
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]
    )

    dataset = CDataset(train_args.reference_file, train_args.image_dir, transform)
    if train_args.validation:

        train_loader, validation_loader = train_val_split(
            dataset, BATCH_SIZE, val_percent=validation_split, shuffle=shuffle_dataset
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=shuffle_dataset
        )

    model_path = os.path.join(PROJECT_DIR, train_args.save_dir, train_args.model)

    if os.path.exists(model_path):
        print("Saved model found, loading...")
        net = models[train_args.model](NUM_CLASSES)
        net.load_state_dict(torch.load(model_path))
        print("Model loaded")
    else:
        print("No saved model found, training...")
        net = models[train_args.model](NUM_CLASSES)
        print("Model created")

    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    if not os.path.exists("results"):
        os.mkdir("results")

    for epoch in range(train_args.epochs):
        train_acc = []
        train_precisions = []
        train_recalls = []
        train_f1s = []

        for i, data in enumerate(train_loader, 0):
            net.train()
            net.to(device)
            inputs, labels = data[0].to(device), data[1].to(device)
            print(labels)
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

        if train_args.validation:
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
                                np.mean(val_f1s),
                            ]
                        )

                    print(
                        "Val set: Average loss: {:.4f}, Accuracy: {}\n".format(
                            val_loss,
                            np.mean(val_acc),
                        )
                    )
    print("Finished Training")


if __name__ == "__main__":
    parser = train_parser()
    train(parser)
