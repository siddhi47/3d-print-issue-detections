import os
import sys
import torch
import pandas as pd
import torchvision.transforms as transforms


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import CDataset, load_config
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
    parser.add_argument(
            "--image_dir",
            type=str,
            default=test_config["image_dir"],
            help="Image directory to use for testing",
        )
    return parser



def test(test_args):
    test_args = test_args.parse_args()
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(PROJECT_DIR, test_args.save_dir, test_args.model)
    if not os.path.exists(model_path):
        print("Model not found")
        return

    net = models[test_args.model](test_args.num_classes).to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((test_args.image_size, test_args.image_size)),
        ]
    )

    test_dataset = CDataset(
        test_args.reference_file, test_args.image_dir, test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_args.batch_size, shuffle=False
    )

    predictions = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, _  = data[0].to(device), data[1].to(device)

            outputs = net(inputs)
            outputs = torch.argmax(outputs, dim=1)
            predictions.extend(outputs.cpu().numpy())
            print(f"Batch {i} out of {len(test_loader)}")
    print(predictions)
    test_ref = pd.read_csv(test_args.reference_file)
    test_ref["label"] = predictions
    test_ref.to_csv(f"results/{test_args.model}_test.csv", index=False)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test a model")
    parser = test_args(parser)
    test(parser)

