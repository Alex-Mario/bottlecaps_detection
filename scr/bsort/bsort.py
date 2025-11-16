import argparse
from train import train_yolo
from infer import run_inference
import yaml
from ultralytics import YOLO


def load_config(config_path: str) -> dict:
    """Load YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Dictionary with configuration parameters.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main() -> None:
    """Entry point for the bsort CLI."""
    parser = argparse.ArgumentParser(
        description="bsort CLI for YOLOv8 bottle cap detection"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train YOLO model")
    train_parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")

    # Infer command
    infer_parser = subparsers.add_parser("infer", help="Run inference")
    infer_parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    infer_parser.add_argument("--image", type=str, required=True, help="Path to input image")

    args = parser.parse_args()

    config = load_config(args.config)

    if args.command == "train":
        model_name = config.get("model_name", "yolov8n.pt")
        data_yaml = config["dataset_path"] + "/data.yaml"
        epochs = config.get("epochs", 100)
        # Panggil fungsi training
        train_yolo(model_name=model_name, data_yaml=data_yaml, epochs=epochs)

    elif args.command == "infer":
        model_path = config.get("best_model_path", "best.pt")
        # Panggil fungsi inference
        run_inference(model_path=model_path, image_path=args.image)


if __name__ == "__main__":
    main()
