import argparse
from scr.bsort.bsort import train_model
from scr.bsort.bsort import infer_image
import yaml

def main():
    parser = argparse.ArgumentParser(prog="bsort")
    parser.add_argument("command", choices=["train", "infer"])
    parser.add_argument("--config", type=str, default="settings.yaml")
    parser.add_argument("--image", type=str, help="image path for inference")

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.command == "train":
        train_model(config)
    elif args.command == "infer":
        if not args.image:
            raise ValueError("Please provide --image for inference")
        infer_image(args.image, config)

if __name__ == "__main__":
    main()
