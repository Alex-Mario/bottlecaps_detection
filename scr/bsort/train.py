from ultralytics import YOLO
import wandb


def train_model(config: dict):
    """
    Train YOLO model
    Args:
        config (dict): configuration dictionary
    """
    wandb.init(project=config["wandb"]["project"])

    model = YOLO(config["model"]["pretrained"])
    model.train(
        data=config["dataset"]["train_path"],
        epochs=config["model"]["epochs"],
        batch=config["model"]["batch_size"],
        lr=config["model"]["lr"],
    )

    # Save best weights
    model.save("best.pt")
