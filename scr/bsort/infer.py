from ultralytics import YOLO
from pathlib import Path


def infer_image(image_path: str, config: dict):
    """
    Run inference on a single image
    Args:
        image_path (str): path to image
        config (dict): configuration dictionary
    """
    model = YOLO("best.pt")
    results = model.predict(source=image_path, conf=config["inference"]["conf_threshold"])

    # Save or show results
    save_path = Path(image_path).parent / "result.jpg"
    results[0].save(save_path)
    print(f"Result saved to {save_path}")
