from scr.bsort.train import train_model

def test_train_model():
    # config minimal untuk test
    config = {
        "model": {
            "pretrained": "yolov8n.pt",
            "epochs": 1,  # cukup 1 epoch untuk test
            "batch_size": 2,
            "lr0": 0.0001,
            "optimizer": "Adam",
            "dropout": 0.0,
            "patience": 1,
            "project": "test_project",
            "run_name": "test_run"
        },
        "dataset": {
            "data_yaml": "tests/data_dummy/data_dummy.yaml"
        },
        "wandb": {"project": "test_project"}
    }

    # Jalankan train_model, seharusnya tidak error
    train_model(config)
