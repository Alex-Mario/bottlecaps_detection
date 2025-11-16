"""CLI program for bottle cap detection (training and inference)."""

# src/bsort/main.py

from pathlib import Path

import typer
import wandb
import yaml
from ultralytics import YOLO

# Buat aplikasi Typer, ini adalah entry point CLI kita
app = typer.Typer(help="CLI untuk training dan inferensi model deteksi tutup botol.")


def load_config(config_path: Path) -> dict:
    """
    Memuat file konfigurasi YAML. (Ini adalah Docstring Google Style)

    Args:
        config_path (Path): Path ke file settings.yaml.

    Returns:
        dict: Konfigurasi dalam bentuk dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


@app.command()
def train(
    config: Path = typer.Option(
        "configs/settings.yaml", "--config", help="Path ke file settings.yaml."
    )
):
    """
    Melatih model DAN otomatis mengekspor ke ONNX ke folder 'models/'.
    """
    typer.echo(f"Memuat konfigurasi dari: {config}")
    try:
        params = load_config(config)
    except FileNotFoundError as exc:
        typer.secho(
            f"Error: File konfigurasi {config} tidak ditemukan.", fg=typer.colors.RED
        )
        raise typer.Exit(1) from exc

    typer.echo("Logging in to W&B...")
    wandb.login()

    model = YOLO(params["model"]["pretrained"])
    imgsz = params["model"]["imgsz"]  # Ambil imgsz untuk export

    typer.echo("Memulai pelatihan model...")
    # 'model' akan di-update 'in-place' dengan bobot terbaik setelah training
    model.train(
        data=params["data"]["data_config_file"],
        epochs=params["train"]["epochs"],
        batch=params["train"]["batch_size"],
        imgsz=imgsz,
        project=params["train"]["wandb_project"],
        # patience=params["model"]["patience"],
        name="cli_train_run4",
    )

    typer.echo("Pelatihan selesai. Memindahkan model...")

    # Dapatkan path ke best.pt secara dinamis
    # Ini jauh lebih aman daripada hardcoding path
    best_pt_path = Path(model.trainer.best)

    models_folder = Path("models")
    models_folder.mkdir(exist_ok=True)  # Buat folder 'models/' jika belum ada

    if best_pt_path.exists():
        # 1. Pindahkan best.pt ke models/
        target_pt_path = models_folder / best_pt_path.name  # (e.g., models/best.pt)
        best_pt_path.replace(target_pt_path)
        typer.secho(
            f"Best model dipindahkan ke: {target_pt_path}", fg=typer.colors.GREEN
        )

        # 2. Export ke ONNX
        # model.export() akan meng-export model yang ada di memori
        # dan me-return string path ke file .onnx yang baru dibuat
        typer.echo("Mengekspor model ke ONNX...")
        original_onnx_path_str = model.export(format="onnx", imgsz=imgsz)

        original_onnx_path = Path(original_onnx_path_str)

        # 3. Pindahkan file .onnx ke folder models/
        target_onnx_path = (
            models_folder / original_onnx_path.name
        )  # (e.g., models/best.onnx)
        original_onnx_path.replace(target_onnx_path)

        typer.secho(
            f"ONNX model disimpan di: {target_onnx_path}", fg=typer.colors.GREEN
        )

    else:
        typer.secho(
            f"Error: best.pt tidak ditemukan di path: {best_pt_path}",
            fg=typer.colors.RED,
        )
        typer.secho("Export ONNX dibatalkan.", fg=typer.colors.RED)

    typer.secho("Proses training dan export selesai.", fg=typer.colors.GREEN)


@app.command()
def infer(
    config: Path = typer.Option(
        "configs/settings.yaml", "--config", help="Path ke file settings.yaml."
    ),
    image: Path = typer.Option(
        ..., "--image", help="Path ke gambar yang ingin diprediksi."
    ),
):
    """
    Menjalankan inferensi pada satu gambar.
    """
    if not image.exists():
        typer.secho(f"Error: File gambar {image} tidak ditemukan.", fg=typer.colors.RED)
        raise typer.Exit(1)

    params = load_config(config)
    model_path = params["infer"]["model_weights"]

    if not Path(model_path).exists():
        typer.secho(
            f"Error: File model {model_path} tidak ditemukan.", fg=typer.colors.RED
        )
        raise typer.Exit(1)

    typer.echo(f"Memuat model dari {model_path}...")
    model = YOLO(model_path)

    typer.echo(f"Menjalankan inferensi pada {image}...")
    results = model(
        image,
        conf=params["infer"]["confidence_threshold"],
        iou=params["infer"]["iou_threshold"],
        # # imgsz=params["infer"]["image_size"], #(tidak berpengaruh jika format model onnx)
        # device=params["infer"]["device"],
        # half=params["infer"]["half_precision"],
        # augment=params["infer"]["augment"],
        # batch=params["infer"]["batch_size"]
    )

    results[0].show()
    typer.secho("Inferensi selesai.", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
