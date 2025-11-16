# src/bsort/main.py

from pathlib import Path

import typer
import yaml
from ultralytics import YOLO

import wandb  # Import W&B

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
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


@app.command()
def train(
    config: Path = typer.Option(
        "configs/settings.yaml", "--configs", help="Path ke file settings.yaml."
    )
):
    """
    Melatih model YOLO berdasarkan file konfigurasi.
    """
    typer.echo(f"Memuat konfigurasi dari: {config}")
    try:
        params = load_config(config)
    except FileNotFoundError:
        typer.secho(
            f"Error: File konfigurasi {config} tidak ditemukan.", fg=typer.colors.RED
        )
        raise typer.Exit(1)

    typer.echo("Logging in to W&B...")
    wandb.login()

    model = YOLO(params["model"]["base_model"])

    typer.echo("Memulai pelatihan model...")
    model.train(
        data=params["data"]["data_config_file"],
        epochs=params["train"]["epochs"],
        batch=params["train"]["batch_size"],
        imgsz=params["model"]["imgsz"],
        project=params["train"]["wandb_project"],
        name="cli_train_run",
    )

    typer.secho(
        "Pelatihan selesai. Model terbaik ada di folder 'runs/'.", fg=typer.colors.GREEN
    )
    typer.echo("PENTING: Pindahkan 'best.pt' ke 'models/' dan update 'settings.yaml'.")


@app.command()
def infer(
    config: Path = typer.Option(
        "configs/settings.yaml", "--configs", help="Path ke file settings.yaml."
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
    results = model(image, conf=params["infer"]["confidence_threshold"])

    results[0].show()
    typer.secho(f"Inferensi selesai.", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
