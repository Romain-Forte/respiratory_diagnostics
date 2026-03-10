from __future__ import annotations

from pathlib import Path

from PIL import Image


def convert_png_to_eps(png_path: str | Path) -> Path:
    """
    Convert a PNG image to EPS format in the same directory.

    Args:
        png_path: Path to the source PNG image.

    Returns:
        Path pointing to the generated EPS file.

    Raises:
        FileNotFoundError: If the PNG file does not exist.
        ValueError: If the provided file is not a PNG image.
    """

    source_path = Path(png_path)
    if not source_path.exists():
        raise FileNotFoundError(f"No file found at {source_path}")

    if source_path.suffix.lower() != ".png":
        raise ValueError(f"Expected a .png file, got {source_path.suffix}")

    target_path = source_path.with_suffix(".eps")

    with Image.open(source_path) as image:
        # EPS requires RGB; convert if needed to avoid Pillow warnings.
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(target_path, format="EPS")

    return target_path


if __name__ == "__main__":
    png_path = r"C:\Users\romai\Desktop\travail\avancement\model_grid_search\Invasive_pulmonary_aspergillosis\roc_curve.png"
    
    resultats = convert_png_to_eps(png_path)
    


