"""
EXR → PNG multi-weather data conversion for WeatherSynthetic Street.

Converts foggy and night EXR images to PNG (matching existing rain PNG layout).
Pairs each weather image with its sunny (clean) counterpart by numeric index.

Usage:
    # Convert all weather types
    python scripts/prepare_multiweather_data.py

    # Verify only (check counts, no conversion)
    python scripts/prepare_multiweather_data.py --verify

    # Convert specific weather type only
    python scripts/prepare_multiweather_data.py --weather foggy
"""

import os
import sys
import argparse
import struct
from pathlib import Path

import numpy as np
from PIL import Image

# ── Optional OpenEXR import ──
try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False


# ── Paths ──
EXR_ROOT = Path("/gz-data/weathersynthetic_street/image")
PNG_ROOT = Path("/gz-data/weathersynthetic_street_png")
NIGHT_EXR_DIR = Path("/root/OPWA/night")


# ── EXR reading ──

def read_exr_channels(filepath: str) -> np.ndarray:
    """Read an EXR file and return RGB float array (H, W, 3) in linear space."""
    if not HAS_OPENEXR:
        raise ImportError("OpenEXR is required. Install with: pip install OpenEXR")

    exr_file = OpenEXR.InputFile(filepath)
    header = exr_file.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Determine pixel type
    channels = header["channels"]
    if "R" in channels and "G" in channels and "B" in channels:
        pixel_type = channels["R"].type
    elif "B" in channels and "G" in channels and "R" in channels:
        pixel_type = channels["B"].type
    else:
        raise KeyError(f"EXR file {filepath} has no RGB channels (has: {list(channels.keys())})")

    # Float16 or Float32
    if pixel_type == Imath.PixelType(Imath.PixelType.FLOAT):
        fmt = "f"
        nbytes = 4
    elif pixel_type == Imath.PixelType(Imath.PixelType.HALF):
        fmt = "e"
        nbytes = 2
    else:
        raise ValueError(f"Unsupported EXR pixel type: {pixel_type}")

    # Read RGBA (or just RGB)
    channel_names = ["R", "G", "B"]
    raw_channels = exr_file.channels(channel_names, pixel_type)
    exr_file.close()

    channels_arr = []
    for raw in raw_channels:
        arr = struct.unpack(f"<{width * height}{fmt}", raw)
        channels_arr.append(np.array(arr, dtype=np.float32).reshape(height, width))

    return np.stack(channels_arr, axis=-1)  # (H, W, 3)


def exr_to_png_array(filepath: str) -> np.ndarray:
    """读 EXR，线性 clamp 到 [0,1]，转 uint8。

    和已有 rain PNG 保持一致：线性值 × 255，不做 tonemap/gamma。
    WeatherSynthetic 的 EXR 存的是线性辐射值，大部分值远小于 1.0，
    直接 clamp 即可保留正确亮度。
    """
    linear = read_exr_channels(filepath)
    return (np.clip(linear, 0.0, 1.0) * 255.0).astype(np.uint8)


def save_exr_as_png(exr_path: str, png_path: str, target_size: tuple = None):
    """Convert a single EXR file to PNG, resizing if target_size given."""
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    arr = exr_to_png_array(exr_path)
    img = Image.fromarray(arr, mode="RGB")
    if target_size:
        img = img.resize(target_size, Image.BILINEAR)
    img.save(png_path)


# ── Pairing logic ──

def find_exr_pairs(weather_dir: Path) -> list:
    """List (_image.exr) files sorted numerically."""
    exr_files = sorted(weather_dir.glob("*_image.exr"))
    return exr_files


def index_from_name(path: Path) -> str:
    """Extract numeric index from '0000_image.exr' → '0000'."""
    return path.stem.split("_")[0]  # "0000_image" → "0000"


def train_test_split(files: list, train_count: int = 400) -> tuple:
    """Split sorted files: first train_count go to train, rest to test."""
    return files[:train_count], files[train_count:]


def weather_name(weather_exr_dir_name: str) -> str:
    """Map EXR directory name to output short name."""
    mapping = {
        "foggy": "fog",
        "night": "night",
        "rainy": "rain",
    }
    return mapping.get(weather_exr_dir_name, weather_exr_dir_name)


# ── Per-weather conversion ──

def convert_weather(
    weather_exr_dir: Path,
    sunny_exr_dir: Path,
    output_subdir: str,
    train_count: int = 400,
    target_size: tuple = (512, 512),
):
    """
    Convert one weather's EXR images to PNG.

    Args:
        weather_exr_dir: Directory with weather EXR files (e.g. .../foggy/)
        sunny_exr_dir: Directory with sunny EXR files (e.g. .../sunny/)
        output_subdir: Output subdir name (e.g. "fog", "night")
        train_count: Number of train pairs (rest go to test)
    """
    weather_images = find_exr_pairs(weather_exr_dir)
    if not weather_images:
        print(f"  No _image.exr files found in {weather_exr_dir}, skipping.")
        return

    # Build sunny index lookup
    sunny_images = find_exr_pairs(sunny_exr_dir)
    sunny_by_index = {index_from_name(s): s for s in sunny_images}

    train_weather, test_weather = train_test_split(weather_images, train_count)

    splits = [("train", train_weather), ("test", test_weather)]
    converted = 0
    skipped = 0

    for split_name, weather_files in splits:
        out_a = PNG_ROOT / output_subdir / f"{split_name}_A"
        out_b = PNG_ROOT / output_subdir / f"{split_name}_B"

        for wf in weather_files:
            idx = index_from_name(wf)
            out_name = f"{idx}.png"

            # Weather image (A)
            dest_a = out_a / out_name
            save_exr_as_png(str(wf), str(dest_a), target_size)

            # Sunny counterpart (B)
            sf = sunny_by_index.get(idx)
            if sf is not None:
                dest_b = out_b / out_name
                save_exr_as_png(str(sf), str(dest_b), target_size)
            else:
                print(f"  Warning: no sunny counterpart for {wf.name}")
                skipped += 1

            converted += 1

    print(f"  {output_subdir}: {converted} images converted"
          f" ({len(train_weather)} train + {len(test_weather)} test)"
          f"{', ' + str(skipped) + ' missing sunny pairs' if skipped else ''}")


def verify_conversion():
    """Report file counts for all output weather directories."""
    print("=" * 55)
    print("  Multi-weather PNG Conversion — Verification")
    print("=" * 55)
    total = 0
    for weather_dir in sorted(PNG_ROOT.iterdir()):
        if not weather_dir.is_dir():
            continue
        train_a = weather_dir / "train_A"
        train_b = weather_dir / "train_B"
        test_a = weather_dir / "test_A"
        test_b = weather_dir / "test_B"

        ta = len(list(train_a.glob("*.png"))) if train_a.exists() else 0
        tb = len(list(train_b.glob("*.png"))) if train_b.exists() else 0
        tea = len(list(test_a.glob("*.png"))) if test_a.exists() else 0
        teb = len(list(test_b.glob("*.png"))) if test_b.exists() else 0
        total += ta + tb + tea + teb
        status = "✅" if (ta > 0 and tb > 0) else "⚠️"
        print(f"  {status} {weather_dir.name}/")
        print(f"      train_A: {ta:>4d} | train_B: {tb:>4d}")
        print(f"      test_A:  {tea:>4d} | test_B:  {teb:>4d}")
    print("─" * 55)
    print(f"  Total PNG files: {total}")
    print("=" * 55)


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(
        description="Convert WeatherSynthetic EXR → PNG for multi-weather training"
    )
    parser.add_argument("--weather", type=str, default=None,
                        choices=["foggy", "night", "rainy"],
                        help="Which weather to convert (default: all)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify output counts without converting")
    args = parser.parse_args()

    if not HAS_OPENEXR and not args.verify:
        print("Error: OpenEXR package required. Run: pip install OpenEXR")
        sys.exit(1)

    sunny_dir = EXR_ROOT / "sunny"
    if not sunny_dir.exists():
        print(f"Error: sunny EXR dir not found at {sunny_dir}")
        sys.exit(1)

    if args.verify:
        verify_conversion()
        return

    # Determine which weathers to convert
    weathers_to_process = {}
    if args.weather:
        if args.weather == "foggy":
            weathers_to_process["foggy"] = EXR_ROOT / "foggy"
        elif args.weather == "night":
            weathers_to_process["night"] = NIGHT_EXR_DIR
        elif args.weather == "rainy":
            weathers_to_process["rainy"] = EXR_ROOT / "rainy"
    else:
        weathers_to_process["foggy"] = EXR_ROOT / "foggy"
        weathers_to_process["night"] = NIGHT_EXR_DIR

    for exr_name, exr_dir in weathers_to_process.items():
        if not exr_dir.exists():
            print(f"Warning: {exr_dir} not found, skipping {exr_name}")
            continue
        out_name = weather_name(exr_name)
        print(f"Converting {exr_name} ({exr_dir})...")
        convert_weather(exr_dir, sunny_dir, out_name, train_count=400)

    print("\nDone. Run with --verify to check results.")


if __name__ == "__main__":
    main()
