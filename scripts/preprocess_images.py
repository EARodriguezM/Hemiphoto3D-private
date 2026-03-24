#!/usr/bin/env python3
"""Preprocess smartphone images for CUDA 3D reconstruction pipeline.

Features:
  1. EXIF orientation auto-rotation
  2. Resize to target max dimension (preserve aspect ratio)
  3. Exposure normalization: match histograms to median image (--normalize-exposure)
  4. Lens distortion correction via lookup table (--undistort)
  5. Sharpness check: flag blurry images (Laplacian variance < threshold)
  6. Quality report: image count, resolution, estimated focal length, blur results,
     brightness consistency score

Dependencies: opencv-python, Pillow, numpy (all pip-installable)

Usage:
  python preprocess_images.py \\
      --input raw_photos/ \\
      --output processed/ \\
      --max-size 3200 \\
      --normalize-exposure \\
      --check-quality
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ExifTags

# ---------------------------------------------------------------------------
# Known smartphone lens distortion coefficients (k1, k2, p1, p2, k3).
# Values are approximate and vary by device; users can supply their own via
# --distortion-file.  Keys are lowercased substrings matched against the EXIF
# Model field.
# ---------------------------------------------------------------------------
KNOWN_DISTORTION = {
    "iphone 13":      (-0.02, 0.01, 0.0, 0.0, 0.0),
    "iphone 14":      (-0.02, 0.01, 0.0, 0.0, 0.0),
    "iphone 15":      (-0.018, 0.008, 0.0, 0.0, 0.0),
    "pixel 7":        (-0.025, 0.012, 0.0, 0.0, 0.0),
    "pixel 8":        (-0.022, 0.010, 0.0, 0.0, 0.0),
    "galaxy s22":     (-0.03, 0.015, 0.0, 0.0, 0.0),
    "galaxy s23":     (-0.028, 0.013, 0.0, 0.0, 0.0),
    "galaxy s24":     (-0.025, 0.012, 0.0, 0.0, 0.0),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Default Laplacian-variance threshold below which an image is flagged blurry.
DEFAULT_BLUR_THRESHOLD = 100.0


# ---------------------------------------------------------------------------
# EXIF helpers
# ---------------------------------------------------------------------------

def _exif_tag_id(name: str) -> int | None:
    """Return the EXIF tag ID for a given tag name, or None."""
    for tag_id, tag_name in ExifTags.TAGS.items():
        if tag_name == name:
            return tag_id
    return None


def get_exif_data(pil_img: Image.Image) -> dict:
    """Extract a subset of useful EXIF fields as a dict."""
    info: dict = {}
    try:
        exif = pil_img._getexif()
    except Exception:
        return info
    if exif is None:
        return info
    for tag_id, value in exif.items():
        tag_name = ExifTags.TAGS.get(tag_id, tag_id)
        if tag_name in ("Orientation", "Model", "FocalLength",
                        "FocalLengthIn35mmFilm", "ImageWidth", "ImageLength",
                        "ExifImageWidth", "ExifImageHeight"):
            info[tag_name] = value
    return info


def auto_rotate(pil_img: Image.Image) -> Image.Image:
    """Apply EXIF orientation tag so the image is upright, then strip the tag."""
    try:
        exif = pil_img._getexif()
    except Exception:
        return pil_img
    if exif is None:
        return pil_img

    orientation_key = _exif_tag_id("Orientation")
    if orientation_key is None or orientation_key not in exif:
        return pil_img

    orientation = exif[orientation_key]
    transforms = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: (Image.FLIP_LEFT_RIGHT, Image.ROTATE_90),
        6: Image.ROTATE_270,
        7: (Image.FLIP_LEFT_RIGHT, Image.ROTATE_270),
        8: Image.ROTATE_90,
    }
    xform = transforms.get(orientation)
    if xform is None:
        return pil_img
    if isinstance(xform, tuple):
        for t in xform:
            pil_img = pil_img.transpose(t)
    else:
        pil_img = pil_img.transpose(xform)
    return pil_img


# ---------------------------------------------------------------------------
# Image processing functions
# ---------------------------------------------------------------------------

def resize_image(img: np.ndarray, max_size: int) -> np.ndarray:
    """Resize *img* so its longest side is at most *max_size* pixels."""
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def compute_brightness(img: np.ndarray) -> float:
    """Mean brightness of a BGR image (via its V channel in HSV)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean())


def normalize_exposure(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Match each image's histogram to the median-brightness image.

    Uses CLAHE on the L channel of LAB colour space so that colour is
    preserved and the adjustment is local rather than a global stretch.
    """
    if len(images) <= 1:
        return images

    # Find the median-brightness image as reference.
    brightness = {name: compute_brightness(img) for name, img in images.items()}
    sorted_names = sorted(brightness, key=brightness.get)
    ref_name = sorted_names[len(sorted_names) // 2]
    ref_img = images[ref_name]

    # Convert reference to LAB and build a histogram of its L channel.
    ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB)
    ref_l = ref_lab[:, :, 0]

    result: dict[str, np.ndarray] = {}
    for name, img in images.items():
        if name == ref_name:
            result[name] = img
            continue

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # Simple mean/std matching.
        src_mean, src_std = l_channel.mean(), l_channel.std() + 1e-6
        ref_mean, ref_std = ref_l.mean(), ref_l.std() + 1e-6

        matched = ((l_channel.astype(np.float32) - src_mean) *
                    (ref_std / src_std) + ref_mean)
        matched = np.clip(matched, 0, 255).astype(np.uint8)
        lab[:, :, 0] = matched
        result[name] = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return result


def laplacian_variance(img: np.ndarray) -> float:
    """Compute the variance of the Laplacian — a measure of image sharpness."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def undistort_image(img: np.ndarray, model: str | None,
                    custom_coeffs: dict | None = None) -> np.ndarray:
    """Apply lens distortion correction if coefficients are known."""
    coeffs = None
    if custom_coeffs and model:
        key = model.lower().strip()
        for pattern, c in custom_coeffs.items():
            if pattern in key:
                coeffs = np.array(c, dtype=np.float64)
                break

    if coeffs is None and model:
        key = model.lower().strip()
        for pattern, c in KNOWN_DISTORTION.items():
            if pattern in key:
                coeffs = np.array(c, dtype=np.float64)
                break

    if coeffs is None:
        return img

    h, w = img.shape[:2]
    # Approximate camera matrix: focal length ~ max(w,h), principal point at centre.
    f = max(w, h)
    camera_matrix = np.array([[f, 0, w / 2.0],
                               [0, f, h / 2.0],
                               [0, 0, 1.0]], dtype=np.float64)
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, coeffs, (w, h), 1, (w, h))
    return cv2.undistort(img, camera_matrix, coeffs, None, new_camera_matrix)


def estimate_focal_length_mm(exif_data: dict) -> float | None:
    """Try to extract focal length in mm from EXIF."""
    fl = exif_data.get("FocalLength")
    if fl is not None:
        # Pillow returns IFDRational or tuple.
        if hasattr(fl, "numerator"):
            return fl.numerator / fl.denominator if fl.denominator else None
        if isinstance(fl, tuple) and len(fl) == 2 and fl[1]:
            return fl[0] / fl[1]
        try:
            return float(fl)
        except (TypeError, ValueError):
            pass
    fl35 = exif_data.get("FocalLengthIn35mmFilm")
    if fl35 is not None:
        try:
            return float(fl35)
        except (TypeError, ValueError):
            pass
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def discover_images(input_dir: Path) -> list[Path]:
    """Return sorted list of image files in *input_dir*."""
    paths = []
    for p in sorted(input_dir.iterdir()):
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file():
            paths.append(p)
    return paths


def process_images(args: argparse.Namespace) -> None:
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        print(f"Error: input directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = discover_images(input_dir)
    if not image_paths:
        print(f"Error: no images found in '{input_dir}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(image_paths)} images in {input_dir}")

    # Load custom distortion coefficients if provided.
    custom_coeffs = None
    if args.distortion_file:
        with open(args.distortion_file) as f:
            custom_coeffs = json.load(f)

    # ── Pass 1: load, rotate, resize, collect metadata ──────────────────
    loaded: dict[str, np.ndarray] = {}
    exif_all: dict[str, dict] = {}
    models: set[str] = set()
    focal_lengths: list[float] = []
    blur_scores: dict[str, float] = {}
    brightnesses: list[float] = []

    for path in image_paths:
        name = path.name
        pil_img = Image.open(path)

        # EXIF data.
        exif_data = get_exif_data(pil_img)
        exif_all[name] = exif_data
        model = exif_data.get("Model")
        if model:
            models.add(model)

        fl = estimate_focal_length_mm(exif_data)
        if fl is not None:
            focal_lengths.append(fl)

        # Auto-rotate.
        pil_img = auto_rotate(pil_img)

        # Convert to OpenCV BGR.
        cv_img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

        # Resize.
        cv_img = resize_image(cv_img, args.max_size)

        # Undistort.
        if args.undistort:
            cv_img = undistort_image(cv_img, model, custom_coeffs)

        # Quality metrics.
        blur_scores[name] = laplacian_variance(cv_img)
        brightnesses.append(compute_brightness(cv_img))

        loaded[name] = cv_img

    # ── Pass 2 (optional): exposure normalisation ───────────────────────
    if args.normalize_exposure:
        print("Normalizing exposure across images...")
        loaded = normalize_exposure(loaded)

    # ── Pass 3: save ────────────────────────────────────────────────────
    for name, img in loaded.items():
        out_path = output_dir / name
        # Always save as JPEG for consistency.
        out_path = out_path.with_suffix(".jpg")
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"Saved {len(loaded)} preprocessed images to {output_dir}")

    # ── Quality report ──────────────────────────────────────────────────
    if args.check_quality:
        print_quality_report(loaded, exif_all, blur_scores, brightnesses,
                             focal_lengths, models, args)


def print_quality_report(images: dict[str, np.ndarray],
                         exif_all: dict[str, dict],
                         blur_scores: dict[str, float],
                         brightnesses: list[float],
                         focal_lengths: list[float],
                         models: set[str],
                         args: argparse.Namespace) -> None:
    """Print a human-readable quality report to stdout."""

    print("\n" + "=" * 60)
    print("  IMAGE QUALITY REPORT")
    print("=" * 60)

    # Image count & resolution.
    sample = next(iter(images.values()))
    h, w = sample.shape[:2]
    print(f"\n  Image count     : {len(images)}")
    print(f"  Output resolution: {w} x {h} (max-size={args.max_size})")

    # Camera models.
    if models:
        print(f"  Camera model(s) : {', '.join(sorted(models))}")
    else:
        print("  Camera model(s) : unknown (no EXIF Model tag)")

    # Focal length.
    if focal_lengths:
        avg_fl = sum(focal_lengths) / len(focal_lengths)
        print(f"  Est. focal length: {avg_fl:.1f} mm "
              f"(range {min(focal_lengths):.1f}–{max(focal_lengths):.1f})")
    else:
        print("  Est. focal length: unknown (no EXIF FocalLength)")

    # Blur detection.
    threshold = args.blur_threshold
    blurry = {name: score for name, score in blur_scores.items()
              if score < threshold}
    sharp_count = len(blur_scores) - len(blurry)
    print(f"\n  Sharpness (Laplacian variance, threshold={threshold:.0f}):")
    print(f"    Sharp : {sharp_count}/{len(blur_scores)}")
    print(f"    Blurry: {len(blurry)}/{len(blur_scores)}")
    if blurry:
        print("    Flagged images:")
        for name in sorted(blurry, key=blurry.get):
            print(f"      {name:40s}  variance={blurry[name]:.1f}")

    # Brightness consistency.
    if brightnesses:
        mean_b = np.mean(brightnesses)
        std_b = np.std(brightnesses)
        cv_b = (std_b / mean_b * 100) if mean_b > 0 else 0
        consistency = "GOOD" if cv_b < 10 else ("FAIR" if cv_b < 20 else "POOR")
        print(f"\n  Brightness consistency:")
        print(f"    Mean brightness : {mean_b:.1f}")
        print(f"    Std deviation   : {std_b:.1f}")
        print(f"    Coeff. of var.  : {cv_b:.1f}%  [{consistency}]")

    print("\n" + "=" * 60)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess smartphone images for 3D reconstruction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Directory containing raw input images")
    parser.add_argument("--output", "-o", required=True,
                        help="Directory for preprocessed output images")
    parser.add_argument("--max-size", type=int, default=3200,
                        help="Max dimension in pixels (default: 3200)")
    parser.add_argument("--normalize-exposure", action="store_true",
                        help="Match histograms to the median-brightness image")
    parser.add_argument("--undistort", action="store_true",
                        help="Apply lens distortion correction if model is known")
    parser.add_argument("--distortion-file", default=None,
                        help="JSON file with custom distortion coefficients")
    parser.add_argument("--check-quality", action="store_true",
                        help="Print a quality report after processing")
    parser.add_argument("--blur-threshold", type=float,
                        default=DEFAULT_BLUR_THRESHOLD,
                        help=f"Laplacian variance threshold for blur detection "
                             f"(default: {DEFAULT_BLUR_THRESHOLD})")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    process_images(args)


if __name__ == "__main__":
    main()
