from pathlib import Path
import argparse
import random
import shutil
import sys
from functools import partial
import cv2
import numpy as np
import albumentations as A

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

from src.core import config
from src.core.metadata.dataset_builder import DatasetMetadata

# Available individual photometric and sensor degradation effects
AVAILABLE_EFFECTS = ["downscale", "contrast", "noise", "compression", "motion_blur", "chromatic", "vignette"]

# Presets grouping effects by realistic operational drone causes
EFFECT_GROUPS = {
    "all": AVAILABLE_EFFECTS,
    "transmission": ["downscale", "compression"],             # Video streaming bitrate, digital zoom & packet quantization
    "flight":       ["motion_blur", "chromatic", "vignette"], # Gimbal micro-vibrations, wind gusts, wide-angle lens optics
    "weather":      ["contrast", "vignette", "noise"],        # Atmospheric haze, Rayleigh contrast loss, harsh ambient sunlight
    "sensor":       ["noise", "chromatic", "contrast"]        # CMOS small-sensor noise, color fringing, ISP tone curve
}

# Hardness level presets (light, medium, hard) calibrated for aerial drone inspection
EFFECT_PRESETS = {
    "downscale": {
        "light":  {"scale_range": (0.75, 0.90), "p": 0.3},
        "medium": {"scale_range": (0.60, 0.80), "p": 0.5},
        "hard":   {"scale_range": (0.45, 0.65), "p": 0.7}
    },
    "contrast": {
        "light":  {"brightness_limit": 0.03, "contrast_limit": (-0.10, 0.0), "p": 0.3},
        "medium": {"brightness_limit": 0.06, "contrast_limit": (-0.18, -0.02), "p": 0.5},
        "hard":   {"brightness_limit": 0.10, "contrast_limit": (-0.28, -0.05), "p": 0.7}
    },
    "noise": {
        "light":  {"std_range": (0.01, 0.03), "p": 0.3},
        "medium": {"std_range": (0.02, 0.06), "p": 0.5},
        "hard":   {"std_range": (0.05, 0.10), "p": 0.7}
    },
    "compression": {
        "light":  {"quality_range": (70, 90), "p": 0.4},
        "medium": {"quality_range": (50, 75), "p": 0.6},
        "hard":   {"quality_range": (30, 50), "p": 0.8}
    },
    "motion_blur": {
        "light":  {"blur_limit": (3, 3), "p": 0.2},
        "medium": {"blur_limit": (3, 5), "p": 0.4},
        "hard":   {"blur_limit": (5, 7), "p": 0.6}
    },
    "chromatic": {
        "light":  {"primary_distortion_limit": (-0.008, 0.008), "p": 0.25},
        "medium": {"primary_distortion_limit": (-0.015, 0.015), "p": 0.4},
        "hard":   {"primary_distortion_limit": (-0.030, 0.030), "p": 0.6}
    },
    "vignette": {
        "light":  {"intensity": 0.20, "p": 0.25},
        "medium": {"intensity": 0.35, "p": 0.4},
        "hard":   {"intensity": 0.50, "p": 0.6}
    }
}


def _apply_vignette(img, intensity=0.35, **kwargs):
    """Applies radial optical vignetting (peripheral lens illumination falloff)."""
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - w / 2.0)**2 + (y - h / 2.0)**2) / np.sqrt((w / 2.0)**2 + (h / 2.0)**2)
    mask = 1.0 - intensity * np.clip((dist - 0.5) / 0.5, 0.0, 1.0)
    return (img * mask[..., None]).clip(0, 255).astype(np.uint8)


def resolve_effects(effects_arg):
    """Expands group names (e.g. 'transmission', 'flight') and validates individual effects."""
    if not effects_arg or "all" in effects_arg:
        return EFFECT_GROUPS["all"]

    resolved = []
    for item in effects_arg:
        key = item.lower()
        if key in EFFECT_GROUPS:
            resolved.extend(EFFECT_GROUPS[key])
        elif key in AVAILABLE_EFFECTS:
            resolved.append(key)
        else:
            print(f"⚠️ Unrecognized effect or preset group: '{item}'.")
            print(f"   Available presets: {list(EFFECT_GROUPS.keys())} | Individual: {AVAILABLE_EFFECTS}")

    unique_effects = list(dict.fromkeys(resolved))
    return unique_effects if unique_effects else EFFECT_GROUPS["all"]


def build_pipeline(effects, hardness):
    """Builds Albumentations Compose pipeline for selected effects and hardness level."""
    transforms = []
    summary = {}

    for eff in effects:
        params = EFFECT_PRESETS[eff][hardness]
        summary[eff] = params
        p = params["p"]

        if eff == "downscale":
            transforms.append(A.Downscale(
                scale_range=params["scale_range"],
                interpolation_pair={'downscale': cv2.INTER_AREA, 'upscale': cv2.INTER_LINEAR},
                p=p
            ))
        elif eff == "contrast":
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=params["brightness_limit"],
                contrast_limit=params["contrast_limit"],
                p=p
            ))
        elif eff == "noise":
            transforms.append(A.GaussNoise(std_range=params["std_range"], p=p))
        elif eff == "compression":
            transforms.append(A.ImageCompression(
                quality_range=params["quality_range"],
                compression_type='jpeg',
                p=p
            ))
        elif eff == "motion_blur":
            transforms.append(A.MotionBlur(blur_limit=params["blur_limit"], p=p))
        elif eff == "chromatic":
            transforms.append(A.ChromaticAberration(
                primary_distortion_limit=params["primary_distortion_limit"],
                p=p
            ))
        elif eff == "vignette":
            transforms.append(A.Lambda(
                image=partial(_apply_vignette, intensity=params["intensity"]),
                p=p
            ))

    return A.Compose(transforms), summary


def degrade_dataset(mode="replace", hardness="medium", effects=None, ratio=0.7, dry=False, subset="train"):
    """Orchestrates the degradation process on dataset images and updates metadata."""
    selected_effects = resolve_effects(effects)

    images_dir = config.DATASET_IMAGES / subset
    labels_dir = config.DATASET_LABELS / subset

    if not images_dir.exists():
        print(f"❌ Directory not found: {images_dir}")
        return

    all_images = [
        p for p in images_dir.iterdir()
        if p.suffix.lower() in config.VALID_IMAGE_EXTENSIONS and not p.stem.endswith("_deg")
    ]
    if not all_images:
        print(f"⚠️ No original images found in {images_dir}")
        return

    count = max(1, int(len(all_images) * max(0.0, min(1.0, ratio))))
    selected = random.sample(all_images, count)
    pipeline, summary = build_pipeline(selected_effects, hardness)

    print(f"\n🛠️  Sim-to-Real Degradation: {subset.upper()} | Mode: {mode.upper()} | Hardness: {hardness.upper()}")
    print(f"   Active Effects ({len(selected_effects)}): {', '.join(selected_effects)} | Images: {count}/{len(all_images)} ({ratio*100:.0f}%)")
    if dry:
        print("   ⚠️ DRY-RUN active: simulating in-memory transformations (no disk writes)...\n")

    new_objs, new_bgs = 0, 0
    for idx, img_path in enumerate(selected, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        aug = pipeline(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))['image']
        aug_bgr = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)

        if not dry:
            if mode == "replace":
                cv2.imwrite(str(img_path), aug_bgr)
            else:
                dest_img = images_dir / f"{img_path.stem}_deg{img_path.suffix}"
                dest_lbl = labels_dir / f"{img_path.stem}_deg.txt"
                src_lbl = labels_dir / f"{img_path.stem}.txt"

                cv2.imwrite(str(dest_img), aug_bgr)
                if src_lbl.exists():
                    shutil.copy2(str(src_lbl), str(dest_lbl))
                    with open(dest_lbl, "r", encoding="utf-8") as f:
                        lines = [l.strip() for l in f if l.strip()]
                    new_objs += len(lines)
                    if not lines:
                        new_bgs += 1
                else:
                    dest_lbl.touch()
                    new_bgs += 1

    if dry:
        print(f"✅ [DRY-RUN] Successfully verified and transformed {len(selected)} images in memory (disk intact).\n")
        return

    print(f"✅ Processed {len(selected)} images ({mode.upper()}).")

    # Update metadata for Sim-to-Real experiment traceability
    meta = DatasetMetadata(config.DATASET_METADATA_PATH)
    meta.record_degradation(
        mode=mode,
        subset=subset,
        ratio=ratio,
        processed_count=len(selected),
        total_subset_images=len(all_images),
        active_degradations=summary,
        severity=hardness,
        new_added_images=len(selected) if mode == "augment" else 0,
        new_added_objects=new_objs,
        new_added_backgrounds=new_bgs
    )
    meta.set_timestamp(config.UPDATE_TIMESTAMP_KEY)
    meta.commit()


def main():
    parser = argparse.ArgumentParser(
        description="Simplified Sim-to-Real domain degradation for synthetic datasets"
    )
    parser.add_argument("--mode", choices=["replace", "augment"], default="replace",
                        help="replace: in-place overwrite | augment: creates duplicated _deg copies (default: replace)")
    parser.add_argument("--hardness", choices=["light", "medium", "hard"], default="medium",
                        help="Global degradation hardness level: light, medium, hard (default: medium)")
    parser.add_argument("--effects", nargs="+", default=["all"],
                        help=f"Effects or presets to apply. Presets: {list(EFFECT_GROUPS.keys())} | Individual: {AVAILABLE_EFFECTS} (default: all)")
    parser.add_argument("--ratio", type=float, default=0.7,
                        help="Proportion of subset images to degrade from 0.0 to 1.0 (default: 0.7)")
    parser.add_argument("--subset", choices=["train", "val", "test"], default="train",
                        help="Target subset to degrade (default: train)")
    parser.add_argument("--dry", action="store_true",
                        help="Simulation mode: runs in-memory pipeline without modifying files on disk")

    args = parser.parse_args()
    degrade_dataset(
        mode=args.mode,
        hardness=args.hardness,
        effects=args.effects,
        ratio=args.ratio,
        dry=args.dry,
        subset=args.subset
    )


if __name__ == "__main__":
    main()
