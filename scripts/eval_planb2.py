"""
Plan B2 — End-to-end weather-routed evaluation.

Pipeline:
  1. For each test sample: classify weather → route to matching GPPI → restore → SegFormer → mIoU
  2. Supports multiple GPPI modules (rain, fog, night) + passthrough fallback
  3. Reports: per-weather mIoU, overall mIoU, misclassification rate, comparison vs raw

Usage:
    python scripts/eval_planb2.py \
        --gppi_rain /gz-data/checkpoints/gppi_rain/opwa_a1_step_2000.pt \
        --gppi_fog /gz-data/checkpoints/gppi_fog/opwa_a1_step_2000.pt
"""
import torch, sys, json, argparse, os
from collections import defaultdict
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import AutoTokenizer, CLIPTextModel, SegformerForSemanticSegmentation
from opwa.models import OPWA_A1
from opwa.evaluation.metrics import compute_miou
from opwa.training.dataset import WeatherDatasetConfig, WeatherDataset
from opwa.utils.weather_classifier import classify_by_tensor
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--gppi_rain", type=str, default=None, help="GPPI-rain checkpoint path")
parser.add_argument("--gppi_fog", type=str, default=None, help="GPPI-fog checkpoint path")
parser.add_argument("--gppi_night", type=str, default=None, help="GPPI-night checkpoint path")
parser.add_argument("--data_root", type=str, default="/gz-data/weathersynthetic_street_png")
parser.add_argument("--lora_rank", type=int, default=8)
parser.add_argument("--vae_lora_rank", type=int, default=0)
parser.add_argument("--train_d_enc", action="store_true")
parser.add_argument("--output", type=str, default=None, help="Output JSON path")
args = parser.parse_args()

device = torch.device("cuda:0")

# ── Load perception model (stays in VRAM) ──
print("Loading perception model...")
perception = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    cache_dir="/gz-data/huggingface_cache/hub",
).to(device)
perception.eval()

# ── Load text encoder (shared) ──
print("Loading text encoder...")
tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer",
    cache_dir="/gz-data/huggingface_cache/hub")
text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder",
    cache_dir="/gz-data/huggingface_cache/hub").to(device)
with torch.no_grad():
    tokens = tokenizer("a photo of a street scene, clear weather, high quality",
        padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
    text_emb = text_encoder(**tokens).last_hidden_state
del text_encoder

# ── Load dataset (full test set: mixed rain + fog) ──
data_cfg = WeatherDatasetConfig(
    weather_synthetic_root=args.data_root,
    image_size=(512, 512), use_pseudo_labels=True,
)
ds = WeatherDataset(data_cfg, split="test", perception_model=perception, device=device)
print(f"Test samples: {len(ds)}")

# ── Prepare GPPI module loader ──
def load_gppi(checkpoint_path, train_d_enc=False):
    """Load a GPPI checkpoint and return the model."""
    print(f"  Loading backbone for GPPI from {checkpoint_path}...")
    unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet",
        cache_dir="/gz-data/huggingface_cache/hub").to(device)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae",
        cache_dir="/gz-data/huggingface_cache/hub").to(device)
    model = OPWA_A1(unet, vae, lora_rank=args.lora_rank,
                     vae_lora_rank=args.vae_lora_rank,
                     train_d_enc=train_d_enc).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if "trainable_state_dict" in state:
        model.load_state_dict(state["trainable_state_dict"], strict=False)
    else:
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
    model.eval()
    return model, unet, vae


def unload_gppi(model, unet, vae):
    """Unload GPPI model to free VRAM."""
    del model, unet, vae
    torch.cuda.empty_cache()


# ── Build checkpoint map ──
checkpoints = {}
for weather in ["rain", "fog", "night"]:
    ckpt = getattr(args, f"gppi_{weather}")
    if ckpt:
        checkpoints[weather] = ckpt
        print(f"  Registered GPPI-{weather}: {ckpt}")
if not checkpoints:
    print("ERROR: no GPPI checkpoints provided!")
    sys.exit(1)

# ── Phase 1: Classify all test samples and group ──
print("\n=== Phase 1: Classifying test samples ===")
sample_buckets = defaultdict(list)  # weather → [(idx, degraded, label)]
all_raw, all_label = [], []

with torch.no_grad():
    for i in range(len(ds)):
        s = ds[i]
        deg = s["degraded"].unsqueeze(0).to(device)
        clean = s["clean"].unsqueeze(0).to(device)
        label = s["label"].unsqueeze(0).to(device)
        all_label.append(label)

        # Classify weather
        weather = classify_by_tensor(deg, clean)
        sample_buckets[weather].append((i, deg, label))

        # Raw mIoU
        out = perception(pixel_values=deg)
        logits = F.interpolate(out.logits, size=label.shape[-2:],
                                mode="bilinear", align_corners=False)
        raw_miou = float(compute_miou(logits, label, 19)["mIoU"])
        all_raw.append(raw_miou)
        print(f"  [{i+1}/{len(ds)}] classified as {weather:6s}  raw mIoU={raw_miou:.2%}")

    print(f"\n  Weather distribution: { {k: len(v) for k, v in sorted(sample_buckets.items())} }")

# ── Phase 2: Process each weather bucket with its GPPI module ──
print("\n=== Phase 2: Weather-routed GPPI restoration ===")
results = {}  # idx → {"pred_weather": str, "raw_miou": float, "gppi_miou": float, "routed_to": str}

for weather, bucket in sample_buckets.items():
    if weather in checkpoints:
        print(f"\n  Processing {weather} ({len(bucket)} samples) with GPPI-{weather}...")
        model, unet, vae = load_gppi(checkpoints[weather], train_d_enc=args.train_d_enc)

        for idx, deg, label in bucket:
            with torch.no_grad():
                t = torch.full((1,), 1, dtype=torch.long, device=device)
                restored = model(deg, t, text_emb)["reconstructed"]
                out = perception(pixel_values=restored)
                logits = F.interpolate(out.logits, size=label.shape[-2:],
                                        mode="bilinear", align_corners=False)
                gppi_miou = float(compute_miou(logits, label, 19)["mIoU"])

            results[idx] = {
                "pred_weather": weather,
                "gppi_miou": round(gppi_miou, 4),
                "routed_to": weather,
            }
            print(f"    [{idx+1}] {weather:6s}  gppi mIoU={gppi_miou:.2%}")

        unload_gppi(model, unet, vae)
    else:
        # No GPPI for this weather → passthrough
        print(f"\n  {weather}: no checkpoint, using passthrough")
        for idx, deg, label in bucket:
            with torch.no_grad():
                out = perception(pixel_values=deg)
                logits = F.interpolate(out.logits, size=label.shape[-2:],
                                        mode="bilinear", align_corners=False)
                gppi_miou = float(compute_miou(logits, label, 19)["mIoU"])
            results[idx] = {
                "pred_weather": weather,
                "gppi_miou": round(gppi_miou, 4),
                "routed_to": "passthrough",
            }
            print(f"    [{idx+1}] {weather:6s}  passthrough mIoU={gppi_miou:.2%}")

# ── Phase 3: Report ──
print("\n" + "=" * 70)
print("Plan B2 RESULTS")
print("=" * 70)

# Group results by predicted weather
by_weather = defaultdict(list)
total_raw, total_gppi = 0.0, 0.0
for idx in sorted(results.keys()):
    w = results[idx]["pred_weather"]
    r = all_raw[idx]
    g = results[idx]["gppi_miou"]
    by_weather[w].append((r, g))
    total_raw += r
    total_gppi += g

for w in ["rain", "fog", "night"]:
    if w in by_weather:
        rr = [x[0] for x in by_weather[w]]
        gg = [x[1] for x in by_weather[w]]
        n = len(rr)
        ar = sum(rr) / n
        ag = sum(gg) / n
        print(f"  {w:6s}: n={n:2d}  raw={ar:.2%}  plan-b2={ag:.2%}  Δ={ag-ar:+.2%}")

n_total = len(results)
print(f"  {'all':6s}: n={n_total:2d}  raw={total_raw/n_total:.2%}  "
      f"plan-b2={total_gppi/n_total:.2%}  "
      f"Δ={total_gppi/n_total - total_raw/n_total:+.2%}")

# Summary
print("\n" + "-" * 70)
print("SUMMARY: Plan B2 vs Best OPWA A1 (v6f-500 = 46.69%)")
print(f"  Plan B2 mIoU:   {total_gppi/n_total:.2%}")
print(f"  OPWA A1 best:   46.69%")
print(f"  Δ vs A1 best:   {total_gppi/n_total - 0.4669:+.2%}")
print(f"  Δ vs raw:       {total_gppi/n_total - 0.4526:+.2%}")

# Save
if args.output:
    out = {
        "n": n_total,
        "raw_miou": round(total_raw / n_total, 4),
        "planb2_miou": round(total_gppi / n_total, 4),
        "per_weather": {},
    }
    for w, bucket in by_weather.items():
        out["per_weather"][w] = {
            "n": len(bucket),
            "raw_miou": round(sum(x[0] for x in bucket) / len(bucket), 4),
            "planb2_miou": round(sum(x[1] for x in bucket) / len(bucket), 4),
        }
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"\nResults saved to {args.output}")
