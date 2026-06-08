"""
Evaluate a single GPPI checkpoint on its matching weather test set.

Usage:
    # Evaluate GPPI-fog
    python scripts/eval_gppi.py \
        --checkpoint /gz-data/checkpoints/gppi_fog/opwa_a1_step_2000.pt \
        --weather_subset fog

    # Evaluate GPPI-rain
    python scripts/eval_gppi.py \
        --checkpoint /gz-data/checkpoints/gppi_rain/opwa_a1_step_2000.pt \
        --weather_subset rain
"""
import torch, sys, json, argparse, os
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import AutoTokenizer, CLIPTextModel, SegformerForSemanticSegmentation
from opwa.models import OPWA_A1
from opwa.evaluation.metrics import compute_miou
from opwa.training.dataset import WeatherDatasetConfig, WeatherDataset
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--weather_subset", type=str, required=True, choices=["rain", "fog", "night"])
parser.add_argument("--data_root", type=str, default="/gz-data/weathersynthetic_street_png")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--lora_rank", type=int, default=8)
parser.add_argument("--vae_lora_rank", type=int, default=0)
parser.add_argument("--train_d_enc", action="store_true")
parser.add_argument("--branch_type", type=str, default="d_enc", choices=["d_enc", "dcp", "noise"])
args = parser.parse_args()

device = torch.device("cuda:0")

# ── Load backbone ──
print("Loading backbone...")
unet = UNet2DConditionModel.from_pretrained("stabilityai/sd-turbo", subfolder="unet",
    cache_dir="/gz-data/huggingface_cache/hub").to(device)
vae = AutoencoderKL.from_pretrained("stabilityai/sd-turbo", subfolder="vae",
    cache_dir="/gz-data/huggingface_cache/hub").to(device)
tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer",
    cache_dir="/gz-data/huggingface_cache/hub")
text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder",
    cache_dir="/gz-data/huggingface_cache/hub").to(device)

with torch.no_grad():
    tokens = tokenizer("a photo of a street scene, clear weather, high quality",
        padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
    text_emb = text_encoder(**tokens).last_hidden_state

# ── Load model ──
print(f"Loading checkpoint from {args.checkpoint}...")
model = OPWA_A1(unet, vae, lora_rank=args.lora_rank, vae_lora_rank=args.vae_lora_rank,
                 train_d_enc=args.train_d_enc, branch_type=args.branch_type).to(device)
state = torch.load(args.checkpoint, map_location=device)
if "trainable_state_dict" in state:
    model.load_state_dict(state["trainable_state_dict"], strict=False)
    print(f"Loaded trainable_state_dict (step {state.get('step', '?')})")
else:
    model.load_state_dict(state.get("model_state_dict", state), strict=False)
model.eval()
gate_vals = [float(v) for v in model.gate.get_gate_values()]
print(f"Gate values: {gate_vals}")

# ── Perception model ──
perception = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    cache_dir="/gz-data/huggingface_cache/hub",
).to(device)
perception.eval()

# ── Dataset (weather-specific) ──
data_cfg = WeatherDatasetConfig(
    weather_synthetic_root=args.data_root,
    weather_subset=args.weather_subset,
    image_size=(512, 512), use_pseudo_labels=True,
)
ds = WeatherDataset(data_cfg, split="test", perception_model=perception, device=device)

# ── Evaluate ──
raw_mious, opwa_mious = [], []
with torch.no_grad():
    for i in range(len(ds)):
        s = ds[i]
        deg = s["degraded"].unsqueeze(0).to(device)
        label = s["label"].unsqueeze(0).to(device)

        # Raw
        out = perception(pixel_values=deg)
        logits = F.interpolate(out.logits, size=label.shape[-2:], mode="bilinear", align_corners=False)
        raw_miou = float(compute_miou(logits, label, 19)["mIoU"])

        # GPPI
        t = torch.full((1,), 1, dtype=torch.long, device=device)
        restored = model(deg, t, text_emb)["reconstructed"]
        out2 = perception(pixel_values=restored)
        logits2 = F.interpolate(out2.logits, size=label.shape[-2:], mode="bilinear", align_corners=False)
        gppi_miou = float(compute_miou(logits2, label, 19)["mIoU"])

        raw_mious.append(raw_miou)
        opwa_mious.append(gppi_miou)
        print(f"  [{i+1}/{len(ds)}] raw={raw_miou:.2%}  gppi={gppi_miou:.2%}  Δ={gppi_miou-raw_miou:+.2%}")

# ── Report ──
ar = sum(raw_mious) / len(raw_mious)
ao = sum(opwa_mious) / len(opwa_mious)
print(f"\n=== GPPI-{args.weather_subset} RESULTS ===")
print(f"  n={len(raw_mious)}  raw={ar:.2%}  gppi={ao:.2%}  Δ={ao-ar:+.2%}")
print(f"  Gate: {gate_vals}")

out = {
    "weather_subset": args.weather_subset,
    "n": len(raw_mious),
    "raw_miou": round(ar, 4),
    "gppi_miou": round(ao, 4),
    "gate": gate_vals,
}
out_path = os.path.join(os.path.dirname(args.checkpoint), "eval_results.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
json.dump(out, open(out_path, "w"), indent=2)
print(f"\nResults saved to {out_path}")
