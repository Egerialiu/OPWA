"""
Evaluate OPWA A1 checkpoint on WeatherSynthetic test set.
Reports mIoU per weather type (rain/fog/night) and overall.
"""
import torch, sys, json, argparse
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import AutoTokenizer, CLIPTextModel, SegformerForSemanticSegmentation
from opwa.models import OPWA_A1
from opwa.evaluation.metrics import compute_miou
from opwa.training.dataset import WeatherDatasetConfig, WeatherDataset
from opwa.utils.weather_classifier import classify_by_tensor
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
parser.add_argument("--data_root", type=str, default="/gz-data/weathersynthetic_street_png")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--model_type", type=str, default="opwa_a1")
parser.add_argument("--lora_rank", type=int, default=8)
parser.add_argument("--vae_lora_rank", type=int, default=4)
parser.add_argument("--train_d_enc", action="store_true", help="D-Enc was trainable")
parser.add_argument("--noise_probe", action="store_true", help="Noise probe mode")
parser.add_argument("--use_conditional_gate", action="store_true", help="Conditional gate (A2)")
parser.add_argument("--output_tag", type=str, default=None, help="Tag for output JSON")
args = parser.parse_args()

device = torch.device("cuda:0")
DATA = args.data_root
CKPT = args.checkpoint
LORA_RANK = args.lora_rank
VAE_LORA_RANK = args.vae_lora_rank
TRAIN_D_ENC = args.train_d_enc
NOISE_PROBE = args.noise_probe
USE_CONDITIONAL_GATE = args.use_conditional_gate

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
print("Loading OPWA A1 checkpoint...")
model = OPWA_A1(unet, vae, lora_rank=LORA_RANK, vae_lora_rank=VAE_LORA_RANK,
                 train_d_enc=TRAIN_D_ENC, noise_probe=NOISE_PROBE,
                 use_conditional_gate=USE_CONDITIONAL_GATE).to(device)
state = torch.load(CKPT, map_location=device)
if "trainable_state_dict" in state:
    model.load_state_dict(state["trainable_state_dict"], strict=False)
    print(f"Loaded trainable_state_dict (step {state.get('step', '?')})")
else:
    model.load_state_dict(state.get("model_state_dict", state), strict=False)
model.eval()
gate_vals = [float(v) for v in model.gate.get_gate_values()] if hasattr(model.gate, 'get_gate_values') else []
print(f"Gate values: {gate_vals}" if gate_vals else "Gate: Conditional (per-sample)")

# ── Perception model ──
perception = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    cache_dir="/gz-data/huggingface_cache/hub",
).to(device)
perception.eval()

# ── Dataset ──
data_cfg = WeatherDatasetConfig(
    weather_synthetic_root=DATA, image_size=(512,512), use_pseudo_labels=True,
)
ds = WeatherDataset(data_cfg, split="test", perception_model=perception, device=device)

# ── Evaluate ──
results = {"rain": {"raw": [], "opwa": []},
           "fog": {"raw": [], "opwa": []},
           "night": {"raw": [], "opwa": []}}

with torch.no_grad():
    for i in range(len(ds)):
        s = ds[i]
        deg = s["degraded"].unsqueeze(0).to(device)
        clean = s["clean"].unsqueeze(0).to(device)
        label = s["label"].unsqueeze(0).to(device)

        weather = classify_by_tensor(deg, clean)

        # Raw (degraded → perception)
        out = perception(pixel_values=deg)
        logits = F.interpolate(out.logits, size=label.shape[-2:], mode="bilinear", align_corners=False)
        raw_miou = float(compute_miou(logits, label, 19)["mIoU"])
        results[weather]["raw"].append(raw_miou)

        # OPWA (restored → perception)
        t_eval = torch.full((1,), 1, dtype=torch.long, device=device)
        opwa_out = model(deg, t_eval, text_emb)["reconstructed"]
        out2 = perception(pixel_values=opwa_out)
        logits2 = F.interpolate(out2.logits, size=label.shape[-2:], mode="bilinear", align_corners=False)
        opwa_miou = float(compute_miou(logits2, label, 19)["mIoU"])
        results[weather]["opwa"].append(opwa_miou)

        print(f"  [{i+1}/{len(ds)}] {weather:6s}  raw={raw_miou:.2%}  opwa={opwa_miou:.2%}")

# ── Report ──
print("\n" + "="*60)
print("A1-no-percept RESULTS")
print("="*60)

all_raw, all_opwa = [], []
for w in ["rain", "fog", "night"]:
    r = results[w]
    if not r["raw"]:
        print(f"  {w}: no samples")
        continue
    avg_raw = sum(r["raw"]) / len(r["raw"])
    avg_opwa = sum(r["opwa"]) / len(r["opwa"])
    all_raw.extend(r["raw"])
    all_opwa.extend(r["opwa"])
    print(f"  {w:6s}: n={len(r['raw']):2d}  raw={avg_raw:.2%}  opwa={avg_opwa:.2%}  imprv={avg_opwa-avg_raw:+.2%}")

ar = sum(all_raw) / len(all_raw)
ao = sum(all_opwa) / len(all_opwa)
print(f"  {'all':6s}: n={len(all_raw):2d}  raw={ar:.2%}  opwa={ao:.2%}  imprv={ao-ar:+.2%}")
if gate_vals:
    print(f"\nGate: {gate_vals}")
else:
    print("\nGate: conditional (per-sample)")

# Save
out = {"gate": gate_vals, "conditional_gate": USE_CONDITIONAL_GATE, "per_weather": {}, "overall": {}}
for w in ["rain", "fog", "night"]:
    out["per_weather"][w] = {"n": len(results[w]["raw"]),
                              "raw_miou": round(sum(results[w]["raw"])/len(results[w]["raw"]), 4) if results[w]["raw"] else None,
                              "opwa_miou": round(sum(results[w]["opwa"])/len(results[w]["opwa"]), 4) if results[w]["opwa"] else None}
out["overall"] = {"n": len(all_raw),
                   "raw_miou": round(ar, 4),
                   "opwa_miou": round(ao, 4)}
output_tag = args.output_tag or os.path.basename(os.path.dirname(CKPT))
json.dump(out, open(f"/gz-data/checkpoints/{output_tag}/eval_results.json", "w"), indent=2)
print("\nResults saved to eval_results.json")
