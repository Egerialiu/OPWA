"""
Evaluate A1-no-percept checkpoint on WeatherSynthetic test set.
Reports mIoU per weather type (rain/fog/night) and overall.
"""
import torch, sys, json
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import AutoTokenizer, CLIPTextModel, SegformerForSemanticSegmentation
from opwa.models import OPWA_A1
from opwa.evaluation.metrics import compute_miou
from opwa.training.dataset import WeatherDatasetConfig, WeatherDataset
from opwa.utils.weather_classifier import classify_by_tensor
import torch.nn.functional as F

device = torch.device("cuda:0")
DATA = "/gz-data/weathersynthetic_street_png"
CKPT = "/gz-data/checkpoints/opwa_a1_nopercept/opwa_a1_step_2000.pt"

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
model = OPWA_A1(unet, vae).to(device)
state = torch.load(CKPT, map_location=device)
model.load_state_dict(state["model_state_dict"])
model.eval()
gate_vals = [float(v) for v in model.gate.get_gate_values()]
print(f"Gate values: {gate_vals}")

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
print(f"\nGate: {gate_vals}")

# Save
out = {"gate": gate_vals, "per_weather": {}, "overall": {}}
for w in ["rain", "fog", "night"]:
    out["per_weather"][w] = {"n": len(results[w]["raw"]),
                              "raw_miou": round(sum(results[w]["raw"])/len(results[w]["raw"]), 4) if results[w]["raw"] else None,
                              "opwa_miou": round(sum(results[w]["opwa"])/len(results[w]["opwa"]), 4) if results[w]["opwa"] else None}
out["overall"] = {"n": len(all_raw),
                   "raw_miou": round(ar, 4),
                   "opwa_miou": round(ao, 4)}
json.dump(out, open("/gz-data/checkpoints/opwa_a1_nopercept/eval_results.json", "w"), indent=2)
print("\nResults saved to eval_results.json")
