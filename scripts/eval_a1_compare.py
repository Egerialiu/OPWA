"""
Quick A1 comparison: no-percept vs full, on WeatherSynthetic pseudo-labels.
"""
"""Quick A1 comparison: no-percept vs full, on WeatherSynthetic pseudo-labels."""
import os, sys, json, logging
import torch
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import AutoTokenizer, CLIPTextModel, SegformerForSemanticSegmentation
from opwa.models import OPWA_A1
from opwa.evaluation import Evaluator, compute_recovery_rate
from opwa.training.dataset import WeatherDatasetConfig, create_weather_dataloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda:0")
DATA = "/gz-data/weathersynthetic_street_png"
CKPTS = {
    "no-percept": "/gz-data/checkpoints/opwa_a1_nopercept/opwa_a1_step_2000.pt",
    "full":       "/gz-data/checkpoints/opwa_a1_full/opwa_a1_step_2000.pt",
}

# ── Load perception model ──
perception = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
    cache_dir="/gz-data/huggingface_cache/hub",
).to(device)
perception.eval()
for p in perception.parameters(): p.requires_grad = False

evaluator = Evaluator(perception, num_classes=19)

# ── Dataset with pseudo-labels ──
data_cfg = WeatherDatasetConfig(
    weather_synthetic_root=DATA, image_size=(512,512), use_pseudo_labels=True,
)
test_loader = create_weather_dataloader(
    data_cfg, batch_size=1, split="test", shuffle=False,
    perception_model=perception, device=device,
)

# ── Precompute text embeddings ──
logger.info("Loading SD-Turbo backbone...")
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
    text_emb = text_encoder(**tokens).last_hidden_state  # (1, 77, 1024)

# ── Evaluate ──
results = {}
for tag, ckpt_path in CKPTS.items():
    logger.info(f"\n{'='*60}\nEvaluating {tag}...\n{'='*60}")
    model = OPWA_A1(unet=unet, vae=vae).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    gate_vals = model.gate.get_gate_values()
    logger.info(f"Gate: {gate_vals}")

    # Raw
    raw = evaluator.evaluate(model, test_loader, device, opwa_enabled=False)

    # OPWA — inject text embeddings into batch
    opwa = evaluator.evaluate(model, test_loader, device, opwa_enabled=True,
                               text_embedding=text_emb)

    # Clean reference (model on clean images = upper bound)
    clean_ckpt = evaluator.evaluate(model, test_loader, device, opwa_enabled=False,
                                     use_clean_images=True)
    clean_miou = clean_ckpt["mIoU"]
    rr = compute_recovery_rate(opwa["mIoU"], clean_miou)
    raw_rr = compute_recovery_rate(raw["mIoU"], clean_miou)

    logger.info(f"  Clean mIoU: {clean_miou:.2%}")
    logger.info(f"  Raw mIoU:   {raw['mIoU']:.2%}  (r={raw_rr:.1f}%)")
    logger.info(f"  OPWA mIoU:  {opwa['mIoU']:.2%}  (r={rr:.1f}%)")
    logger.info(f"  Imprv:      +{opwa['mIoU']-raw['mIoU']:.2%}")

    results[tag] = {
        "gate": [float(v) for v in gate_vals],
        "clean_miou": float(clean_miou),
        "raw_miou": float(raw["mIoU"]),
        "opwa_miou": float(opwa["mIoU"]),
        "raw_recovery": float(raw_rr),
        "opwa_recovery": float(rr),
    }

print("\n" + "="*60)
print("COMPARISON SUMMARY")
print("="*60)
for tag, r in results.items():
    print(f"  {tag:12s}: clean={r['clean_miou']:.2%} raw={r['raw_miou']:.2%} → "
          f"opwa={r['opwa_miou']:.2%} (r={r['opwa_recovery']:.1f}%) gate={r['gate']}")
