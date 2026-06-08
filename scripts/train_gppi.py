"""
GPPI single-weather training script.

Trains one GPPI module (A1-equivalent: D-Enc + StaticGate + LoRA) on a
single weather subset (rain/fog/night). Designed for Plan B2 — weather-routed
architecture where each weather type gets its own specialized module.

Usage:
    # Train GPPI-fog
    python scripts/train_gppi.py --weather_subset fog

    # Train GPPI-rain (500 steps smoke test)
    python scripts/train_gppi.py --weather_subset rain --stage1_steps 500
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from diffusers import UNet2DConditionModel, AutoencoderKL, LCMScheduler
from transformers import AutoTokenizer, CLIPTextModel

from opwa.models import OPWA_A1
from opwa.training import OPWATrainer, TrainingConfig, WeatherDatasetConfig
from opwa.training.dataset import create_weather_dataloader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train GPPI single-weather module")

    # ── Required ──
    parser.add_argument("--weather_subset", type=str, required=True,
                        choices=["rain", "fog", "night"],
                        help="Weather type to train on")

    # ── Paths ──
    parser.add_argument("--data_root", type=str, default="/gz-data/weathersynthetic_street_png")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--hf_cache", type=str, default="/gz-data/huggingface_cache")

    # ── Model ──
    parser.add_argument("--pretrained_model", type=str, default="stabilityai/sd-turbo")
    parser.add_argument("--prompt", type=str,
                        default="a photo of a street scene, clear weather, high quality")

    # ── Training ──
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--stage1_steps", type=int, default=2000)
    parser.add_argument("--stage2_steps", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--vae_lora_rank", type=int, default=0)
    parser.add_argument("--lora_lr", type=float, default=5e-6)
    parser.add_argument("--gate_lr", type=float, default=5e-4)
    parser.add_argument("--lpips_weight", type=float, default=5.0)
    parser.add_argument("--gan_weight", type=float, default=0)
    parser.add_argument("--percept_weight_max", type=float, default=0)
    parser.add_argument("--gate_reg_weight", type=float, default=1e-3)
    parser.add_argument("--d_enc_weight_decay", type=float, default=1e-4)
    parser.add_argument("--gate_init", type=float, nargs=4, default=[-2.0, -2.0, -2.0, -2.0])

    # ── D-Enc ──
    parser.add_argument("--train_d_enc", action="store_true",
                        help="Unfreeze D-Enc and train jointly")
    parser.add_argument("--d_enc_dropout", type=float, default=0.2,
                        help="Dropout2d prob for D-Enc features")
    parser.add_argument("--branch_type", type=str, default="d_enc",
                        choices=["d_enc", "dcp", "noise"],
                        help="Branch feature source: d_enc, dcp, or noise")

    # ── Device ──
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mixed_precision", type=str, default="fp16")

    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["HF_HOME"] = args.hf_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(args.hf_cache, "hub")

    # Auto-generate output/checkpoint dirs if not specified
    subset = args.weather_subset
    output_dir = args.output_dir or f"/gz-data/outputs/gppi_{subset}"
    checkpoint_dir = args.checkpoint_dir or f"/gz-data/checkpoints/gppi_{subset}"

    device = torch.device(args.device)
    logger.info(f"Device: {device}")
    logger.info(f"Weather subset: {subset}")
    logger.info(f"Data root: {args.data_root}/{subset}/")
    logger.info(f"Output dir: {output_dir}")

    # ── Dataset ──
    data_config = WeatherDatasetConfig(
        weather_synthetic_root=args.data_root,
        weather_subset=subset,
        image_size=(args.image_size, args.image_size),
        use_pseudo_labels=False,  # GPPI trains with pixel loss only
    )
    train_loader = create_weather_dataloader(
        data_config, batch_size=args.batch_size, split="train",
    )
    logger.info(f"Train: {len(train_loader.dataset)} samples, "
                f"{len(train_loader)} batches")

    # ── Load SD-Turbo ──
    model_id = args.pretrained_model
    hf_cache_dir = os.path.join(args.hf_cache, "hub")
    logger.info(f"Loading {model_id}...")
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", cache_dir=hf_cache_dir,
    )
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", cache_dir=hf_cache_dir,
    )
    scheduler = LCMScheduler.from_pretrained(
        model_id, subfolder="scheduler", cache_dir=hf_cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, subfolder="tokenizer", cache_dir=hf_cache_dir,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", cache_dir=hf_cache_dir,
    )
    logger.info(f"UNet: {sum(p.numel() for p in unet.parameters()):,} params")

    # ── Build OPWA A1 (GPPI-equivalent: D-Enc + StaticGate + LoRA) ──
    model = OPWA_A1(unet=unet, vae=vae, scheduler=scheduler,
                     gate_init=args.gate_init,
                     noise_probe=False,
                     lora_rank=args.lora_rank,
                     vae_lora_rank=args.vae_lora_rank,
                     train_d_enc=args.train_d_enc,
                     branch_type=args.branch_type)

    if args.d_enc_dropout > 0:
        model.set_d_enc_dropout(args.d_enc_dropout)
        logger.info(f"  D-Enc Dropout2d(p={args.d_enc_dropout}) applied")

    model.to(device)
    info = model.get_total_params()
    logger.info(f"GPPI-{subset}: {info['total']:,} total, {info['trainable']:,} trainable")

    # ── Text embeddings ──
    text_encoder.to(device)
    with torch.no_grad():
        tokens = tokenizer(
            args.prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        ).to(device)
        text_embeddings = text_encoder(**tokens).last_hidden_state
    logger.info(f"Text embeddings: {text_embeddings.shape}")

    # ── Trainer ──
    train_config = TrainingConfig(
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_lr=args.lora_lr,
        gate_lr=args.gate_lr,
        lpips_weight=args.lpips_weight,
        gan_weight=args.gan_weight,
        percept_weight_max=args.percept_weight_max,
        warmup_start=0,
        warmup_end=1,  # Must be != warmup_start to avoid div-by-zero
        gate_reg_weight=args.gate_reg_weight,
        d_enc_weight_decay=args.d_enc_weight_decay,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        total_steps=args.stage1_steps + args.stage2_steps,
        device=str(device),
        mixed_precision=args.mixed_precision,
        prompt=args.prompt,
    )

    trainer = OPWATrainer(
        model=model,
        config=train_config,
        dataloader=train_loader,
        eval_dataloader=None,
        perception_model=None,
        text_embeddings=text_embeddings,
    )

    # ── Train ──
    logger.info("=" * 50)
    logger.info(f"Training GPPI-{subset}: {train_config.total_steps} steps")
    logger.info("=" * 50)

    metrics = trainer.train()
    trainer.save_checkpoint()

    logger.info(f"GPPI-{subset} training complete!")
    if metrics["gate_vals"]:
        final_gate = metrics["gate_vals"][-1]
        logger.info(f"Final gate values: {final_gate}")


if __name__ == "__main__":
    main()
