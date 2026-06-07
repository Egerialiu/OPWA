"""
OPWA A1 Training Script — adapted for actual env.

Usage:
    # A1 (full): reconstruction + perception-driven (warmup)
    python scripts/train_opwa_a1.py --output_dir /gz-data/outputs/opwa_a1_full

    # A1 (no-percept): reconstruction only (ablation)
    python scripts/train_opwa_a1.py --percept_weight_max 0.0 \
        --output_dir /gz-data/outputs/opwa_a1_nopercept
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from diffusers import UNet2DConditionModel, AutoencoderKL, LCMScheduler
from transformers import AutoTokenizer, CLIPTextModel, SegformerForSemanticSegmentation

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
    parser = argparse.ArgumentParser(description="Train OPWA A1")
    parser.add_argument("--output_dir", type=str, default="/gz-data/outputs/opwa_a1")
    parser.add_argument("--checkpoint_dir", type=str, default="/gz-data/checkpoints")

    # Data
    parser.add_argument("--data_root", type=str, default="/gz-data/weathersynthetic_street_png")

    # Model
    parser.add_argument("--pretrained_model", type=str, default="stabilityai/sd-turbo")
    parser.add_argument("--perception_model", type=str,
                        default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    parser.add_argument("--prompt", type=str,
                        default="a photo of a street scene, clear weather, high quality")

    # Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--stage1_steps", type=int, default=1000)
    parser.add_argument("--stage2_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_lr", type=float, default=1e-4)
    parser.add_argument("--gate_lr", type=float, default=1e-3)
    parser.add_argument("--forward_mode", type=str, default='dual',
                        choices=['direct', 'denoise', 'dual'])
    parser.add_argument("--lpips_weight", type=float, default=1.0)
    parser.add_argument("--percept_weight_max", type=float, default=0.5)
    parser.add_argument("--warmup_start", type=int, default=500)
    parser.add_argument("--warmup_end", type=int, default=1500)
    parser.add_argument("--gate_reg_weight", type=float, default=1e-3)

    # Device
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mixed_precision", type=str, default="fp16")

    # HF cache
    parser.add_argument("--hf_cache", type=str, default="/gz-data/huggingface_cache")

    # Resume
    parser.add_argument("--resume", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["HF_HOME"] = args.hf_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(args.hf_cache, "hub")

    device = torch.device(args.device)
    logger.info(f"Device: {device}")
    logger.info(f"Data root: {args.data_root}")

    # ── Dataset ──
    data_config = WeatherDatasetConfig(
        weather_synthetic_root=args.data_root,
        image_size=(args.image_size, args.image_size),
        use_pseudo_labels=(args.percept_weight_max > 0),
    )

    # Need perception model first if generating pseudo-labels
    perception_model_for_data = None
    if data_config.use_pseudo_labels:
        logger.info("Loading perception model for pseudo-label generation...")
        perception_model_for_data = SegformerForSemanticSegmentation.from_pretrained(
            args.perception_model,
            cache_dir=os.path.join(args.hf_cache, "hub"),
        )
        perception_model_for_data.to(device)
        perception_model_for_data.eval()
        for p in perception_model_for_data.parameters():
            p.requires_grad = False

    train_loader = create_weather_dataloader(
        data_config, batch_size=args.batch_size, split="train",
        perception_model=perception_model_for_data, device=device,
    )
    eval_loader = create_weather_dataloader(
        data_config, batch_size=1, split="test", shuffle=False,
    )

    logger.info(f"Train: {len(train_loader.dataset)} samples, "
                f"{len(train_loader)} batches")
    logger.info(f"Eval:  {len(eval_loader.dataset)} samples")

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

    # ── Build OPWA A1 ──
    model = OPWA_A1(unet=unet, vae=vae, scheduler=scheduler, gate_init=[0.2, 0.0, -0.12, -0.2])
    model.to(device)
    info = model.get_total_params()
    logger.info(f"OPWA A1: {info['total']:,} total, {info['trainable']:,} trainable")

    # ── Text embeddings ──
    text_encoder.to(device)
    with torch.no_grad():
        tokens = tokenizer(
            args.prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt",
        ).to(device)
        text_embeddings = text_encoder(**tokens).last_hidden_state
    logger.info(f"Text embeddings: {text_embeddings.shape}")

    # ── Perception model for training loss ──
    perception_model = None
    if args.percept_weight_max > 0:
        perception_model = SegformerForSemanticSegmentation.from_pretrained(
            args.perception_model, cache_dir=hf_cache_dir,
        )
        perception_model.to(device)
        perception_model.eval()
        for p in perception_model.parameters():
            p.requires_grad = False
        logger.info("Perception model loaded for training loss")

    # ── Trainer ──
    train_config = TrainingConfig(
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_lr=args.lora_lr,
        gate_lr=args.gate_lr,
        lpips_weight=args.lpips_weight,
        percept_weight_max=args.percept_weight_max,
        warmup_start=args.warmup_start,
        warmup_end=args.warmup_end,
        gate_reg_weight=args.gate_reg_weight,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        device=str(device),
        mixed_precision=args.mixed_precision,
        prompt=args.prompt,
        forward_mode=args.forward_mode,
    )

    trainer = OPWATrainer(
        model=model,
        config=train_config,
        dataloader=train_loader,
        eval_dataloader=eval_loader,
        perception_model=perception_model,
        text_embeddings=text_embeddings,
    )

    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # ── Train ──
    logger.info("=" * 50)
    logger.info(f"Stage 1: {train_config.stage1_steps} steps (reconstruction)")
    logger.info(f"Stage 2: {train_config.stage2_steps} steps (perception-driven)")
    logger.info(f"Total:   {train_config.total_steps} steps")
    logger.info(f"λ_p max: {args.percept_weight_max}")
    logger.info("=" * 50)

    metrics = trainer.train()
    trainer.save_checkpoint()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
