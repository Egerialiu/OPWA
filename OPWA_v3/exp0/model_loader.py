import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import sys
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from config import DAV2_REPO, DAV2_CKPT, DEVICE


def load_segformer():
    """Load SegFormer-B0 finetuned on Cityscapes 1024x1024.
    Returns (model, processor)."""
    model = SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/segformer-b0-finetuned-cityscapes-1024-1024'
    ).eval().to(DEVICE)
    processor = SegformerImageProcessor.from_pretrained(
        'nvidia/segformer-b0-finetuned-cityscapes-1024-1024'
    )
    return model, processor


def load_depth_anything():
    """Load Depth Anything V2 Small.
    Returns depth_model (callable via .infer_image())."""
    sys.path.insert(0, DAV2_REPO)
    from depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
        'vits': {
            'encoder': 'vits',
            'features': 64,
            'out_channels': [48, 96, 192, 384],
        }
    }
    depth_model = DepthAnythingV2(**model_configs['vits'])
    depth_model.load_state_dict(
        torch.load(DAV2_CKPT, map_location='cpu')
    )
    depth_model = depth_model.eval().to(DEVICE)
    return depth_model
