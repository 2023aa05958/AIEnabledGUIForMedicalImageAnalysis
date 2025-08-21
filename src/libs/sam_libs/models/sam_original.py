"""
Original SAM (ViT-Base) model support for brain tumor segmentation
This module provides compatibility with original SAM models while maintaining
the same interface as SAM 2.1 models.
"""

import numpy as np
import torch
from typing import Optional

try:
    # Original SAM imports
    from segment_anything import SamPredictor, sam_model_registry
    from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
    SAM_ORIGINAL_AVAILABLE = True
except ImportError:
    print("⚠️  Original SAM not installed. Install with: pip install segment-anything")
    SAM_ORIGINAL_AVAILABLE = False

# Original SAM model configurations
SAM_ORIGINAL_MODELS = {
    "sam_vit_base": {
        "model_type": "vit_b",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "description": "ViT-Base SAM model"
    },
    "sam_vit_large": {
        "model_type": "vit_l", 
        "checkpoint_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "description": "ViT-Large SAM model"
    },
    "sam_vit_huge": {
        "model_type": "vit_h",
        "checkpoint_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", 
        "description": "ViT-Huge SAM model"
    },
    # Your custom trained model
    "brain_tumor_sam_vit_base": {
        "model_type": "vit_b",
        "local_path": "./sam_libs/models/model_weights/brain_tumor_sam_vit_base50.pth",
        "description": "Custom trained SAM ViT-Base for brain tumor segmentation"
    }
}


class OriginalSAM:
    """
    Wrapper for original SAM models with interface compatible with SAM 2.1 implementation
    """
    
    def __init__(self, sam_type: str = "sam_vit_base", ckpt_path: Optional[str] = None, device: str = "cuda"):
        if not SAM_ORIGINAL_AVAILABLE:
            raise ImportError("Original SAM not available. Please install segment-anything package.")
            
        self.sam_type = sam_type
        self.ckpt_path = ckpt_path
        self.device = device
        
        # Build and load the model
        self.model = self._build_model()
        self.predictor = SamPredictor(self.model)
        self.mask_generator = SamAutomaticMaskGenerator(self.model)
        
        print(f"✅ Original SAM model loaded: {sam_type}")
    
    def _build_model(self):
        """Build the original SAM model"""
        if self.sam_type not in SAM_ORIGINAL_MODELS:
            raise ValueError(f"Unsupported SAM type: {self.sam_type}")
        
        model_config = SAM_ORIGINAL_MODELS[self.sam_type]
        model_type = model_config["model_type"]
        
        # Load checkpoint path
        if self.ckpt_path:
            checkpoint_path = self.ckpt_path
            print(f"🔧 Loading custom model from: {checkpoint_path}")
        elif "local_path" in model_config:
            checkpoint_path = model_config["local_path"]
            print(f"🧠 Loading brain tumor model from: {checkpoint_path}")
        else:
            # Download default model
            checkpoint_path = model_config["checkpoint_url"]
            print(f"📥 Downloading model from: {checkpoint_path}")
        
        # Check if local file exists
        if checkpoint_path.startswith("./") and not checkpoint_path.startswith("http"):
            import os
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"❌ Model file not found: {checkpoint_path}\n"
                    f"💡 Please ensure your brain tumor model is placed at the correct location.\n"
                    f"🔍 Looking for: {os.path.abspath(checkpoint_path)}"
                )
        
        # Build model
        print(checkpoint_path)
        
        # Handle CUDA/CPU mapping for checkpoints
        if checkpoint_path.startswith("http"):
            # For downloaded models, let SAM handle the loading
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        else:
            # For local files, handle device mapping manually
            import torch
            
            # Create model first without checkpoint
            sam = sam_model_registry[model_type]()
            
            # Load checkpoint with proper device mapping
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            sam.load_state_dict(checkpoint)
            
        sam.to(device=self.device)
        
        return sam
    
    def generate(self, image_rgb: np.ndarray) -> list[dict]:
        """
        Generate masks automatically (compatible with SAM 2.1 interface)
        
        Returns:
            List of mask dictionaries with same format as SAM 2.1
        """
        sam_result = self.mask_generator.generate(image_rgb)
        return sam_result
    
    def predict(self, image_rgb: np.ndarray, xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks from bounding box (compatible with SAM 2.1 interface)
        
        Args:
            image_rgb: RGB image array
            xyxy: Bounding box in [x1, y1, x2, y2] format
            
        Returns:
            masks, scores, logits (same format as SAM 2.1)
        """
        self.predictor.set_image(image_rgb)
        
        # Convert xyxy to box format expected by original SAM
        if len(xyxy.shape) == 1:
            input_box = xyxy  # Already in correct format
        else:
            input_box = xyxy[0] if len(xyxy) > 0 else xyxy
        
        masks, scores, logits = self.predictor.predict(
            box=input_box,
            multimask_output=False
        )
        
        return masks, scores, logits
    
    def predict_batch(self, images_rgb: list[np.ndarray], xyxy: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """
        Batch prediction (processes one by one for original SAM)
        """
        all_masks, all_scores, all_logits = [], [], []
        
        for image, box in zip(images_rgb, xyxy):
            masks, scores, logits = self.predict(image, box)
            all_masks.append(masks)
            all_scores.append(scores)
            all_logits.append(logits)
        
        return all_masks, all_scores, all_logits


def build_original_sam(sam_type: str, ckpt_path: Optional[str] = None, device: str = "cuda") -> OriginalSAM:
    """
    Factory function to build original SAM models
    """
    return OriginalSAM(sam_type=sam_type, ckpt_path=ckpt_path, device=device)
