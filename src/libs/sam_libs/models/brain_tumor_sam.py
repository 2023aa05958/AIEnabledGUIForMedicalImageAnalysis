"""
Custom SAM model for brain tumor segmentation
Handles the specific architecture and weight naming of the brain tumor trained model
"""

import torch
from torch import nn
import torch.nn.functional as F
from segment_anything import SamPredictor, sam_model_registry
from typing import Tuple, Optional, Dict, List
import warnings

class BrainTumorSAM:
    """Custom wrapper for brain tumor SAM model with specific weight mapping"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model_path = model_path
        self.predictor = None
        self.model = None
        
    def load_model(self):
        """Load the brain tumor model with custom weight mapping"""
        try:
            print(f"🧠 Loading brain tumor SAM model from: {self.model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            print(f"✅ Checkpoint loaded, type: {type(checkpoint)}")
            
            # Initialize base SAM model (vit_b)
            print("🔧 Initializing base SAM model...")
            sam_model = sam_model_registry["vit_b"](checkpoint=None)
            sam_model = sam_model.to(self.device)
            
            # Map the brain tumor weights to standard SAM format
            print("🔄 Mapping brain tumor weights to SAM format...")
            mapped_state_dict = self._map_weights(checkpoint, sam_model.state_dict())
            
            # Load the mapped weights
            missing_keys, unexpected_keys = sam_model.load_state_dict(mapped_state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️  Missing keys: {len(missing_keys)} (some may be expected)")
            if unexpected_keys:
                print(f"⚠️  Unexpected keys: {len(unexpected_keys)} (some may be expected)")
            
            # Create predictor
            self.predictor = SamPredictor(sam_model)
            self.model = sam_model
            
            print("✅ Brain tumor SAM model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading brain tumor model: {e}")
            return False
    
    def _map_weights(self, checkpoint: Dict, target_state_dict: Dict) -> Dict:
        """Map brain tumor model weights to standard SAM format"""
        mapped_dict = {}
        
        # Create mapping rules
        mapping_rules = {
            # Vision encoder mappings
            'vision_encoder.pos_embed': 'image_encoder.pos_embed',
            'vision_encoder.patch_embed.projection.weight': 'image_encoder.patch_embed.proj.weight',
            'vision_encoder.patch_embed.projection.bias': 'image_encoder.patch_embed.proj.bias',
            
            # Layers mapping (vision_encoder.layers -> image_encoder.blocks)
            'vision_encoder.layers.': 'image_encoder.blocks.',
            'layer_norm1': 'norm1',
            'layer_norm2': 'norm2',
            
            # Neck mapping
            'vision_encoder.neck.conv1.weight': 'image_encoder.neck.0.weight',
            'vision_encoder.neck.layer_norm1.weight': 'image_encoder.neck.1.weight',
            'vision_encoder.neck.layer_norm1.bias': 'image_encoder.neck.1.bias',
            'vision_encoder.neck.conv2.weight': 'image_encoder.neck.2.weight',
            'vision_encoder.neck.layer_norm2.weight': 'image_encoder.neck.3.weight',
            'vision_encoder.neck.layer_norm2.bias': 'image_encoder.neck.3.bias',
            
            # Prompt encoder mappings
            'prompt_encoder.shared_embedding.positional_embedding': 'prompt_encoder.pe_layer.positional_encoding_gaussian_matrix',
            'prompt_encoder.point_embed.0.weight': 'prompt_encoder.point_embeddings.0.weight',
            'prompt_encoder.point_embed.1.weight': 'prompt_encoder.point_embeddings.1.weight',
            'prompt_encoder.point_embed.2.weight': 'prompt_encoder.point_embeddings.2.weight',
            'prompt_encoder.point_embed.3.weight': 'prompt_encoder.point_embeddings.3.weight',
            'prompt_encoder.not_a_point_embed.weight': 'prompt_encoder.not_a_point_embed.weight',
            
            # Mask embedding mappings
            'prompt_encoder.mask_embed.conv1.weight': 'prompt_encoder.mask_downscaling.0.weight',
            'prompt_encoder.mask_embed.conv1.bias': 'prompt_encoder.mask_downscaling.0.bias',
            'prompt_encoder.mask_embed.layer_norm1.weight': 'prompt_encoder.mask_downscaling.1.weight',
            'prompt_encoder.mask_embed.layer_norm1.bias': 'prompt_encoder.mask_downscaling.1.bias',
            'prompt_encoder.mask_embed.conv2.weight': 'prompt_encoder.mask_downscaling.3.weight',
            'prompt_encoder.mask_embed.conv2.bias': 'prompt_encoder.mask_downscaling.3.bias',
            'prompt_encoder.mask_embed.layer_norm2.weight': 'prompt_encoder.mask_downscaling.4.weight',
            'prompt_encoder.mask_embed.layer_norm2.bias': 'prompt_encoder.mask_downscaling.4.bias',
            'prompt_encoder.mask_embed.conv3.weight': 'prompt_encoder.mask_downscaling.6.weight',
            'prompt_encoder.mask_embed.conv3.bias': 'prompt_encoder.mask_downscaling.6.bias',
            
            # Mask decoder mappings
            'mask_decoder.transformer.layer_norm_final_attn': 'mask_decoder.transformer.norm_final_attn',
            'mask_decoder.upscale_conv1': 'mask_decoder.output_upscaling.0',
            'mask_decoder.upscale_conv2': 'mask_decoder.output_upscaling.1',
            'mask_decoder.upscale_layer_norm': 'mask_decoder.output_upscaling.3',
            
            # Additional mask decoder layer norm mappings
            'mask_decoder.transformer.layers.0.layer_norm1': 'mask_decoder.transformer.layers.0.norm1',
            'mask_decoder.transformer.layers.0.layer_norm2': 'mask_decoder.transformer.layers.0.norm2',
            'mask_decoder.transformer.layers.0.layer_norm3': 'mask_decoder.transformer.layers.0.norm3',
            'mask_decoder.transformer.layers.0.layer_norm4': 'mask_decoder.transformer.layers.0.norm4',
            'mask_decoder.transformer.layers.1.layer_norm1': 'mask_decoder.transformer.layers.1.norm1',
            'mask_decoder.transformer.layers.1.layer_norm2': 'mask_decoder.transformer.layers.1.norm2',
            'mask_decoder.transformer.layers.1.layer_norm3': 'mask_decoder.transformer.layers.1.norm3',
            'mask_decoder.transformer.layers.1.layer_norm4': 'mask_decoder.transformer.layers.1.norm4',
            
            # Hypernetwork mappings
            'mask_decoder.output_hypernetworks_mlps.0.proj_in': 'mask_decoder.output_hypernetworks_mlps.0.layers.0',
            'mask_decoder.output_hypernetworks_mlps.0.proj_out': 'mask_decoder.output_hypernetworks_mlps.0.layers.2',
            'mask_decoder.output_hypernetworks_mlps.1.proj_in': 'mask_decoder.output_hypernetworks_mlps.1.layers.0',
            'mask_decoder.output_hypernetworks_mlps.1.proj_out': 'mask_decoder.output_hypernetworks_mlps.1.layers.2',
            'mask_decoder.output_hypernetworks_mlps.2.proj_in': 'mask_decoder.output_hypernetworks_mlps.2.layers.0',
            'mask_decoder.output_hypernetworks_mlps.2.proj_out': 'mask_decoder.output_hypernetworks_mlps.2.layers.2',
            'mask_decoder.output_hypernetworks_mlps.3.proj_in': 'mask_decoder.output_hypernetworks_mlps.3.layers.0',
            'mask_decoder.output_hypernetworks_mlps.3.proj_out': 'mask_decoder.output_hypernetworks_mlps.3.layers.2',
            
            # IoU prediction head mappings
            'mask_decoder.iou_prediction_head.proj_in': 'mask_decoder.iou_prediction_head.layers.0',
            'mask_decoder.iou_prediction_head.proj_out': 'mask_decoder.iou_prediction_head.layers.2',
        }
        
        print(f"🔍 Processing {len(checkpoint)} checkpoint keys...")
        mapped_count = 0
        
        for key, value in checkpoint.items():
            mapped_key = key
            
            # Apply mapping rules
            for old_pattern, new_pattern in mapping_rules.items():
                if old_pattern in mapped_key:
                    mapped_key = mapped_key.replace(old_pattern, new_pattern)
                    break
            
            # Special handling for shared_image_embedding -> skip (not needed in standard SAM)
            if 'shared_image_embedding' in key:
                continue
                
            # Only include keys that exist in target model
            if mapped_key in target_state_dict:
                # Check shape compatibility
                if value.shape == target_state_dict[mapped_key].shape:
                    mapped_dict[mapped_key] = value
                    mapped_count += 1
                else:
                    print(f"⚠️  Shape mismatch for {mapped_key}: {value.shape} vs {target_state_dict[mapped_key].shape}")
            else:
                # Try to find partial matches for debugging
                partial_matches = [t_key for t_key in target_state_dict.keys() if mapped_key.split('.')[-1] in t_key]
                if len(partial_matches) <= 3:  # Only show if few matches
                    pass  # Skip verbose output for now
        
        print(f"✅ Successfully mapped {mapped_count} weights")
        return mapped_dict
    
    def predict(self, image, text_prompt: str, box_threshold: float = 0.3, text_threshold: float = 0.25):
        """Make predictions using the brain tumor model"""
        if self.predictor is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Set image
        self.predictor.set_image(image)
        
        # For now, return empty predictions as we need to implement text prompting
        # This is a placeholder - you'd need to integrate with GroundingDINO for text prompts
        return [], [], []
    
    def predict_with_points(self, image, point_coords=None, point_labels=None, box=None):
        """Make predictions using points/boxes"""
        if self.predictor is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.predictor.set_image(image)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
        
        return masks, scores, logits

def load_brain_tumor_sam(model_path: str, device: str = "cpu") -> BrainTumorSAM:
    """Convenience function to load brain tumor SAM model"""
    brain_sam = BrainTumorSAM(model_path, device)
    if brain_sam.load_model():
        return brain_sam
    else:
        raise RuntimeError("Failed to load brain tumor SAM model")
