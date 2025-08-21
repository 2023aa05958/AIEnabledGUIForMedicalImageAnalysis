import numpy as np
from PIL import Image

from .models.gdino import GDINO
from .models.utils import DEVICE

# Import SAM 2.1 support (optional)
try:
    from .models.sam import SAM
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("⚠️  SAM 2.1 support not available")

# Import original SAM support
try:
    from .models.sam_original import OriginalSAM, SAM_ORIGINAL_MODELS
    ORIGINAL_SAM_AVAILABLE = True
except ImportError:
    ORIGINAL_SAM_AVAILABLE = False
    print("⚠️  Original SAM support not available")

# Import brain tumor SAM support
try:
    from .models.brain_tumor_sam import load_brain_tumor_sam
    BRAIN_TUMOR_SAM_AVAILABLE = True
except ImportError:
    BRAIN_TUMOR_SAM_AVAILABLE = False
    print("⚠️  Brain tumor SAM support not available")


class LangSAM:
    def __init__(self, sam_type="sam2.1_hiera_small", ckpt_path: str | None = None, device=DEVICE):
        self.sam_type = sam_type

        # Check if this is the brain tumor model
        if sam_type == "brain_tumor_sam_vit_base" and BRAIN_TUMOR_SAM_AVAILABLE:
            print(f"🧠 Loading Brain Tumor SAM (Custom Architecture): {sam_type}")
            if ckpt_path is None:
                # Use path relative to project root (when running from src/apps/)
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                ckpt_path = os.path.join(current_dir, "models", "model_weights", "brain_tumor_sam_vit_base50.pth")
            self.sam = load_brain_tumor_sam(ckpt_path, device=device)
            self.is_brain_tumor_sam = True
            self.is_original_sam = False
        # Check if this is an original SAM model
        elif sam_type in SAM_ORIGINAL_MODELS and ORIGINAL_SAM_AVAILABLE:
            print(f"🔧 Loading Original SAM model: {sam_type}")
            self.sam = OriginalSAM(sam_type=sam_type, ckpt_path=ckpt_path, device=device)
            self.is_original_sam = True
            self.is_brain_tumor_sam = False
        # Check if SAM 2.1 is available
        elif SAM2_AVAILABLE:
            # Use SAM 2.1
            print(f"🔧 Loading SAM 2.1 model: {sam_type}")
            self.sam = SAM()
            self.sam.build_model(sam_type, ckpt_path, device=device)
            self.is_original_sam = False
            self.is_brain_tumor_sam = False
        else:
            raise ImportError(f"No compatible SAM implementation available for model type: {sam_type}")
            
        self.gdino = GDINO()
        self.gdino.build_model(device=device)

    def predict(
        self,
        images_pil: list[Image.Image],
        texts_prompt: list[str],
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ):
        """Predicts masks for given images and text prompts using GDINO and SAM models.

        Parameters:
            images_pil (list[Image.Image]): List of input images.
            texts_prompt (list[str]): List of text prompts corresponding to the images.
            box_threshold (float): Threshold for box predictions.
            text_threshold (float): Threshold for text predictions.

        Returns:
            list[dict]: List of results containing masks and other outputs for each image.
            Output format:
            [{
                "boxes": np.ndarray,
                "scores": np.ndarray,
                "masks": np.ndarray,
                "mask_scores": np.ndarray,
            }, ...]
        """

        gdino_results = self.gdino.predict(images_pil, texts_prompt, box_threshold, text_threshold)
        all_results = []
        sam_images = []
        sam_boxes = []
        sam_indices = []
        for idx, result in enumerate(gdino_results):
            result = {k: (v.cpu().numpy() if hasattr(v, "numpy") else v) for k, v in result.items()}
            processed_result = {
                **result,
                "masks": [],
                "mask_scores": [],
            }

            if result["labels"]:
                sam_images.append(np.asarray(images_pil[idx]))
                sam_boxes.append(processed_result["boxes"])
                sam_indices.append(idx)

            all_results.append(processed_result)
        if sam_images:
            print(f"Predicting {len(sam_boxes)} masks")
            
            if self.is_brain_tumor_sam:
                # Handle brain tumor SAM model (individual predictions)
                masks = []
                mask_scores = []
                for img, boxes in zip(sam_images, sam_boxes):
                    img_masks = []
                    img_scores = []
                    for box in boxes:
                        # Convert box to correct format for brain tumor SAM
                        box_coords = np.array([[box[0], box[1]], [box[2], box[3]]])
                        result_masks, result_scores, _ = self.sam.predict_with_points(
                            img, box=box
                        )
                        # Use the best mask
                        best_idx = np.argmax(result_scores)
                        img_masks.append(result_masks[best_idx])
                        img_scores.append(result_scores[best_idx])
                    masks.append(np.array(img_masks))
                    mask_scores.append(np.array(img_scores))
            else:
                # Handle SAM 2.1 and original SAM models
                if hasattr(self.sam, 'predict_batch'):
                    masks, mask_scores, _ = self.sam.predict_batch(sam_images, xyxy=sam_boxes)
                else:
                    raise NotImplementedError(f"Batch prediction not supported for model type: {self.sam_type}")
            
            for idx, mask, score in zip(sam_indices, masks, mask_scores):
                all_results[idx].update(
                    {
                        "masks": mask,
                        "mask_scores": score,
                    }
                )
            print(f"Predicted {len(all_results)} masks")
        return all_results


if __name__ == "__main__":
    model = LangSAM()
    out = model.predict(
        [Image.open("./assets/food.jpg"), Image.open("./assets/car.jpeg")],
        ["food", "car"],
    )
    print(out)
