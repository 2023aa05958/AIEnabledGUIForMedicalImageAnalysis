from io import BytesIO
import json

import litserve as ls
import numpy as np
from fastapi import Response, UploadFile
from PIL import Image

from . import LangSAM
from .utils import draw_image

PORT = 8001


class LangSAMAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        """Initialize or load the LangSAM model."""
        # 🚀 Default SAM 2.1 model (supports dynamic switching)
        self.model = LangSAM("brain_tumour_sam2")  # Default SAM 2.1
        
        # Alternative options:
        # Option 1: Brain tumor model (if you need medical specialization)
        # self.model = LangSAM("brain_tumor_sam_vit_base")  # Your trained brain tumor model
        
        # Option 2: Explicit path to your brain tumor model (if needed)
        # self.model = LangSAM(sam_type="brain_tumour_sam2", ckpt_path="./src/libs/sam_libs/models/model_weights/brain_tumor_sam_vit_base50.pth", device=device)
        
        print("🚀 SAM 2.1 model initialized with dynamic model switching enabled.")
    
    def _boxes_overlap(self, box1, box2, threshold=0.1):
        """Check if two bounding boxes overlap significantly"""
        # box1 and box2 are in format [x1, y1, x2, y2]
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_int = max(x1_1, x1_2)
        y1_int = max(y1_1, y1_2)
        x2_int = min(x2_1, x2_2)
        y2_int = min(y2_1, y2_2)
        
        if x2_int <= x1_int or y2_int <= y1_int:
            return False  # No intersection
        
        # Calculate areas
        intersection_area = (x2_int - x1_int) * (y2_int - y1_int)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate IoU (Intersection over Union)
        union_area = box1_area + box2_area - intersection_area
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou > threshold

    def decode_request(self, request) -> dict:
        """Decode the incoming request to extract parameters and image bytes.

        Assumes the request is sent as multipart/form-data with fields:
        - sam_type: str
        - box_threshold: float
        - text_threshold: float
        - text_prompt: str
        - bounding_box: str (optional, format: "x1,y1,x2,y2")
        - image: UploadFile
        """
        # Extract form data
        sam_type = request.get("sam_type")
        box_threshold = float(request.get("box_threshold", 0.3))
        text_threshold = float(request.get("text_threshold", 0.25))
        text_prompt = request.get("text_prompt", "")
        
        # Extract bounding box coordinates (optional)
        bounding_box_str = request.get("bounding_box", "")
        bounding_box = None
        if bounding_box_str and bounding_box_str.strip():
            try:
                # Parse format: "x1,y1,x2,y2"
                coords = [float(x.strip()) for x in bounding_box_str.split(',')]
                if len(coords) == 4:
                    bounding_box = coords  # [x1, y1, x2, y2]
                    print(f"🎯 Using bounding box: {bounding_box}")
                else:
                    print(f"⚠️  Invalid bounding box format: {bounding_box_str}")
            except Exception as e:
                print(f"⚠️  Error parsing bounding box: {e}")

        # Extract image file
        image_file: UploadFile = request.get("image")
        if image_file is None:
            raise ValueError("No image file provided in the request.")

        image_bytes = image_file.file.read()

        return {
            "sam_type": sam_type,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "image_bytes": image_bytes,
            "text_prompt": text_prompt,
            "bounding_box": bounding_box,
        }

    def predict(self, inputs: dict) -> dict:
        """Perform prediction using the LangSAM model.
        
        Architecture:
        1. User Input: Image + Text prompt
        2. GDINO: Text → Object detection → Bounding boxes
        3. SAM: Image + GDINO's bounding boxes → Segmentation masks
        
        User-provided bounding box serves as guidance/filtering for GDINO results.
        """
        print("Starting prediction with parameters:")
        print(
            f"sam_type: {inputs['sam_type']}, \
                box_threshold: {inputs['box_threshold']}, \
                text_threshold: {inputs['text_threshold']}, \
                text_prompt: {inputs['text_prompt']}"
        )
        print(f"🔧 Current model type: {self.model.sam_type}")
        print(f"🔧 Requested model type: {inputs['sam_type']}")
        
        # Check if user provided bounding box guidance
        if inputs.get('bounding_box'):
            print(f"🎯 User guidance bounding box: {inputs['bounding_box']}")
            print("🔍 Will filter GDINO detections within this region")
        else:
            print("🌐 Full image analysis (no user bounding box guidance)")

        # Handle dynamic model type switching
        if inputs["sam_type"] != self.model.sam_type:
            print(f"🔄 Switching SAM model from {self.model.sam_type} to {inputs['sam_type']}")
            try:
                # Handle different model switching methods based on model type
                if inputs["sam_type"] == "brain_tumor_sam_vit_base":
                    # Switch TO brain tumor model
                    print("🧠 Switching to specialized brain tumor model...")
                    self.model = LangSAM("brain_tumor_sam_vit_base")
                    print("✅ Brain tumor model loaded successfully")
                
                elif self.model.sam_type == "brain_tumor_sam_vit_base":
                    # Switch FROM brain tumor model to standard SAM
                    print("🚀 Switching from brain tumor model to standard SAM...")
                    self.model = LangSAM(inputs["sam_type"])
                    print(f"✅ Standard SAM model {inputs['sam_type']} loaded successfully")
                
                else:
                    # Standard SAM model switching
                    print("🔧 Switching between standard SAM models...")
                    if hasattr(self.model.sam, 'build_model'):
                        self.model.sam.build_model(inputs["sam_type"])
                        print(f"✅ Model switched to {inputs['sam_type']} successfully")
                    else:
                        # Fallback: reinitialize the entire model
                        self.model = LangSAM(inputs["sam_type"])
                        print(f"✅ Model reinitialized to {inputs['sam_type']} successfully")
                        
            except Exception as e:
                print(f"⚠️  Failed to switch model: {e}")
                print(f"🔧 Continuing with current model: {self.model.sam_type}")
                print(f"💡 If switching to brain tumor model fails, ensure the model weights are available")

        try:
            image_pil = Image.open(BytesIO(inputs["image_bytes"])).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid image data: {e}")

        # **CORE ARCHITECTURE: GDINO → SAM Pipeline**
        print("🔄 Step 1: GDINO detecting objects from text prompt...")
        
        # Use GDINO to detect objects based on text prompt
        results = self.model.predict(
            images_pil=[image_pil],
            texts_prompt=[inputs["text_prompt"]],
            box_threshold=inputs["box_threshold"],
            text_threshold=inputs["text_threshold"],
        )
        results = results[0]
        
        print(f"📊 GDINO found {len(results.get('boxes', []))} initial detections")
        
        # **USER GUIDANCE: Filter GDINO results with user bounding box (if provided)**
        if inputs.get('bounding_box') and len(results.get("boxes", [])) > 0:
            user_bbox = inputs['bounding_box']
            print(f"🎯 Step 2: Filtering GDINO detections within user guidance box: {user_bbox}")
            print(f"📏 User guidance box area: {(user_bbox[2]-user_bbox[0]) * (user_bbox[3]-user_bbox[1]):.1f} pixels²")
            
            filtered_boxes = []
            filtered_scores = []
            filtered_masks = []
            filtered_labels = []
            
            print(f"🔍 Checking {len(results['boxes'])} GDINO detections:")
            for i, detected_box in enumerate(results.get("boxes", [])):
                detection_area = (detected_box[2] - detected_box[0]) * (detected_box[3] - detected_box[1])
                print(f"  Detection {i+1}: {detected_box} (area: {detection_area:.1f} pixels², score: {results['scores'][i]:.3f})")
                
                # Calculate overlap details
                overlap_result = self._boxes_overlap(detected_box, user_bbox, threshold=0.1)
                
                # Calculate IoU for debugging
                x1_int = max(detected_box[0], user_bbox[0])
                y1_int = max(detected_box[1], user_bbox[1])
                x2_int = min(detected_box[2], user_bbox[2])
                y2_int = min(detected_box[3], user_bbox[3])
                
                if x2_int > x1_int and y2_int > y1_int:
                    intersection_area = (x2_int - x1_int) * (y2_int - y1_int)
                    union_area = detection_area + (user_bbox[2]-user_bbox[0])*(user_bbox[3]-user_bbox[1]) - intersection_area
                    iou = intersection_area / union_area if union_area > 0 else 0
                    print(f"    📊 IoU with guidance box: {iou:.3f} (intersection: {intersection_area:.1f})")
                else:
                    iou = 0
                    print(f"    📊 No intersection with guidance box")
                
                if overlap_result:
                    filtered_boxes.append(detected_box)
                    filtered_scores.append(results["scores"][i])
                    filtered_masks.append(results["masks"][i])
                    filtered_labels.append(results["labels"][i])
                    print(f"    ✅ KEPT: Detection overlaps with guidance (IoU: {iou:.3f})")
                else:
                    print(f"    ❌ FILTERED: Detection outside guidance area (IoU: {iou:.3f} < threshold)")
            
            results = {
                "boxes": filtered_boxes,
                "scores": filtered_scores,
                "masks": filtered_masks,
                "labels": filtered_labels
            }
            
            print(f"🎯 After user guidance filtering: {len(filtered_boxes)} detections remaining")
            
            # Provide helpful feedback when no detections remain
            if len(filtered_boxes) == 0:
                print("💡 SUGGESTION: Try one of these approaches:")
                print("   1. Expand your bounding box to include the detected region")
                print("   2. Use 'Text Prompt' mode to see all GDINO detections first")
                print("   3. Adjust the overlap threshold in the filtering logic")
                print("   4. Check if the tumor is actually in the detected location")
        
        elif inputs.get('bounding_box'):
            print("ℹ️  User provided guidance box but GDINO found no detections to filter")
        
        print(f"✅ Step 3: Final results - {len(results.get('masks', []))} segmented regions")

        if not len(results["masks"]):
            print("No masks detected. Returning original image.")
            return {"output_image": image_pil}

        # Convert empty lists to proper numpy arrays for supervision
        if len(results["boxes"]) == 0:
            print("No detections after filtering. Returning original image.")
            return {"output_image": image_pil}
        
        # **SAVE ALL MASKS: Save all generated masks before selecting the smallest bounding box**
        if len(results["masks"]) > 0:
            print(f"💾 Saving all {len(results['masks'])} generated masks before selection...")
            
            # Create directory for saving all masks
            import os
            from datetime import datetime
            all_masks_dir = os.path.join(os.path.dirname(__file__), "..", "..", "images", "all_masks")
            os.makedirs(all_masks_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            for i, (mask, box, score, label) in enumerate(zip(results["masks"], results["boxes"], results["scores"], results["labels"])):
                # Save individual mask as image
                mask_array = np.array(mask)
                mask_img = Image.fromarray((mask_array * 255).astype(np.uint8))
                
                # Create descriptive filename with metadata
                area = (box[2] - box[0]) * (box[3] - box[1])
                mask_filename = f"mask_{timestamp}_det{i+1}_area{area:.0f}_score{score:.3f}_label{label}.png"
                mask_path = os.path.join(all_masks_dir, mask_filename)
                mask_img.save(mask_path)
                
                print(f"   💾 Saved mask {i+1}: {mask_filename}")
                print(f"       Box: {box}, Area: {area:.1f} pixels², Score: {score:.3f}, Label: {label}")
            
            print(f"✅ All masks saved to: {all_masks_dir}")
        
        # **SMALLEST BOUNDING BOX SELECTION: If multiple detections, select the one with smallest area**
        if len(results["masks"]) > 1 and len(results["boxes"]) == len(results["masks"]):
            print(f"🎯 Step 4: Selecting smallest bounding box from {len(results['boxes'])} detections...")
            
            # Calculate area for each bounding box: (x2-x1)*(y2-y1)
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in results["boxes"]]
            min_idx = areas.index(min(areas))
            
            print(f"📊 Bounding box areas:")
            for i, (box, area) in enumerate(zip(results["boxes"], areas)):
                status = "← SELECTED" if i == min_idx else ""
                print(f"   Detection {i+1}: {box} → {area:.1f} pixels² {status}")
            
            # Create directory for saving selected masks
            import os
            from datetime import datetime
            mask_dir = os.path.join(os.path.dirname(__file__), "..", "..", "images", "annotated")
            os.makedirs(mask_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Select only the detection with smallest bounding box
            selected_masks = [results["masks"][min_idx]]
            selected_boxes = [results["boxes"][min_idx]]
            selected_labels = [results["labels"][min_idx]]
            selected_scores = [results["scores"][min_idx]]
            
            # Save the selected mask for debugging
            mask_array = selected_masks[0]
            mask_img = Image.fromarray((mask_array * 255).astype(np.uint8))
            mask_filename = f"selected_smallest_mask_{timestamp}.png"
            mask_path = os.path.join(mask_dir, mask_filename)
            mask_img.save(mask_path)
            print(f"💾 Saved selected smallest mask to: {mask_path}")
            
            # Update results to contain only the selected detection
            results = {
                "masks": selected_masks,
                "boxes": selected_boxes,
                "labels": selected_labels,
                "scores": selected_scores
            }
            
            print(f"✅ Selected detection {min_idx+1} with smallest area: {areas[min_idx]:.1f} pixels²")
        else:
            print(f"ℹ️  Only {len(results['masks'])} detection(s) found - using all detections")
        
        # **ARRAY VALIDATION: Ensure all results are proper numpy arrays with correct shapes**
        # Convert masks to numpy array
        if isinstance(results["masks"], list):
            results["masks"] = np.array(results["masks"])
        
        # Convert and validate boxes
        if isinstance(results["boxes"], list):
            boxes_array = np.array(results["boxes"])
        else:
            boxes_array = results["boxes"]
        
        if boxes_array.ndim != 2 or boxes_array.shape[1] != 4:
            print(f"⚠️  Invalid bounding box shape: {boxes_array.shape}. Returning original image.")
            return {"output_image": image_pil}
        results["boxes"] = boxes_array
        
        # Convert and validate labels
        if isinstance(results["labels"], list):
            labels_array = np.array(results["labels"])
        else:
            labels_array = results["labels"]
        
        if labels_array.ndim != 1:
            labels_array = labels_array.flatten()
        results["labels"] = labels_array
        
        # Convert and validate scores
        if isinstance(results["scores"], list):
            scores_array = np.array(results["scores"])
        else:
            scores_array = results["scores"]
        
        if scores_array.ndim != 1:
            scores_array = scores_array.flatten()
        results["scores"] = scores_array
        
        print(f"✅ Array validation completed - all results converted to proper numpy arrays")

        # Draw results on the image
        image_array = np.asarray(image_pil)
        
        # Debug: Check what we're passing to draw_image
        print(f"🔍 Drawing debug info:")
        print(f"   - Number of masks: {len(results['masks']) if results.get('masks') is not None and hasattr(results['masks'], '__len__') else 0}")
        print(f"   - Number of boxes: {len(results['boxes']) if hasattr(results['boxes'], '__len__') else 'N/A'}")
        print(f"   - Number of scores: {len(results['scores']) if hasattr(results['scores'], '__len__') else 'N/A'}")
        print(f"   - Number of labels: {len(results['labels']) if results.get('labels') is not None and hasattr(results['labels'], '__len__') else 0}")
        
        if results.get("masks") is not None and hasattr(results["masks"], '__len__') and len(results["masks"]) > 0:
            mask_sample = results["masks"][0]
            print(f"   - First mask shape: {mask_sample.shape if hasattr(mask_sample, 'shape') else 'No shape'}")
            print(f"   - First mask type: {type(mask_sample)}")
            if hasattr(mask_sample, 'shape') and len(mask_sample.shape) >= 2:
                print(f"   - First mask dimensions: {mask_sample.shape[0]}x{mask_sample.shape[1]}")
                print(f"   - First mask data type: {mask_sample.dtype if hasattr(mask_sample, 'dtype') else 'No dtype'}")
                print(f"   - First mask min/max values: {mask_sample.min():.3f}/{mask_sample.max():.3f}" if hasattr(mask_sample, 'min') else 'No min/max')
        
        # Calculate IoU scores for the segmentation masks
        iou_scores = []
        total_mask_area = 0
        
        if results.get("masks") is not None and hasattr(results["masks"], '__len__') and len(results["masks"]) > 0:
            for i, mask in enumerate(results["masks"]):
                if hasattr(mask, 'sum'):
                    mask_area = float(mask.sum())
                    total_mask_area += mask_area
                    
                    # Calculate IoU with the detection box if available
                    if results.get("boxes") is not None and i < len(results["boxes"]):
                        box = results["boxes"][i]
                        box_area = (box[2] - box[0]) * (box[3] - box[1])
                        
                        # Simple IoU estimation: mask area / box area
                        iou_estimate = min(mask_area / box_area, 1.0) if box_area > 0 else 0.0
                        iou_scores.append(iou_estimate)
                    else:
                        iou_scores.append(0.0)
                else:
                    iou_scores.append(0.0)
        
        # Calculate overall IoU score
        avg_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
        
        print(f"📊 IoU Analysis:")
        print(f"   - Individual IoU scores: {[f'{score:.3f}' for score in iou_scores]}")
        print(f"   - Average IoU: {avg_iou:.3f}")
        print(f"   - Total mask area: {total_mask_area:.1f} pixels")

        output_image = draw_image(
            image_array,
            results["masks"],
            results["boxes"],
            results["scores"],
            results["labels"],
        )
        output_image = Image.fromarray(np.uint8(output_image)).convert("RGB")

        return {
            "output_image": output_image,
            "iou_score": float(avg_iou),
            "mask_area": float(total_mask_area),
            "num_detections": len(results["masks"]) if results.get("masks") is not None and hasattr(results["masks"], '__len__') else 0
        }

    def encode_response(self, output: dict) -> Response:
        """Encode the prediction result into an HTTP response.

        Returns:
            Response: Contains JSON with image data and metrics.
        """
        try:
            image = output["output_image"]
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            
            # Encode image as base64 for JSON response
            import base64
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            response_data = {
                "image": image_b64,
                "iou_score": float(output.get("iou_score", 0.0)),
                "mask_area": float(output.get("mask_area", 0.0)),
                "num_detections": int(output.get("num_detections", 0))
            }
            
            return Response(
                content=json.dumps(response_data),
                media_type="application/json"
            )
        except Exception as e:
            print(f"❌ Error encoding response: {e}")
            # Fallback to image-only response
            image = output["output_image"]
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            return Response(content=buffer.getvalue(), media_type="image/png")


lit_api = LangSAMAPI()
server = ls.LitServer(lit_api)


if __name__ == "__main__":
    print(f"Starting LitServe and Gradio server on port {PORT}...")
    server.run(port=PORT)
