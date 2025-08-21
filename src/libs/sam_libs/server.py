from io import BytesIO

import litserve as ls
import numpy as np
from fastapi import Response, UploadFile
from PIL import Image

from . import LangSAM
from .utils import draw_image

PORT = 8000


class LangSAMAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        """Initialize or load the LangSAM model."""
        # 🧠 Use your custom trained brain tumor model
        self.model = LangSAM("brain_tumor_sam_vit_base")  # Your trained brain tumor model
        
        # Alternative options:
        # Option 1: Default SAM 2.1
        # self.model = LangSAM("sam2.1_hiera_small")  # Default SAM 2.1
        
        # Option 2: Explicit path to your brain tumor model (if needed)
        # self.model = LangSAM(sam_type="sam2.1_hiera_small", ckpt_path="./src/libs/sam_libs/models/model_weights/brain_tumor_sam_vit_base50.pth", device=device)
        
        print("🧠 Brain tumor SAM model initialized for medical image annotation.")
    
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
        
        # Check if user provided bounding box guidance
        if inputs.get('bounding_box'):
            print(f"🎯 User guidance bounding box: {inputs['bounding_box']}")
            print("🔍 Will filter GDINO detections within this region")
        else:
            print("🌐 Full image analysis (no user bounding box guidance)")

        # Handle model type switching (only for compatible models)
        if inputs["sam_type"] != self.model.sam_type:
            if inputs["sam_type"] == "brain_tumor_sam_vit_base":
                print("⚠️  Cannot switch to brain tumor model dynamically. Using current model.")
            elif self.model.sam_type == "brain_tumor_sam_vit_base":
                print("⚠️  Cannot switch from brain tumor model dynamically. Using brain tumor model.")
            else:
                print(f"Updating SAM model type to {inputs['sam_type']}")
                self.model.sam.build_model(inputs["sam_type"])

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
        
        # Convert boxes to numpy array if it's a list
        if isinstance(results["boxes"], list):
            boxes_array = np.array(results["boxes"])
        else:
            boxes_array = results["boxes"]
            
        # Convert scores to numpy array if it's a list
        if isinstance(results["scores"], list):
            scores_array = np.array(results["scores"])
        else:
            scores_array = results["scores"]

        # Draw results on the image
        image_array = np.asarray(image_pil)
        output_image = draw_image(
            image_array,
            results["masks"],
            boxes_array,
            scores_array,
            results["labels"],
        )
        output_image = Image.fromarray(np.uint8(output_image)).convert("RGB")

        return {"output_image": output_image}

    def encode_response(self, output: dict) -> Response:
        """Encode the prediction result into an HTTP response.

        Returns:
            Response: Contains the processed image in PNG format.
        """
        try:
            image = output["output_image"]
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)
            return Response(content=buffer.getvalue(), media_type="image/png")
        except StopIteration:
            raise ValueError("No output generated by the prediction.")


lit_api = LangSAMAPI()
server = ls.LitServer(lit_api)


if __name__ == "__main__":
    print(f"Starting LitServe and Gradio server on port {PORT}...")
    server.run(port=PORT)
