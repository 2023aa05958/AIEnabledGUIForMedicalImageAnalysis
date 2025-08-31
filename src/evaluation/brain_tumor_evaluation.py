"""
Brain Tumor Dataset Evaluation using Grounding DINO
Evaluates Grounding DINO model on actual brain tumor dataset with ground truth masks.
"""

import torch
import numpy as np
from PIL import Image
import sys
import os
import glob
from pathlib import Path
import requests
import json
import time
from datetime import datetime

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.gdino_statistics import GdinoStatisticsCalculator, run_gdino_evaluation

# Import SAM models
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'libs'))
    from sam_libs.models.sam import SAM_MODELS
except ImportError:
    print("⚠️ Could not import SAM_MODELS, using default models")
    SAM_MODELS = {
        "brain_tumor_sam_vit_base": {"description": "Brain Tumor SAM ViT Base"},
        "brain_tumour_sam2": {"description": "Brain Tumor SAM 2.1"},
    }

class BrainTumorDataset:
    """
    Custom dataset class for brain tumor images and masks.
    """
    
    def __init__(self, images_dir, masks_dir):
        """
        Initialize brain tumor dataset.
        
        Args:
            images_dir: Directory containing test images
            masks_dir: Directory containing ground truth masks
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        
        # Get all jpg images
        self.image_files = sorted(glob.glob(str(self.images_dir / "*.jpg")))
        
        print(f"Found {len(self.image_files)} images in {images_dir}")
        
        # Match corresponding masks
        self.data_pairs = []
        for img_path in self.image_files:
            img_name = Path(img_path).stem
            
            # Look for corresponding mask (try different extensions)
            mask_patterns = [
                self.masks_dir / f"{img_name}.jpg",
                self.masks_dir / f"{img_name}.png",
                self.masks_dir / f"{img_name}.tif",
                self.masks_dir / f"{img_name}.tiff"
            ]
            
            mask_path = None
            for pattern in mask_patterns:
                if pattern.exists():
                    mask_path = str(pattern)
                    break
            
            if mask_path:
                self.data_pairs.append((img_path, mask_path))
            else:
                print(f"⚠️ No mask found for {img_name}")
        
        print(f"Found {len(self.data_pairs)} valid image-mask pairs")
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        """
        Get image and mask pair.
        
        Returns:
            dict: {"image": PIL.Image, "label": numpy.ndarray}
        """
        img_path, mask_path = self.data_pairs[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Load mask and convert to binary
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale
        mask_array = np.array(mask)
        
        # Convert to binary mask (assuming white pixels are tumor regions)
        binary_mask = (mask_array > 128).astype(np.uint8) * 255
        
        return {
            "image": image,
            "label": binary_mask,
            "image_path": img_path,
            "mask_path": mask_path
        }


class GdinoModelPredictor:
    """
    Wrapper for making predictions using the existing Grounding DINO server.
    """
    
    def __init__(self, server_url="http://localhost:8001", sam_type="brain_tumor_sam_vit_base"):
        """
        Initialize the model predictor.
        
        Args:
            server_url: URL of the running SAM server
            sam_type: Type of SAM model to use
        """
        self.server_url = server_url
        self.sam_type = sam_type
        print(f"🤖 Initialized predictor with model: {sam_type}")
        
    def predict_mask(self, image, text_prompt="brain tumor"):
        """
        Predict segmentation mask for an image.
        
        Args:
            image: PIL Image
            text_prompt: Text prompt for segmentation
            
        Returns:
            numpy.ndarray: Predicted binary mask
        """
        try:
            # Save image temporarily
            temp_image_path = "temp_prediction_image.jpg"
            image.save(temp_image_path, "JPEG", quality=95)
            
            # Prepare request data
            url = f"{self.server_url}/predict"
            
            with open(temp_image_path, "rb") as f:
                files = {"image": f}
                data = {
                    "sam_type": self.sam_type,
                    "text_prompt": text_prompt,
                    "box_threshold": 0.25,
                    "text_threshold": 0.25,
                    "analysis_mode": "medical_annotation"
                }
                
                response = requests.post(url, files=files, data=data, timeout=60)  # Increased timeout
            
            # Clean up temp file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            
            if response.status_code == 200:
                # Check if response is JSON (with IoU data) or image
                content_type = response.headers.get('content-type', '')
                
                if 'application/json' in content_type:
                    # Parse JSON response
                    response_data = response.json()
                    
                    # Decode base64 image
                    import base64
                    from io import BytesIO
                    
                    image_data = base64.b64decode(response_data['image'])
                    pred_image = Image.open(BytesIO(image_data))
                    
                    # Convert predicted image to binary mask
                    # This is a simplified approach - you might need to adjust based on your output format
                    pred_array = np.array(pred_image.convert("L"))
                    binary_mask = (pred_array > 128).astype(np.uint8)
                    
                    return binary_mask, response_data
                else:
                    # Image response
                    pred_image = Image.open(BytesIO(response.content))
                    pred_array = np.array(pred_image.convert("L"))
                    binary_mask = (pred_array > 128).astype(np.uint8)
                    
                    return binary_mask, {}
            else:
                print(f"❌ Prediction failed: {response.status_code} - {response.text}")
                return None, {}
                
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return None, {}


class BrainTumorGdinoEvaluator:
    """
    Custom evaluator for brain tumor dataset using Grounding DINO.
    """
    
    def __init__(self, images_dir, masks_dir, server_url="http://localhost:8001", sam_type="brain_tumor_sam_vit_base"):
        """
        Initialize the evaluator.
        
        Args:
            images_dir: Directory with test images
            masks_dir: Directory with ground truth masks
            server_url: URL of the running server
            sam_type: Type of SAM model to use
        """
        self.dataset = BrainTumorDataset(images_dir, masks_dir)
        self.predictor = GdinoModelPredictor(server_url, sam_type)
        self.results = []
        
    def evaluate_dataset(self, text_prompt="brain tumor", output_dir="brain_tumor_evaluation", start_idx=0, max_images=None):
        """
        Evaluate the brain tumor dataset.
        
        Args:
            text_prompt: Text prompt for segmentation
            output_dir: Output directory for results
            start_idx: Starting index (for resuming evaluation)
            max_images: Maximum number of images to process (None for all)
            
        Returns:
            List of evaluation results
        """
        total_dataset_size = len(self.dataset)
        
        # Determine actual number of images to process
        if max_images is None:
            end_idx = total_dataset_size
        else:
            end_idx = min(start_idx + max_images, total_dataset_size)
        
        actual_count = end_idx - start_idx
        
        print(f"🧠 Starting brain tumor dataset evaluation...")
        print(f"📊 Total dataset size: {total_dataset_size} images")
        print(f"🎯 Processing: {actual_count} images (from {start_idx + 1} to {end_idx})")
        
        if start_idx > 0:
            print(f"🔄 Resuming from image {start_idx + 1}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load existing results if resuming
        results_file = os.path.join(output_dir, "partial_results.json")
        if start_idx > 0 and os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    self.results = json.load(f)
                print(f"📂 Loaded {len(self.results)} existing results")
            except:
                self.results = []
        else:
            self.results = []
        
        for idx in range(start_idx, end_idx):
            print(f"🔄 Evaluating image {idx + 1}/{end_idx} (Progress: {idx - start_idx + 1}/{actual_count})")
            
            try:
                # Get data
                data = self.dataset[idx]
                image = data["image"]
                ground_truth = data["label"]
                
                # Get prediction
                pred_mask, extra_data = self.predictor.predict_mask(image, text_prompt)
                
                if pred_mask is None:
                    print(f"❌ Prediction failed for image {idx}")
                    self.results.append({
                        "image_id": idx,
                        "error": "Prediction failed",
                        "image_path": data.get("image_path", ""),
                        "mask_path": data.get("mask_path", "")
                    })
                    continue
                
                # Calculate metrics manually since we're using server predictions
                result = self.calculate_metrics(
                    pred_mask=pred_mask,
                    ground_truth=ground_truth,
                    image_id=idx,
                    image_path=data["image_path"],
                    mask_path=data["mask_path"],
                    extra_data=extra_data
                )
                
                self.results.append(result)
                
                # Save individual prediction for inspection
                pred_image = Image.fromarray(pred_mask * 255)
                pred_save_path = os.path.join(output_dir, f"prediction_{idx:04d}.png")
                pred_image.save(pred_save_path)
                
                # Add delay between requests to prevent server overload
                time.sleep(0.5)  # 500ms delay
                
                # Save progress every 10 images
                if (idx + 1) % 10 == 0:
                    with open(results_file, 'w') as f:
                        json.dump(self.results, f, indent=2)
                    print(f"💾 Progress saved at image {idx + 1}")
                
            except KeyboardInterrupt:
                print(f"\n⚠️ Evaluation interrupted at image {idx}")
                print(f"💾 Saving progress... Resume with start_idx={idx}")
                with open(results_file, 'w') as f:
                    json.dump(self.results, f, indent=2)
                return self.results
                
            except Exception as e:
                print(f"❌ Error evaluating image {idx}: {e}")
                self.results.append({
                    "image_id": idx,
                    "error": str(e),
                    "image_path": data.get("image_path", ""),
                    "mask_path": data.get("mask_path", "")
                })
        
        # Save final results
        self.save_results(output_dir)
        self.generate_report(output_dir)
        
        print(f"✅ Evaluation completed!")
        print(f"📊 Processed: {actual_count} images")
        print(f"💾 Results saved to {output_dir}")
        return self.results
    
    def calculate_metrics(self, pred_mask, ground_truth, image_id, image_path, mask_path, extra_data):
        """
        Calculate all evaluation metrics for a single image.
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        try:
            # Ensure masks are binary and flattened
            pred_flat = pred_mask.flatten()
            gt_flat = (ground_truth.flatten() > 0).astype(np.uint8)
            
            # Basic metrics
            accuracy = accuracy_score(gt_flat, pred_flat)
            precision = precision_score(gt_flat, pred_flat, average='binary', zero_division=0)
            recall = recall_score(gt_flat, pred_flat, average='binary', zero_division=0)
            dice = f1_score(gt_flat, pred_flat, average='binary', zero_division=0)
            
            # IoU calculation
            intersection = np.logical_and(pred_mask, ground_truth > 0).sum()
            union = np.logical_or(pred_mask, ground_truth > 0).sum()
            iou = intersection / union if union > 0 else 0.0
            
            # Confusion matrix components
            tn = np.sum((gt_flat == 0) & (pred_flat == 0))
            fp = np.sum((gt_flat == 0) & (pred_flat == 1))
            fn = np.sum((gt_flat == 1) & (pred_flat == 0))
            tp = np.sum((gt_flat == 1) & (pred_flat == 1))
            
            # Additional metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # Area analysis
            gt_area = np.sum(gt_flat)
            pred_area = np.sum(pred_flat)
            area_diff = abs(pred_area - gt_area) / gt_area if gt_area > 0 else 0.0
            
            result = {
                "image_id": image_id,
                "image_path": os.path.basename(image_path),
                "mask_path": os.path.basename(mask_path),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "dice_coefficient": float(dice),
                "iou_score": float(iou),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "ground_truth_area": int(gt_area),
                "predicted_area": int(pred_area),
                "area_difference_ratio": float(area_diff),
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
            }
            
            # Add extra data from server if available
            if extra_data:
                result.update({
                    "server_iou_score": extra_data.get("iou_score", 0.0),
                    "server_mask_area": extra_data.get("mask_area", 0.0),
                    "server_num_detections": extra_data.get("num_detections", 0)
                })
            
            return result
            
        except Exception as e:
            return {
                "image_id": image_id,
                "error": str(e),
                "image_path": os.path.basename(image_path),
                "mask_path": os.path.basename(mask_path)
            }
    
    def save_results(self, output_dir):
        """Save results to CSV and JSON files with timestamps."""
        import csv
        import pandas as pd
        
        # Generate timestamp for file names
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Filter valid results
        valid_results = [r for r in self.results if "error" not in r]
        
        if not valid_results:
            print("⚠️ No valid results to save")
            return
        
        # Save to CSV with timestamp
        csv_filename = f"brain_tumor_evaluation_results_{timestamp}.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                if valid_results:
                    writer = csv.DictWriter(f, fieldnames=valid_results[0].keys())
                    writer.writeheader()
                    writer.writerows(valid_results)
            
            print(f"✅ Saved detailed results to {csv_path}")
        except PermissionError:
            # Try alternative filename if still blocked
            csv_filename_alt = f"brain_tumor_results_{timestamp}_{os.getpid()}.csv"
            csv_path_alt = os.path.join(output_dir, csv_filename_alt)
            with open(csv_path_alt, 'w', newline='', encoding='utf-8') as f:
                if valid_results:
                    writer = csv.DictWriter(f, fieldnames=valid_results[0].keys())
                    writer.writeheader()
                    writer.writerows(valid_results)
            print(f"✅ Saved detailed results to {csv_path_alt} (alternative filename)")
        
        # Calculate and save summary statistics
        df = pd.DataFrame(valid_results)
        metrics = ['accuracy', 'precision', 'recall', 'dice_coefficient', 'iou_score', 
                  'sensitivity', 'specificity', 'area_difference_ratio']
        
        summary = {
            "total_images": len(self.results),
            "valid_evaluations": len(valid_results),
            "failed_evaluations": len(self.results) - len(valid_results)
        }
        
        for metric in metrics:
            if metric in df.columns:
                summary[f"{metric}_mean"] = float(df[metric].mean())
                summary[f"{metric}_std"] = float(df[metric].std())
                summary[f"{metric}_min"] = float(df[metric].min())
                summary[f"{metric}_max"] = float(df[metric].max())
                summary[f"{metric}_median"] = float(df[metric].median())
        
        # Save summary with timestamp
        summary_filename = f"evaluation_summary_{timestamp}.json"
        summary_path = os.path.join(output_dir, summary_filename)
        
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            print(f"✅ Saved summary statistics to {summary_path}")
        except PermissionError:
            # Try alternative filename if still blocked
            summary_filename_alt = f"summary_{timestamp}_{os.getpid()}.json"
            summary_path_alt = os.path.join(output_dir, summary_filename_alt)
            with open(summary_path_alt, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            print(f"✅ Saved summary statistics to {summary_path_alt} (alternative filename)")
    
    def generate_report(self, output_dir):
        """Generate a comprehensive evaluation report with timestamp."""
        import time
        start_time = time.time()
        
        valid_results = [r for r in self.results if "error" not in r]
        
        if not valid_results:
            print("⚠️ No valid results for report generation")
            return
        
        import pandas as pd
        df = pd.DataFrame(valid_results)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"evaluation_report_{timestamp}.txt"
        report_path = os.path.join(output_dir, report_filename)
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("BRAIN TUMOR SEGMENTATION EVALUATION REPORT\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Dataset Overview:\n")
                f.write(f"   • Total images processed: {len(self.results)}\n")
                f.write(f"   • Successful evaluations: {len(valid_results)}\n")
                f.write(f"   • Failed evaluations: {len(self.results) - len(valid_results)}\n\n")
                
                f.write(f"Performance Metrics (Mean ± Std):\n")
                metrics = ['accuracy', 'precision', 'recall', 'dice_coefficient', 'iou_score']
                for metric in metrics:
                    if metric in df.columns:
                        mean_val = df[metric].mean()
                        std_val = df[metric].std()
                        f.write(f"   • {metric.replace('_', ' ').title()}: {mean_val:.3f} ± {std_val:.3f}\n")
                
                f.write(f"\nBest Performance:\n")
                for metric in metrics:
                    if metric in df.columns:
                        max_val = df[metric].max()
                        max_idx = df[metric].idxmax()
                        max_image = df.loc[max_idx, 'image_path']
                        f.write(f"   • Best {metric.replace('_', ' ').title()}: {max_val:.3f} ({max_image})\n")
                
                f.write(f"\nWorst Performance:\n")
                for metric in metrics:
                    if metric in df.columns:
                        min_val = df[metric].min()
                        min_idx = df[metric].idxmin()
                        min_image = df.loc[min_idx, 'image_path']
                        f.write(f"   • Worst {metric.replace('_', ' ').title()}: {min_val:.3f} ({min_image})\n")
                
                f.write(f"\nProcessing Time: {time.time() - start_time:.2f} seconds\n")
                
                if len(self.results) - len(valid_results) > 0:
                    f.write(f"\nFailed Images:\n")
                    for result in self.results:
                        if result.get('status') == 'error':
                            f.write(f"   • {result['image_path']}: {result.get('error', 'Unknown error')}\n")
                
                f.write("\n" + "="*80 + "\n")
                
        except PermissionError:
            # Generate unique timestamped filename with PID
            filename_parts = os.path.splitext(report_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"{filename_parts[0]}_{timestamp}_{os.getpid()}{filename_parts[1]}"
            print(f"Permission denied. Saving as: {report_path}")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("BRAIN TUMOR SEGMENTATION EVALUATION REPORT\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Dataset Overview:\n")
                f.write(f"   • Total images processed: {len(self.results)}\n")
                f.write(f"   • Successful evaluations: {len(valid_results)}\n")
                f.write(f"   • Failed evaluations: {len(self.results) - len(valid_results)}\n\n")
                
                f.write(f"Performance Metrics (Mean ± Std):\n")
                metrics = ['accuracy', 'precision', 'recall', 'dice_coefficient', 'iou_score']
                for metric in metrics:
                    if metric in df.columns:
                        mean_val = df[metric].mean()
                        std_val = df[metric].std()
                        f.write(f"   • {metric.replace('_', ' ').title()}: {mean_val:.3f} ± {std_val:.3f}\n")
                
                f.write(f"\nBest Performance:\n")
                for metric in metrics:
                    if metric in df.columns:
                        max_val = df[metric].max()
                        max_idx = df[metric].idxmax()
                        max_image = df.loc[max_idx, 'image_path']
                        f.write(f"   • Best {metric.replace('_', ' ').title()}: {max_val:.3f} ({max_image})\n")
                
                f.write(f"\nWorst Performance:\n")
                for metric in metrics:
                    if metric in df.columns:
                        min_val = df[metric].min()
                        min_idx = df[metric].idxmin()
                        min_image = df.loc[min_idx, 'image_path']
                        f.write(f"   • Worst {metric.replace('_', ' ').title()}: {min_val:.3f} ({min_image})\n")
                
                f.write(f"\nProcessing Time: {time.time() - start_time:.2f} seconds\n")
                
                if len(self.results) - len(valid_results) > 0:
                    f.write(f"\nFailed Images:\n")
                    for result in self.results:
                        if result.get('status') == 'error':
                            f.write(f"   • {result['image_path']}: {result.get('error', 'Unknown error')}\n")
                
                f.write("\n" + "="*80 + "\n")
        
        print(f"✅ Saved evaluation report to {report_path}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("🧠 BRAIN TUMOR EVALUATION SUMMARY")
        print("="*60)
        print(f"📊 Images: {len(valid_results)}/{len(self.results)} successful")
        
        metrics = ['dice_coefficient', 'iou_score', 'accuracy']
        for metric in metrics:
            if metric in df.columns:
                mean_val = df[metric].mean()
                print(f"🎯 Mean {metric.replace('_', ' ').title()}: {mean_val:.3f}")
        print("="*60)


def get_user_model_selection():
    """
    Let user select which model to use for evaluation.
    
    Returns:
        str: Selected model name
    """
    available_models = {
        "1": ("brain_tumor_sam_vit_base", "Brain Tumor SAM ViT Base (Custom trained)"),
        "2": ("brain_tumour_sam2", "Brain Tumor SAM 2.1 (Hiera Small)"),
        "3": ("sam2_base", "SAM 2 Base (General purpose)"),
        "4": ("sam2_large", "SAM 2 Large (General purpose)")
    }
    
    print("\n🤖 Available Models:")
    for key, (model_name, description) in available_models.items():
        print(f"   {key}. {description}")
    
    while True:
        try:
            choice = input("\nSelect model (1-4): ").strip()
            
            if choice in available_models:
                model_name, description = available_models[choice]
                print(f"✅ Selected: {description}")
                return model_name
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4")
        except KeyboardInterrupt:
            print("\n👋 Model selection cancelled")
            return "brain_tumor_sam_vit_base"  # Default model


def get_user_image_selection(total_images):
    """
    Let user select how many images to process.
    
    Args:
        total_images: Total number of available images
        
    Returns:
        int: Number of images to process
    """
    print(f"\n📊 Total available images: {total_images}")
    
    # Ask user how many images to process
    print("\n🔢 How many images would you like to evaluate?")
    print(f"   1. All images ({total_images})")
    print(f"   2. First 10 images")
    print(f"   3. First 25 images") 
    print(f"   4. First 50 images")
    print(f"   5. Custom number")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                return total_images
            elif choice == "2":
                return min(10, total_images)
            elif choice == "3":
                return min(25, total_images)
            elif choice == "4":
                return min(50, total_images)
            elif choice == "5":
                while True:
                    try:
                        custom_num = int(input(f"Enter number of images (1-{total_images}): "))
                        if 1 <= custom_num <= total_images:
                            return custom_num
                        else:
                            print(f"❌ Please enter a number between 1 and {total_images}")
                    except ValueError:
                        print("❌ Please enter a valid number")
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, 4, or 5")
        except KeyboardInterrupt:
            print("\n👋 Image selection cancelled")
            return 10  # Default to 10 images


def main():
    """
    Main function to run brain tumor evaluation.
    """
    print("="*60)
    print("🧠 BRAIN TUMOR DATASET EVALUATION")
    print("="*60)
    
    # Dataset paths
    images_dir = r"C:\DDrive\Personal\M.tech\Dissertation\BRain\Brain Tumour Br35H\images\TEST"
    masks_dir = r"C:\DDrive\Personal\M.tech\Dissertation\BRain\Brain Tumour Br35H\masks\Test"
    
    # Check if paths exist
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    if not os.path.exists(masks_dir):
        print(f"❌ Masks directory not found: {masks_dir}")
        return
    
    print(f"📁 Images: {images_dir}")
    print(f"📁 Masks: {masks_dir}")
    
    # Get user model selection
    selected_model = get_user_model_selection()
    
    # Initialize evaluator with selected model
    evaluator = BrainTumorGdinoEvaluator(
        images_dir=images_dir,
        masks_dir=masks_dir,
        server_url="http://localhost:8001",
        sam_type=selected_model
    )
    
    # Get user image selection
    total_images = len(evaluator.dataset)
    num_images = get_user_image_selection(total_images)
    
    print(f"\n✅ Selected: {num_images} images")
    
    # Ask for confirmation
    estimated_time = num_images * 2  # Estimate 2 seconds per image
    print(f"⏱️ Estimated time: ~{estimated_time // 60}min {estimated_time % 60}sec")
    
    confirm = input("🚀 Start evaluation? (y/n): ").lower().strip()
    if confirm not in ['y', 'yes']:
        print("👋 Evaluation cancelled")
        return
    
    # Run evaluation
    print("\n🚀 Starting evaluation...")
    print("⚠️ Make sure your Grounding DINO server is running on http://localhost:8001")
    
    results = evaluator.evaluate_dataset(
        text_prompt="brain tumor",
        output_dir="brain_tumor_evaluation_results",
        max_images=num_images
    )
    
    # Show final summary
    valid_results = [r for r in results if "error" not in r]
    failed_results = len(results) - len(valid_results)
    
    print(f"\n🎉 EVALUATION COMPLETE!")
    print(f"📊 Summary:")
    print(f"   • Requested: {num_images} images")
    print(f"   • Successful: {len(valid_results)} images")
    print(f"   • Failed: {failed_results} images")
    print(f"   • Success rate: {len(valid_results)/len(results)*100:.1f}%" if results else "   • Success rate: 0%")
    print(f"� Check 'brain_tumor_evaluation_results' folder for detailed results")


if __name__ == "__main__":
    main()
