"""
Grounding DINO Statistics Calculator
Calculates segmentation metrics (accuracy, precision, recall, dice coefficient) for Grounding DINO model evaluation.
"""

import numpy as np
import torch
import csv
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class GdinoStatisticsCalculator:
    """
    Calculate comprehensive statistics for Grounding DINO segmentation model.
    """
    
    def __init__(self, model, processor, device, output_dir: str = "evaluation_results"):
        """
        Initialize the statistics calculator.
        
        Args:
            model: The trained Grounding DINO model
            processor: The model processor for input handling
            device: PyTorch device (cuda/cpu)
            output_dir: Directory to save results
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.output_dir = output_dir
        self.results = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure model is in evaluation mode
        self.model.eval()
        
    def get_bounding_box(self, mask: np.ndarray) -> List[int]:
        """
        Extract bounding box coordinates from a binary mask.
        
        Args:
            mask: Binary mask array
            
        Returns:
            List of bounding box coordinates [x_min, y_min, x_max, y_max]
        """
        # Find non-zero pixels
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            # Return full image bbox if no mask found
            return [0, 0, mask.shape[1], mask.shape[0]]
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        return [cmin, rmin, cmax, rmax]
    
    def calculate_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU) score.
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            
        Returns:
            IoU score (0-1)
        """
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    def calculate_hausdorff_distance(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        Calculate Hausdorff distance between predicted and ground truth masks.
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            
        Returns:
            Hausdorff distance
        """
        try:
            from scipy.spatial.distance import directed_hausdorff
            
            # Get boundary points
            pred_points = np.argwhere(pred_mask)
            gt_points = np.argwhere(gt_mask)
            
            if len(pred_points) == 0 or len(gt_points) == 0:
                return float('inf')
            
            # Calculate directed Hausdorff distances
            d1 = directed_hausdorff(pred_points, gt_points)[0]
            d2 = directed_hausdorff(gt_points, pred_points)[0]
            
            return max(d1, d2)
        except ImportError:
            print("⚠️ scipy not available, skipping Hausdorff distance calculation")
            return 0.0
    
    def evaluate_single_image(self, 
                            test_image: Image.Image, 
                            ground_truth_mask: np.ndarray, 
                            image_id: int,
                            text_prompt: str = "tumor") -> Dict[str, Any]:
        """
        Evaluate a single image and calculate all metrics.
        
        Args:
            test_image: PIL Image to evaluate
            ground_truth_mask: Ground truth segmentation mask
            image_id: Unique identifier for the image
            text_prompt: Text prompt for Grounding DINO
            
        Returns:
            Dictionary containing all calculated metrics
        """
        try:
            # Get bounding box from ground truth
            prompt = self.get_bounding_box(ground_truth_mask)
            
            # Prepare inputs for the model
            inputs = self.processor(test_image, input_boxes=[prompt], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs, multimask_output=False)
            
            # Process model output
            pred_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
            pred_mask = (pred_prob > 0.5).cpu().numpy().squeeze().astype(np.uint8)
            
            # Ensure masks are binary and flattened
            pred_flat = pred_mask.flatten()
            gt_flat = (ground_truth_mask.flatten() > 0).astype(np.uint8)  # Convert 255 to 1
            
            # Calculate basic metrics
            accuracy = accuracy_score(gt_flat, pred_flat)
            precision = precision_score(gt_flat, pred_flat, average='binary', zero_division=0)
            recall = recall_score(gt_flat, pred_flat, average='binary', zero_division=0)
            dice = f1_score(gt_flat, pred_flat, average='binary', zero_division=0)
            
            # Calculate additional metrics
            iou = self.calculate_iou(pred_mask, ground_truth_mask > 0)
            hausdorff = self.calculate_hausdorff_distance(pred_mask, ground_truth_mask > 0)
            
            # Calculate sensitivity and specificity
            tn = np.sum((gt_flat == 0) & (pred_flat == 0))
            fp = np.sum((gt_flat == 0) & (pred_flat == 1))
            fn = np.sum((gt_flat == 1) & (pred_flat == 0))
            tp = np.sum((gt_flat == 1) & (pred_flat == 1))
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # Calculate mask areas
            gt_area = np.sum(gt_flat)
            pred_area = np.sum(pred_flat)
            area_difference = abs(pred_area - gt_area) / gt_area if gt_area > 0 else 0.0
            
            result = {
                "image_id": image_id,
                "text_prompt": text_prompt,
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "dice_coefficient": float(dice),
                "iou_score": float(iou),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "hausdorff_distance": float(hausdorff),
                "ground_truth_area": int(gt_area),
                "predicted_area": int(pred_area),
                "area_difference_ratio": float(area_difference),
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "bbox_coordinates": prompt,
                "evaluation_timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error evaluating image {image_id}: {e}")
            return {
                "image_id": image_id,
                "error": str(e),
                "evaluation_timestamp": datetime.now().isoformat()
            }
    
    def evaluate_dataset(self, 
                        test_dataset, 
                        text_prompt: str = "tumor",
                        save_individual_results: bool = True) -> List[Dict[str, Any]]:
        """
        Evaluate the entire test dataset.
        
        Args:
            test_dataset: Dataset containing test images and ground truth masks
            text_prompt: Text prompt for Grounding DINO
            save_individual_results: Whether to save results for each image
            
        Returns:
            List of evaluation results for each image
        """
        print(f"🔄 Starting evaluation of {len(test_dataset)} images...")
        
        self.results = []
        
        for idx in range(len(test_dataset)):
            print(f"📊 Evaluating image {idx + 1}/{len(test_dataset)}")
            
            try:
                test_image = test_dataset[idx]["image"]
                test_ground_truth_mask = np.array(test_dataset[idx]["label"])
                
                result = self.evaluate_single_image(
                    test_image=test_image,
                    ground_truth_mask=test_ground_truth_mask,
                    image_id=idx,
                    text_prompt=text_prompt
                )
                
                self.results.append(result)
                
                # Save individual result if requested
                if save_individual_results and "error" not in result:
                    self.save_individual_result(result, idx)
                    
            except Exception as e:
                print(f"❌ Failed to evaluate image {idx}: {e}")
                self.results.append({
                    "image_id": idx,
                    "error": str(e),
                    "evaluation_timestamp": datetime.now().isoformat()
                })
        
        print(f"✅ Evaluation completed! {len(self.results)} images processed.")
        return self.results
    
    def save_individual_result(self, result: Dict[str, Any], image_id: int):
        """Save individual result as JSON file."""
        result_file = os.path.join(self.output_dir, f"image_{image_id:04d}_metrics.json")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    def save_results_csv(self, filename: str = "gdino_segmentation_metrics.csv"):
        """
        Save results to CSV file.
        
        Args:
            filename: Output CSV filename
        """
        if not self.results:
            print("⚠️ No results to save")
            return
        
        csv_path = os.path.join(self.output_dir, filename)
        
        # Filter out error results for CSV
        valid_results = [r for r in self.results if "error" not in r]
        
        if not valid_results:
            print("⚠️ No valid results to save")
            return
        
        fieldnames = valid_results[0].keys()
        
        with open(csv_path, mode="w", newline="", encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(valid_results)
        
        print(f"✅ Saved metrics to {csv_path}")
        return csv_path
    
    def calculate_summary_statistics(self) -> Dict[str, Any]:
        """
        Calculate summary statistics across all evaluated images.
        
        Returns:
            Dictionary containing summary statistics
        """
        if not self.results:
            return {}
        
        # Filter valid results
        valid_results = [r for r in self.results if "error" not in r]
        
        if not valid_results:
            return {"error": "No valid results available"}
        
        df = pd.DataFrame(valid_results)
        
        # Numerical metrics to summarize
        metrics = ['accuracy', 'precision', 'recall', 'dice_coefficient', 'iou_score', 
                  'sensitivity', 'specificity', 'hausdorff_distance', 'area_difference_ratio']
        
        summary = {
            "total_images": len(self.results),
            "valid_evaluations": len(valid_results),
            "failed_evaluations": len(self.results) - len(valid_results),
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
        # Calculate statistics for each metric
        for metric in metrics:
            if metric in df.columns:
                summary[f"{metric}_mean"] = float(df[metric].mean())
                summary[f"{metric}_std"] = float(df[metric].std())
                summary[f"{metric}_min"] = float(df[metric].min())
                summary[f"{metric}_max"] = float(df[metric].max())
                summary[f"{metric}_median"] = float(df[metric].median())
        
        return summary
    
    def save_summary_statistics(self, filename: str = "gdino_summary_statistics.json"):
        """Save summary statistics to JSON file."""
        summary = self.calculate_summary_statistics()
        
        if not summary:
            print("⚠️ No summary statistics to save")
            return
        
        summary_path = os.path.join(self.output_dir, filename)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Saved summary statistics to {summary_path}")
        return summary_path
    
    def generate_visualization_plots(self):
        """Generate visualization plots for the evaluation results."""
        if not self.results:
            print("⚠️ No results available for visualization")
            return
        
        valid_results = [r for r in self.results if "error" not in r]
        if not valid_results:
            print("⚠️ No valid results for visualization")
            return
        
        df = pd.DataFrame(valid_results)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Grounding DINO Segmentation Evaluation Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Metrics distribution
        metrics = ['accuracy', 'precision', 'recall', 'dice_coefficient', 'iou_score']
        df[metrics].boxplot(ax=axes[0, 0])
        axes[0, 0].set_title('Metrics Distribution')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Dice coefficient histogram
        axes[0, 1].hist(df['dice_coefficient'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 1].set_title('Dice Coefficient Distribution')
        axes[0, 1].set_xlabel('Dice Score')
        axes[0, 1].set_ylabel('Frequency')
        
        # Plot 3: IoU vs Dice correlation
        axes[0, 2].scatter(df['iou_score'], df['dice_coefficient'], alpha=0.6)
        axes[0, 2].set_title('IoU vs Dice Coefficient')
        axes[0, 2].set_xlabel('IoU Score')
        axes[0, 2].set_ylabel('Dice Coefficient')
        
        # Plot 4: Precision vs Recall
        axes[1, 0].scatter(df['recall'], df['precision'], alpha=0.6, color='green')
        axes[1, 0].set_title('Precision vs Recall')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        
        # Plot 5: Area difference analysis
        axes[1, 1].hist(df['area_difference_ratio'], bins=20, alpha=0.7, color='red', edgecolor='black')
        axes[1, 1].set_title('Area Difference Ratio Distribution')
        axes[1, 1].set_xlabel('Area Difference Ratio')
        axes[1, 1].set_ylabel('Frequency')
        
        # Plot 6: Metrics correlation heatmap
        correlation_matrix = df[metrics].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 2])
        axes[1, 2].set_title('Metrics Correlation Matrix')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.output_dir, "gdino_evaluation_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved visualization plots to {plot_path}")
        return plot_path
    
    def print_summary_report(self):
        """Print a comprehensive summary report."""
        summary = self.calculate_summary_statistics()
        
        if not summary:
            print("⚠️ No summary statistics available")
            return
        
        print("\n" + "="*60)
        print("🔬 GROUNDING DINO SEGMENTATION EVALUATION REPORT")
        print("="*60)
        
        print(f"\n📊 Dataset Overview:")
        print(f"   • Total images: {summary['total_images']}")
        print(f"   • Valid evaluations: {summary['valid_evaluations']}")
        print(f"   • Failed evaluations: {summary['failed_evaluations']}")
        
        print(f"\n🎯 Performance Metrics (Mean ± Std):")
        metrics = ['accuracy', 'precision', 'recall', 'dice_coefficient', 'iou_score']
        for metric in metrics:
            if f"{metric}_mean" in summary:
                mean_val = summary[f"{metric}_mean"]
                std_val = summary[f"{metric}_std"]
                print(f"   • {metric.replace('_', ' ').title()}: {mean_val:.3f} ± {std_val:.3f}")
        
        print(f"\n📈 Best Performance:")
        for metric in metrics:
            if f"{metric}_max" in summary:
                max_val = summary[f"{metric}_max"]
                print(f"   • Max {metric.replace('_', ' ').title()}: {max_val:.3f}")
        
        print(f"\n📉 Worst Performance:")
        for metric in metrics:
            if f"{metric}_min" in summary:
                min_val = summary[f"{metric}_min"]
                print(f"   • Min {metric.replace('_', ' ').title()}: {min_val:.3f}")
        
        print("\n" + "="*60)


def run_gdino_evaluation(model, processor, device, test_dataset, 
                        text_prompt: str = "tumor", 
                        output_dir: str = "gdino_evaluation"):
    """
    Convenience function to run complete Grounding DINO evaluation.
    
    Args:
        model: Trained Grounding DINO model
        processor: Model processor
        device: PyTorch device
        test_dataset: Test dataset
        text_prompt: Text prompt for segmentation
        output_dir: Output directory for results
        
    Returns:
        GdinoStatisticsCalculator instance with results
    """
    print("🚀 Starting Grounding DINO evaluation...")
    
    # Initialize calculator
    calculator = GdinoStatisticsCalculator(model, processor, device, output_dir)
    
    # Run evaluation
    results = calculator.evaluate_dataset(test_dataset, text_prompt)
    
    # Save results
    calculator.save_results_csv()
    calculator.save_summary_statistics()
    calculator.generate_visualization_plots()
    
    # Print report
    calculator.print_summary_report()
    
    print(f"✅ Evaluation completed! Results saved to: {output_dir}")
    
    return calculator


if __name__ == "__main__":
    print("🔬 Grounding DINO Statistics Calculator")
    print("This module provides comprehensive evaluation metrics for Grounding DINO segmentation.")
    print("Import and use the GdinoStatisticsCalculator class or run_gdino_evaluation function.")
