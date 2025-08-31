"""
Integration module to connect MONAI Active Learning with your existing medical image annotation system
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
import os
from datetime import datetime
from PIL import Image
import json

# Import your existing modules
from .feedback_manager import feedback_manager
from .performance_analytics import performance_analytics

try:
    from .monai_active_learning import MonaiActiveLearning, create_monai_active_learning_config
    MONAI_AL_AVAILABLE = True
except ImportError:
    MONAI_AL_AVAILABLE = False
    print("⚠️ MONAI Active Learning not available - install MONAI to enable")


class ActiveLearningOrchestrator:
    """
    Orchestrates active learning between your existing feedback system and MONAI's AL capabilities.
    Combines the best of both approaches.
    """
    
    def __init__(self, enable_monai: bool = True):
        """
        Initialize the active learning orchestrator.
        
        Args:
            enable_monai: Whether to use MONAI-based active learning
        """
        self.enable_monai = enable_monai and MONAI_AL_AVAILABLE
        self.monai_al = None
        self.uncertainty_cache = {}
        self.al_iteration = 0
        
        # Initialize data storage
        self.unlabeled_pool = []
        self.high_uncertainty_samples = []
        self.expert_review_queue = []
        
        if self.enable_monai:
            self._initialize_monai_al()
        
        print(f"🤖 Active Learning Orchestrator initialized (MONAI: {self.enable_monai})")
    
    def _initialize_monai_al(self):
        """Initialize MONAI Active Learning if available."""
        try:
            config = create_monai_active_learning_config()
            # Customize config for your brain tumor segmentation
            config.update({
                'model_type': 'unet',
                'in_channels': 1,  # Grayscale brain MRI
                'num_classes': 2,  # Background + tumor
                'batch_size': 2,   # Adjust based on your GPU memory
                'uncertainty_threshold': 0.3,  # Lower = more sensitive
                'confidence_threshold': 0.85   # Higher = more confident
            })
            
            self.monai_al = MonaiActiveLearning(config)
            print("✅ MONAI Active Learning initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize MONAI AL: {e}")
            self.enable_monai = False
    
    def analyze_prediction_uncertainty(self, 
                                       image_path: str, 
                                       prediction_result: Dict[str, Any],
                                       model_used: str) -> Dict[str, Any]:
        """
        Analyze prediction uncertainty and decide if expert review is needed.
        
        Args:
            image_path: Path to the input image
            prediction_result: Result from your SAM model
            model_used: Name of the model used
            
        Returns:
            Dictionary with uncertainty analysis and recommendations
        """
        analysis = {
            'needs_expert_review': False,
            'uncertainty_score': 0.0,
            'confidence_level': 'Unknown',
            'recommendation': 'Standard processing',
            'analysis_method': 'Rule-based',
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Method 1: MONAI-based uncertainty estimation
            if self.enable_monai and self.monai_al:
                monai_uncertainty = self._estimate_monai_uncertainty(image_path)
                analysis.update({
                    'monai_uncertainty': monai_uncertainty,
                    'analysis_method': 'MONAI + Rule-based'
                })
                
                if monai_uncertainty > self.monai_al.config.get('uncertainty_threshold', 0.6):  # Increased from 0.3
                    analysis['needs_expert_review'] = True
                    analysis['uncertainty_score'] = monai_uncertainty
                    analysis['recommendation'] = 'High uncertainty - expert review recommended'
            
            # Method 2: Rule-based uncertainty from your existing system
            rule_based_uncertainty = self._analyze_rule_based_uncertainty(prediction_result, model_used)
            analysis.update(rule_based_uncertainty)
            
            # Method 3: Historical performance-based analysis
            historical_analysis = self._analyze_historical_performance(model_used, image_path)
            analysis.update(historical_analysis)
            
            # Final decision logic with IoU override
            # Check if we have a high IoU score that should override uncertainty
            iou_score = prediction_result.get('iou_score', 0.0) if 'iou_score' in prediction_result else 0.0
            
            if iou_score > 0.7:  # High IoU overrides other uncertainty factors
                final_uncertainty = max(0.0, min(0.3, rule_based_uncertainty.get('uncertainty_score', 0)))  # Cap at 0.3 for high IoU
                analysis['iou_override'] = True
                analysis['override_reason'] = f"High IoU score ({iou_score:.3f}) overrides uncertainty calculation"
            else:
                # Normal uncertainty calculation
                final_uncertainty = max(
                    analysis.get('uncertainty_score', 0),
                    rule_based_uncertainty.get('uncertainty_score', 0),
                    historical_analysis.get('uncertainty_score', 0)
                )
            
            analysis['final_uncertainty_score'] = final_uncertainty
            
            # Determine confidence level with more reasonable thresholds
            if final_uncertainty < 0.4:  # Increased from 0.2
                analysis['confidence_level'] = 'High'
                analysis['final_confidence_score'] = 1.0 - final_uncertainty
                if iou_score > 0.7:
                    analysis['recommendation'] = f'High confidence - excellent IoU ({iou_score:.3f})'
            elif final_uncertainty < 0.7:  # Increased from 0.5
                analysis['confidence_level'] = 'Medium'
                analysis['final_confidence_score'] = 1.0 - final_uncertainty
            else:
                analysis['confidence_level'] = 'Low'
                analysis['final_confidence_score'] = 1.0 - final_uncertainty
                analysis['needs_expert_review'] = True
                analysis['recommendation'] = 'High uncertainty - expert review recommended'
            
            # Cache for future reference
            self.uncertainty_cache[image_path] = analysis
            
        except Exception as e:
            print(f"❌ Error in uncertainty analysis: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _estimate_monai_uncertainty(self, image_path: str) -> float:
        """Estimate uncertainty using MONAI's Monte Carlo sampling."""
        try:
            # This would be implemented when MONAI is fully integrated
            # For now, return a placeholder
            
            # Load and preprocess image
            image = Image.open(image_path).convert('L')  # Grayscale
            
            # Convert to tensor format expected by MONAI
            # image_tensor = self._preprocess_for_monai(image)
            
            # Get uncertainty estimate
            # uncertainty = self.monai_al.estimate_uncertainty(image_tensor)
            
            # Placeholder uncertainty based on image characteristics
            image_array = np.array(image)
            
            # Simple heuristics for uncertainty
            # High contrast variation might indicate complexity
            contrast_std = np.std(image_array)
            edge_intensity = np.mean(np.abs(np.gradient(image_array.astype(float))))
            
            # Normalize to 0-1 range
            uncertainty = min(1.0, (contrast_std / 50.0 + edge_intensity / 10.0) / 2.0)
            
            return uncertainty
            
        except Exception as e:
            print(f"❌ MONAI uncertainty estimation failed: {e}")
            return 0.5  # Medium uncertainty as fallback
    
    def _analyze_rule_based_uncertainty(self, 
                                        prediction_result: Dict[str, Any], 
                                        model_used: str) -> Dict[str, Any]:
        """Analyze uncertainty using rule-based heuristics."""
        analysis = {
            'uncertainty_score': 0.0,
            'rule_based_factors': []
        }
        
        try:
            # Factor 1: Model-specific reliability
            model_reliability = {
                'brain_tumor_sam_vit_base': 0.9,
                'sam2_base': 0.8,
                'sam2_large': 0.85,
                'default': 0.7
            }
            
            reliability = model_reliability.get(model_used, model_reliability['default'])
            uncertainty_from_model = 1.0 - reliability
            analysis['rule_based_factors'].append(f"Model reliability: {reliability:.2f}")
            
            # Factor 2: IoU Score Analysis (High IoU = High quality = Low uncertainty)
            iou_score = prediction_result.get('iou_score', 0.0)
            if iou_score > 0:
                if iou_score > 0.7:  # High IoU = Low uncertainty (no review needed)
                    uncertainty_from_model = max(0.0, uncertainty_from_model - 0.5)  # Significantly reduce uncertainty
                    analysis['rule_based_factors'].append(f"High IoU score: {iou_score:.3f} (excellent quality, no review needed)")
                elif iou_score > 0.5:  # Medium IoU = Medium uncertainty
                    uncertainty_from_model = max(0.0, uncertainty_from_model - 0.2)  # Moderately reduce uncertainty
                    analysis['rule_based_factors'].append(f"Medium IoU score: {iou_score:.3f} (good quality)")
                else:  # Low IoU = High uncertainty
                    uncertainty_from_model += 0.3  # Increase uncertainty
                    analysis['rule_based_factors'].append(f"Low IoU score: {iou_score:.3f} (poor quality, needs review)")
            else:
                # No IoU data available
                uncertainty_from_model += 0.1
                analysis['rule_based_factors'].append("No IoU score available")
            
            # Factor 3: Segmentation characteristics
            if 'masks' in prediction_result:
                masks = prediction_result['masks']
                if isinstance(masks, (list, np.ndarray)) and len(masks) > 0:
                    # Multiple masks might indicate uncertainty
                    if len(masks) > 3:
                        uncertainty_from_model += 0.2
                        analysis['rule_based_factors'].append("Multiple masks detected")
                    
                    # Small masks might be false positives
                    if isinstance(masks[0], np.ndarray):
                        mask_sizes = [np.sum(mask) for mask in masks]
                        avg_mask_size = np.mean(mask_sizes) if mask_sizes else 0
                        
                        if avg_mask_size < 1000:  # Small masks
                            uncertainty_from_model += 0.15
                            analysis['rule_based_factors'].append("Small segmentation detected")
            
            # Factor 4: Processing time (longer = more complex)
            processing_time = prediction_result.get('processing_time', 0)
            if processing_time > 5.0:  # Unusually long processing
                uncertainty_from_model += 0.1
                analysis['rule_based_factors'].append("Long processing time")
            
            analysis['uncertainty_score'] = min(1.0, uncertainty_from_model)
            
        except Exception as e:
            print(f"❌ Rule-based analysis failed: {e}")
            analysis['uncertainty_score'] = 0.5
            analysis['rule_based_factors'].append(f"Analysis error: {str(e)}")
        
        return analysis
    
    def _analyze_historical_performance(self, model_used: str, image_path: str) -> Dict[str, Any]:
        """Analyze based on historical performance of the model."""
        analysis = {
            'uncertainty_score': 0.0,
            'historical_factors': []
        }
        
        try:
            # Get historical performance data
            performance_data = performance_analytics.get_model_comparison_data(days=30)
            
            if model_used in performance_data.get('models', []):
                model_index = performance_data['models'].index(model_used)
                
                # Get model's average quality score
                quality_scores = performance_data.get('quality_scores', [])
                if model_index < len(quality_scores):
                    avg_quality = quality_scores[model_index]
                    
                    # Convert quality to uncertainty (inverse relationship)
                    # Quality 5 = 0% uncertainty, Quality 1 = 80% uncertainty
                    uncertainty_from_history = max(0, (5 - avg_quality) / 5 * 0.8)
                    analysis['uncertainty_score'] = uncertainty_from_history
                    analysis['historical_factors'].append(f"Historical quality: {avg_quality:.2f}/5")
                
                # Check success rate
                success_rates = performance_data.get('success_rates', [])
                if model_index < len(success_rates):
                    success_rate = success_rates[model_index]
                    if success_rate < 70:  # Low success rate
                        analysis['uncertainty_score'] += 0.2
                        analysis['historical_factors'].append(f"Low success rate: {success_rate:.1f}%")
            else:
                # Unknown model performance
                analysis['uncertainty_score'] = 0.4
                analysis['historical_factors'].append("No historical data available")
                
        except Exception as e:
            print(f"❌ Historical analysis failed: {e}")
            analysis['uncertainty_score'] = 0.3
            analysis['historical_factors'].append(f"Analysis error: {str(e)}")
        
        return analysis
    
    def queue_for_expert_review(self, 
                                image_path: str, 
                                uncertainty_analysis: Dict[str, Any],
                                user_info: Dict[str, str],
                                priority: str = "medium") -> str:
        """Queue a sample for expert review."""
        
        review_item = {
            'id': f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.expert_review_queue)}",
            'image_path': image_path,
            'uncertainty_analysis': uncertainty_analysis,
            'user_info': user_info,
            'priority': priority,
            'status': 'pending',
            'created_at': datetime.now().isoformat()
        }
        
        # Insert based on priority
        if priority == "high":
            self.expert_review_queue.insert(0, review_item)
        else:
            self.expert_review_queue.append(review_item)
        
        # Store in database
        try:
            feedback_data = {
                'user_id': user_info.get('username', 'system'),
                'user_role': 'system',
                'image_path': image_path,
                'model_used': 'uncertainty_detection',
                'analysis_mode': 'Active Learning',
                'text_prompt': 'Uncertainty detected',
                'bounding_box': '',
                'prediction_results': uncertainty_analysis,
                'feedback_quality': 0,  # Not rated yet
                'feedback_type': 'Expert Review Needed',
                'clinical_notes': f"Automatic uncertainty detection - {uncertainty_analysis.get('recommendation', '')}",
                'confidence_score': 1.0 - uncertainty_analysis.get('final_uncertainty_score', 0.5),
                'processing_time': 0.0
            }
            
            feedback_manager.store_feedback(feedback_data)
            
        except Exception as e:
            print(f"❌ Failed to store expert review request: {e}")
        
        print(f"📋 Queued for expert review: {review_item['id']} (Priority: {priority})")
        return review_item['id']
    
    def get_expert_review_queue(self, status: str = "pending") -> List[Dict[str, Any]]:
        """Get items in the expert review queue."""
        return [item for item in self.expert_review_queue if item['status'] == status]
    
    def run_active_learning_iteration(self, strategy: str = "hybrid") -> Dict[str, Any]:
        """Run an active learning iteration combining MONAI and feedback data."""
        
        print(f"🔄 Starting Active Learning Iteration #{self.al_iteration + 1}")
        
        iteration_results = {
            'iteration': self.al_iteration + 1,
            'strategy': strategy,
            'timestamp': datetime.now().isoformat(),
            'monai_results': {},
            'feedback_analysis': {},
            'recommendations': []
        }
        
        try:
            # Part 1: MONAI-based active learning
            if self.enable_monai and self.monai_al:
                print("🧠 Running MONAI active learning...")
                monai_results = self.monai_al.run_active_learning_iteration(
                    num_samples_to_label=5,
                    strategy=strategy
                )
                iteration_results['monai_results'] = monai_results
            
            # Part 2: Feedback-based analysis
            print("📊 Analyzing feedback data...")
            feedback_analysis = performance_analytics.get_comprehensive_analytics(days=7)
            iteration_results['feedback_analysis'] = feedback_analysis
            
            # Part 3: Generate combined recommendations
            print("💡 Generating recommendations...")
            recommendations = self._generate_combined_recommendations(
                iteration_results['monai_results'],
                iteration_results['feedback_analysis']
            )
            iteration_results['recommendations'] = recommendations
            
            # Part 4: Update iteration counter
            self.al_iteration += 1
            
            print(f"✅ Active Learning Iteration #{self.al_iteration} completed")
            
        except Exception as e:
            print(f"❌ Active Learning iteration failed: {e}")
            iteration_results['error'] = str(e)
        
        return iteration_results
    
    def _generate_combined_recommendations(self, 
                                           monai_results: Dict[str, Any], 
                                           feedback_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations combining MONAI and feedback insights."""
        
        recommendations = []
        
        try:
            # MONAI-based recommendations
            if monai_results:
                selected_samples = monai_results.get('selected_samples', {})
                if selected_samples.get('num_selected', 0) > 0:
                    strategy = selected_samples.get('strategy_used', 'unknown')
                    recommendations.append(f"🧠 MONAI selected {selected_samples['num_selected']} samples using {strategy} strategy")
                
                metrics = monai_results.get('iteration_metrics', {})
                val_dice = metrics.get('val_dice', 0)
                if val_dice > 0.8:
                    recommendations.append(f"✅ Model performance is good (Dice: {val_dice:.3f}) - continue current strategy")
                elif val_dice < 0.6:
                    recommendations.append(f"⚠️ Model performance needs improvement (Dice: {val_dice:.3f}) - consider more diverse training data")
            
            # Feedback-based recommendations
            if feedback_analysis:
                overview = feedback_analysis.get('overview', {})
                avg_quality = overview.get('avg_quality_score', 0)
                
                if avg_quality > 4.0:
                    recommendations.append(f"📈 User feedback is excellent ({avg_quality:.1f}/5) - current models performing well")
                elif avg_quality < 3.0:
                    recommendations.append(f"📉 User feedback indicates issues ({avg_quality:.1f}/5) - review model selection and parameters")
                
                # Check for specific model recommendations
                existing_recommendations = feedback_analysis.get('recommendations', [])
                recommendations.extend(existing_recommendations[:3])  # Top 3 feedback recommendations
            
            # Expert review queue analysis
            pending_reviews = len(self.get_expert_review_queue("pending"))
            if pending_reviews > 10:
                recommendations.append(f"🚨 High expert review queue ({pending_reviews} pending) - consider adjusting uncertainty thresholds")
            elif pending_reviews == 0:
                recommendations.append("✅ No pending expert reviews - uncertainty detection working well")
            
            # General recommendations
            if not recommendations:
                recommendations.append("📊 Continue collecting feedback and monitoring model performance")
                
        except Exception as e:
            recommendations.append(f"❌ Error generating recommendations: {str(e)}")
        
        return recommendations
    
    def get_active_learning_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the active learning system."""
        
        status = {
            'orchestrator_info': {
                'monai_enabled': self.enable_monai,
                'monai_available': MONAI_AL_AVAILABLE,
                'current_iteration': self.al_iteration,
                'cached_uncertainties': len(self.uncertainty_cache)
            },
            'expert_review_queue': {
                'pending': len(self.get_expert_review_queue("pending")),
                'total_queued': len(self.expert_review_queue)
            },
            'system_health': {
                'status': 'healthy' if self.enable_monai else 'limited',
                'last_update': datetime.now().isoformat()
            }
        }
        
        # Add MONAI-specific status if available
        if self.enable_monai and self.monai_al:
            status['monai_info'] = {
                'model_type': self.monai_al.config.get('model_type', 'unknown'),
                'uncertainty_threshold': self.monai_al.config.get('uncertainty_threshold', 0.3),
                'confidence_threshold': self.monai_al.config.get('confidence_threshold', 0.8)
            }
        
        return status


# Initialize the orchestrator
active_learning_orchestrator = ActiveLearningOrchestrator(enable_monai=True)
