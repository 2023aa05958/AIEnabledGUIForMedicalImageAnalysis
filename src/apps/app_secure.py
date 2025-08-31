import os
import sys
from io import BytesIO
from datetime import datetime

import gradio as gr
import requests
from PIL import Image, ImageDraw

# Add the parent directories to the path to access sam_libs and auth
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from libs.sam_libs import SAM_MODELS
from libs.sam_libs.server import PORT, server
from auth.auth import user_manager, require_auth, get_user_greeting, check_permission

# 🎛️ DEMO MODE CONFIGURATION
# Change these settings to control demo complexity
DEMO_MODE = "ADVANCED"  # Options: "SIMPLE", "BASIC_AL", "ADVANCED"
FORCE_DISABLE_MONAI = False  # Set True to disable MONAI even if installed

print(f"🎬 Demo Mode: {DEMO_MODE}")
print(f"🧠 MONAI Override: {'Disabled' if FORCE_DISABLE_MONAI else 'Auto-detect'}")

# Import active learning modules
try:
    from active_learning.feedback_manager import feedback_manager
    from active_learning.performance_analytics import performance_analytics
    from active_learning.al_orchestrator import active_learning_orchestrator
    
    # Apply demo mode settings
    if DEMO_MODE == "SIMPLE":
        ACTIVE_LEARNING_ENABLED = False
        MONAI_AL_ENABLED = False
        print("🎬 SIMPLE DEMO: AL features disabled")
    elif DEMO_MODE == "BASIC_AL":
        ACTIVE_LEARNING_ENABLED = True
        MONAI_AL_ENABLED = False
        print("🎬 BASIC AL DEMO: Feedback & analytics enabled, MONAI disabled")
    else:  # ADVANCED
        ACTIVE_LEARNING_ENABLED = True
        MONAI_AL_ENABLED = (not FORCE_DISABLE_MONAI) and active_learning_orchestrator.enable_monai
        print("🎬 ADVANCED DEMO: All AL features enabled")
    
    print("✅ Active Learning modules loaded successfully")
    print(f"📊 Basic AL: {'Enabled' if ACTIVE_LEARNING_ENABLED else 'Disabled'}")
    print(f"🧠 MONAI AL: {'Enabled' if MONAI_AL_ENABLED else 'Disabled'}")
    
except ImportError as e:
    print(f"⚠️ Active Learning modules not available: {e}")
    ACTIVE_LEARNING_ENABLED = False
    MONAI_AL_ENABLED = False
    print("🎬 Falling back to SIMPLE DEMO mode")


def login_user(username: str, password: str):
    """Handle user login."""
    if not username or not password:
        return None, None, "❌ Please enter both username and password", None, None, None, None, gr.Accordion(visible=False)
    
    user_info = user_manager.authenticate(username, password)
    if user_info:
        session_id = user_manager.create_session(user_info)
        greeting = get_user_greeting(user_info)
        
        # Check if user is admin to show admin controls
        is_admin = check_permission(user_info, "can_manage_users")
        
        # Return success state and session info
        return (
            gr.Column(visible=False),  # Hide login
            gr.Column(visible=True),   # Show main app
            f"✅ {greeting}",
            session_id,
            user_info["username"],
            user_info["role"],
            user_info["full_name"],
            gr.Accordion(visible=is_admin)  # Show admin section only for admins
        )
    else:
        return (
            gr.Column(visible=True),   # Keep login visible
            gr.Column(visible=False),  # Keep main app hidden
            "❌ Invalid username or password",
            None,
            None,
            None,
            None,
            gr.Accordion(visible=False)  # Hide admin section
        )


def logout_user(session_id: str):
    """Handle user logout."""
    if session_id:
        user_manager.logout(session_id)
    
    return (
        gr.Column(visible=True),   # Show login
        gr.Column(visible=False),  # Hide main app
        "👋 Logged out successfully",
        None,  # Clear session
        None,  # Clear username
        None,  # Clear role
        None,  # Clear full name
        gr.Accordion(visible=False)  # Hide admin section
    )


# Global variable to store last prediction data
_last_prediction_data = {
    'confidence_score': 0.0,
    'processing_time': 0.0
}

@require_auth
def inference(sam_type, box_threshold, text_threshold, image, text_prompt, clicked_points, 
              annotated_image, bbox_coords, analysis_mode, session_id, user_info=None):
    """Gradio function that makes a request to the /predict LitServe endpoint."""
    import time
    
    # Start timing the prediction
    start_time = time.time()
    
    # Check if user has permission to analyze
    if not check_permission(user_info, "can_analyze"):
        return None, "❌ You don't have permission to perform analysis"
    
    url = f"http://localhost:{PORT}/predict"

    # Use the original image file for processing (not the annotated one)
    # This ensures the output doesn't contain user's drawn bounding boxes
    image_file = image
    
    if not image_file:
        return None, "❌ No image provided"

    try:
        # Prepare the multipart form data
        with open(image_file, "rb") as img_file:
            files = {
                "image": img_file,
            }
            
            # Prepare data based on analysis mode
            data = {
                "sam_type": sam_type,
                "box_threshold": str(box_threshold),
                "text_threshold": str(text_threshold),
                "text_prompt": text_prompt,
                "user": user_info["username"],  # Add user info to request
                "role": user_info["role"]
            }
            
            # Add bounding box if provided (for targeted segmentation)
            if analysis_mode in ["Bounding Box", "Combined (Text + Box)"] and bbox_coords and bbox_coords.strip():
                # Clean up bounding box format - remove parentheses and extra spaces
                cleaned_bbox = bbox_coords.strip().replace('(', '').replace(')', '').replace(' ', '')
                data["bounding_box"] = cleaned_bbox
                print(f"🎯 TARGETED FILTERING: Will filter GDINO detections within region {cleaned_bbox}")
                print("💡 TIP: First try 'Text Prompt' mode to see all GDINO detections, then use bounding box to filter")
            
            # Add point prompts if provided
            elif analysis_mode in ["Point Annotations", "Combined (Text + Points)"]:
                sam_prompts = get_sam_prompts(clicked_points)
                data["point_prompts"] = str(sam_prompts["points"]) if sam_prompts["points"] else ""
                data["point_labels"] = str(sam_prompts["labels"]) if sam_prompts["labels"] else ""

            try:
                # Time the actual prediction request
                request_start = time.time()
                response = requests.post(url, files=files, data=data)
                request_end = time.time()
                request_duration = request_end - request_start
                print(f"⏱️ Prediction request took: {request_duration:.2f} seconds")
            except Exception as e:
                print(f"Request failed: {e}")
                return None, f"❌ Server connection failed: {str(e)}"

        if response.status_code == 200:
            try:
                # Check if response is JSON (new format) or image (fallback)
                content_type = response.headers.get('content-type', '')
                
                if 'application/json' in content_type:
                    # New JSON response format with IoU data
                    import base64
                    response_data = response.json()
                    
                    # Decode base64 image
                    image_data = base64.b64decode(response_data.get('image', ''))
                    output_image = Image.open(BytesIO(image_data)).convert("RGB")
                    
                    # Extract metrics
                    iou_score = response_data.get('iou_score', 0.0)
                    mask_area = response_data.get('mask_area', 0.0)
                    num_detections = response_data.get('num_detections', 0)
                    
                    print(f"📊 Segmentation Metrics:")
                    print(f"   - IoU Score: {iou_score:.3f}")
                    print(f"   - Mask Area: {mask_area:.1f} pixels")
                    print(f"   - Detections: {num_detections}")
                    
                else:
                    # Fallback to image-only response
                    output_image = Image.open(BytesIO(response.content)).convert("RGB")
                    iou_score = 0.0
                    mask_area = 0.0
                    num_detections = 0
                
                # Save the output image to data/images/annotated folder with user info
                output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "images", "annotated")
                os.makedirs(output_dir, exist_ok=True)
                
                # Generate filename based on input image name, user, and timestamp
                input_filename = os.path.splitext(os.path.basename(image_file))[0]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                username = user_info["username"]
                output_filename = f"{input_filename}_{username}_medical_segmented_{timestamp}.jpg"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save the image
                output_image.save(output_path, "JPEG", quality=95)
                print(f"Medical image segmentation saved to: {output_path}")
                
                # Calculate total processing time
                end_time = time.time()
                total_processing_time = end_time - start_time
                print(f"⏱️ Total processing time: {total_processing_time:.2f} seconds")
                
                # 🧠 MONAI Active Learning Integration - Analyze prediction uncertainty
                uncertainty_message = ""
                confidence_score = 0.0  # Default confidence score
                if ACTIVE_LEARNING_ENABLED and MONAI_AL_ENABLED:
                    try:
                        # Create prediction result dict for uncertainty analysis
                        prediction_result = {
                            'image_path': image_file,
                            'output_path': output_path,
                            'text_prompt': text_prompt,
                            'analysis_mode': analysis_mode,
                            'bbox_coords': bbox_coords,
                            'processing_time': total_processing_time,  # Use actual timing
                            'response_status': response.status_code,
                            'iou_score': iou_score,
                            'mask_area': mask_area,
                            'num_detections': num_detections
                        }
                        
                        # Analyze uncertainty
                        uncertainty_analysis = active_learning_orchestrator.analyze_prediction_uncertainty(
                            image_path=image_file,
                            prediction_result=prediction_result,
                            model_used=sam_type
                        )
                        
                        # Extract confidence score from uncertainty analysis
                        confidence_score = uncertainty_analysis.get('final_confidence_score', 0.0)
                        if confidence_score == 0.0:
                            # Try alternative confidence score fields
                            confidence_score = uncertainty_analysis.get('confidence_score', 0.0)
                        
                        # Check if expert review is needed
                        if uncertainty_analysis.get('needs_expert_review', False):
                            review_id = active_learning_orchestrator.queue_for_expert_review(
                                image_path=image_file,
                                uncertainty_analysis=uncertainty_analysis,
                                user_info=user_info,
                                priority="high" if uncertainty_analysis.get('final_uncertainty_score', 0) > 0.7 else "medium"
                            )
                            
                            uncertainty_message = f"\n🧠 AI Assessment: {uncertainty_analysis.get('recommendation', '')}"
                            uncertainty_message += f"\n📋 Expert review queued (ID: {review_id[:8]}...)"
                        else:
                            confidence_level = uncertainty_analysis.get('confidence_level', 'Unknown')
                            uncertainty_message = f"\n🧠 AI Confidence: {confidence_level}"
                        
                        print(f"🧠 MONAI AL: Uncertainty analysis completed - {uncertainty_analysis.get('recommendation', 'No recommendation')}")
                        print(f"🎯 Confidence Score: {confidence_score:.2f}")
                        
                    except Exception as e:
                        print(f"⚠️ MONAI AL uncertainty analysis failed: {e}")
                        uncertainty_message = "\n⚠️ Uncertainty analysis unavailable"
                
                # Store the confidence score and processing time for later use in feedback
                global _last_prediction_data
                _last_prediction_data = {
                    'confidence_score': confidence_score,
                    'processing_time': total_processing_time
                }
                
                # Create comprehensive status message with IoU score
                status_message = f"✅ Analysis complete! Saved to: {output_filename}"
                
                # Add IoU information to status
                if iou_score > 0:
                    status_message += f"\n📊 IoU Score: {iou_score:.3f} | Detections: {num_detections} | Area: {mask_area:.0f}px²"
                
                # Add uncertainty message
                status_message += uncertainty_message
                
                return output_image, status_message
                
            except Exception as e:
                print(f"Failed to process response image: {e}")
                return None, f"❌ Image processing failed: {str(e)}"
        else:
            print(f"Request failed with status code {response.status_code}: {response.text}")
            return None, f"❌ Server error ({response.status_code}): {response.text}"
            
    except Exception as e:
        return None, f"❌ Unexpected error: {str(e)}"


def get_sam_prompts(points):
    """Convert clicked points to SAM-compatible format."""
    if not points or len(points) == 0:
        return {"points": [], "labels": []}
    
    # Convert points to the format SAM expects: [[x, y], [x, y], ...]
    sam_points = []
    sam_labels = []
    
    for point in points:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            sam_points.append([int(point[0]), int(point[1])])
            sam_labels.append(1)  # 1 for foreground points
    
    return {"points": sam_points, "labels": sam_labels}


def process_click(image_path, clicked_points, evt: gr.SelectData):
    """Process image clicks and update visualization."""
    if not image_path:
        return None, clicked_points, "❌ Please upload an image first"
    
    # Add the new click point
    new_point = [evt.index[0], evt.index[1]]
    updated_points = clicked_points + [new_point]
    
    # Load and update the image with visual markers
    try:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        
        # Draw all clicked points
        for i, point in enumerate(updated_points):
            x, y = point[0], point[1]
            # Draw a circle for each point
            radius = 8
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                        fill="red", outline="white", width=2)
            # Draw point number
            draw.text((x+10, y-10), str(i+1), fill="white")
        
        status_msg = f"✅ {len(updated_points)} ROI point(s) marked"
        return image, updated_points, status_msg
        
    except Exception as e:
        return None, clicked_points, f"❌ Error processing click: {str(e)}"


def clear_points(image_path):
    """Clear all annotation points."""
    if not image_path:
        return None, [], "❌ No image loaded"
    
    try:
        # Reload original image
        original_image = Image.open(image_path).convert("RGB")
        return original_image, [], "✅ All annotations cleared"
    except Exception as e:
        return None, [], f"❌ Error clearing annotations: {str(e)}"


def process_bbox_click(image_path, bbox_points, evt: gr.SelectData):
    """Process clicks for bounding box creation (two-click method)."""
    if not image_path:
        return None, bbox_points, "❌ Please upload an image first", "No bounding box created yet", ""
    try:
        new_point = [evt.index[0], evt.index[1]]
        if len(bbox_points) == 0:
            updated_points = [new_point]
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            x, y = new_point
            draw.ellipse([x-5, y-5, x+5, y+5], fill="red", outline="darkred", width=2)
            draw.text((x+10, y-10), "1", fill="red")
            return image, updated_points, "📦 First corner set. Click again for second corner.", "Creating bounding box...", ""
        elif len(bbox_points) == 1:
            updated_points = bbox_points + [new_point]
            x1, y1 = bbox_points[0]
            x2, y2 = new_point
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_y, max_y = min(y1, y2), max(y1, y2)
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            draw.rectangle([min_x, min_y, max_x, max_y], outline="red", width=3)
            draw.ellipse([min_x-5, min_y-5, min_x+5, min_y+5], fill="red", outline="darkred", width=2)
            draw.ellipse([max_x-5, max_y-5, max_x+5, max_y+5], fill="red", outline="darkred", width=2)
            draw.text((min_x+10, min_y-10), "1", fill="red")
            draw.text((max_x+10, max_y-10), "2", fill="red")
            bbox_coords = f"{min_x},{min_y},{max_x},{max_y}"
            bbox_display = f"Bounding Box: ({min_x},{min_y}) to ({max_x},{max_y})"
            return image, updated_points, f"✅ Bounding box created: {bbox_coords}", bbox_display, bbox_coords
        else:
            return process_bbox_click(image_path, [], evt)
    except Exception as e:
        return None, bbox_points, f"❌ Error creating bounding box: {str(e)}", "Error creating bounding box", ""


def clear_bbox(image_path):
    """Clear the bounding box and reset to original image and clear the bbox_coords textbox."""
    if not image_path:
        return None, [], "❌ No image loaded", "No bounding box created yet", ""
    try:
        original_image = Image.open(image_path).convert("RGB")
        return original_image, [], "✅ Bounding box cleared", "No bounding box created yet", ""
    except Exception as e:
        return None, [], f"❌ Error clearing bounding box: {str(e)}", "Error clearing bounding box", ""


def unified_click_handler(image_path, clicked_points, bbox_points, bbox_mode_text, evt: gr.SelectData):
    """Handle image clicks for both point annotation and bounding box creation."""
    if not image_path:
        return None, clicked_points, bbox_points, "❌ Please upload an image first", ""
    
    if "Back to Point Mode" in bbox_mode_text:
        # Bounding box mode
        result = process_bbox_click(image_path, bbox_points, evt)
        return result[0], clicked_points, result[1], result[2], result[4]
    else:
        # Point annotation mode
        result = process_click(image_path, clicked_points, evt)
        return result[0], result[1], bbox_points, "No bounding box created yet", ""


def toggle_bbox_mode(current_mode, image_path):
    """Toggle between normal annotation mode and bounding box mode."""
    if not image_path:
        return (
            gr.Button("📦 Enable Bounding Box Mode"),
            gr.Button(visible=False),
            gr.HTML(visible=False),
            gr.Textbox(visible=False)
        )
    
    if "Enable" in current_mode:
        # Switch TO bounding box mode
        return (
            gr.Button("🎯 Back to Point Mode", variant="primary"),
            gr.Button(visible=True),
            gr.HTML(visible=True),
            gr.Textbox(visible=True)
        )
    else:
        # Switch back to normal mode
        return (
            gr.Button("📦 Enable Bounding Box Mode", variant="secondary"),
            gr.Button(visible=False),
            gr.HTML(visible=False),
            gr.Textbox(visible=False)
        )


def validate_inputs(image, text_prompt, analysis_mode, bbox_coords=""):
    """Validate user inputs before processing."""
    if not image:
        return False, "❌ Please upload an image first"
    
    # Text prompt is ALWAYS required because GDINO needs it to detect objects
    if analysis_mode in ["Text Prompt", "Combined (Text + Points)", "Combined (Text + Box)", "Bounding Box"] and not text_prompt.strip():
        return False, "❌ Please provide a text prompt - GDINO needs text to detect objects (e.g., 'brain tumor')"
    
    if analysis_mode in ["Bounding Box", "Combined (Text + Box)"] and not bbox_coords.strip():
        return False, "❌ Please provide bounding box coordinates (format: x1,y1,x2,y2)"
    
    # Validate bounding box format if provided
    if bbox_coords and bbox_coords.strip():
        try:
            # Clean up format - remove parentheses and extra spaces
            cleaned_coords = bbox_coords.strip().replace('(', '').replace(')', '').replace(' ', '')
            coords = [float(x.strip()) for x in cleaned_coords.split(',')]
            if len(coords) != 4:
                return False, "❌ Bounding box must have exactly 4 coordinates: x1,y1,x2,y2"
            if coords[2] <= coords[0] or coords[3] <= coords[1]:
                return False, "❌ Invalid bounding box: x2 > x1 and y2 > y1 required"
        except ValueError:
            return False, "❌ Bounding box coordinates must be numbers: x1,y1,x2,y2"
    
    return True, "✅ Inputs validated"


def collect_feedback(feedback_quality, feedback_type, clinical_notes, image_path, model_used, 
                    analysis_mode, text_prompt, bbox_coords, user_info, prediction_results, 
                    confidence_score, processing_time, feedback_submitted_state):
    """Collect and store user feedback for active learning."""
    if not ACTIVE_LEARNING_ENABLED:
        return "⚠️ Active learning not enabled", gr.Accordion(visible=False), False
    
    # Only researchers and radiologists can provide feedback
    if user_info.get("role") not in ["researcher", "radiologist"]:
        return "❌ Only researchers and radiologists can provide feedback", gr.Accordion(visible=False), False
    
    # Check if radiologist has already submitted feedback for this image
    if user_info.get("role") == "radiologist" and feedback_submitted_state:
        return "⚠️ Radiologists can only submit feedback once per image", gr.Accordion(visible=True), True
    
    if not feedback_quality:
        return "❌ Please select a quality rating", gr.Accordion(visible=True), feedback_submitted_state
    
    try:
        # Use stored prediction data if confidence_score and processing_time are not provided
        global _last_prediction_data
        if confidence_score is None or confidence_score == 0.0:
            confidence_score = _last_prediction_data.get('confidence_score', 0.0)
        if processing_time is None or processing_time == 0.0:
            processing_time = _last_prediction_data.get('processing_time', 0.0)
        
        print(f"📊 Using feedback data - Confidence: {confidence_score:.2f}, Processing Time: {processing_time:.2f}s")
        
        # Convert quality rating to numeric score
        quality_mapping = {
            "Excellent (5)": 5,
            "Good (4)": 4, 
            "Fair (3)": 3,
            "Poor (2)": 2,
            "Very Poor (1)": 1
        }
        
        feedback_data = {
            'user_id': user_info.get('username', ''),
            'user_role': user_info.get('role', ''),
            'image_path': image_path or '',
            'model_used': model_used or '',
            'analysis_mode': analysis_mode or '',
            'text_prompt': text_prompt or '',
            'bounding_box': bbox_coords or '',
            'prediction_results': prediction_results or {},
            'feedback_quality': quality_mapping.get(feedback_quality, 0),
            'feedback_type': feedback_type or '',
            'clinical_notes': clinical_notes or '',
            'confidence_score': confidence_score,
            'processing_time': processing_time
        }
        
        feedback_id = feedback_manager.store_feedback(feedback_data)
        
        if feedback_id:
            # Mark feedback as submitted for radiologists
            new_feedback_state = True if user_info.get("role") == "radiologist" else feedback_submitted_state
            return f"✅ Feedback submitted successfully! ID: {feedback_id[:8]}", gr.Accordion(visible=False), new_feedback_state
        else:
            return "❌ Failed to submit feedback", gr.Accordion(visible=True), feedback_submitted_state
            
    except Exception as e:
        print(f"❌ Error collecting feedback: {e}")
        return f"❌ Error submitting feedback: {str(e)}", gr.Accordion(visible=True), feedback_submitted_state


def get_performance_dashboard(user_info, days_filter):
    """Generate performance dashboard for researchers."""
    if not ACTIVE_LEARNING_ENABLED:
        return "⚠️ Active learning not enabled", "", ""
    
    # Only researchers can view performance analytics
    if user_info.get("role") != "researcher":
        return "❌ Only researchers can access performance analytics", "", ""
    
    try:
        # Get comprehensive analytics
        analytics = performance_analytics.get_comprehensive_analytics(days=int(days_filter))
        
        if not analytics:
            return "📊 No data available for the selected period", "", ""
        
        # Format overview
        overview = analytics.get('overview', {})
        overview_html = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
            <h3>📊 Performance Overview ({days_filter} days)</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; text-align: center;">
                    <h4>{overview.get('total_feedback', 0)}</h4>
                    <p>Total Feedback</p>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; text-align: center;">
                    <h4>{overview.get('avg_quality_score', 0)}/5</h4>
                    <p>Avg Quality Score</p>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; text-align: center;">
                    <h4>{overview.get('active_users', 0)}</h4>
                    <p>Active Users</p>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; text-align: center;">
                    <h4>{overview.get('models_evaluated', 0)}</h4>
                    <p>Models Evaluated</p>
                </div>
            </div>
            <div style="margin-top: 15px; text-align: center;">
                <span style="background: rgba(255,255,255,0.2); padding: 8px 15px; border-radius: 15px;">
                    📈 Trend: {overview.get('quality_trend', 'Unknown')} 
                    ({'+' if overview.get('trend_change', 0) >= 0 else ''}{overview.get('trend_change', 0)})
                </span>
            </div>
        </div>
        """
        
        # Format model comparison
        model_comparison = analytics.get('model_trends', [])
        comparison_html = "<h4>🏆 Model Performance Comparison</h4>"
        
        if model_comparison:
            # Group by model
            model_stats = {}
            for trend in model_comparison:
                model = trend['model']
                if model not in model_stats:
                    model_stats[model] = {
                        'total_predictions': 0,
                        'total_quality': 0,
                        'total_confidence': 0,
                        'total_time': 0,
                        'days': 0
                    }
                model_stats[model]['total_predictions'] += trend['predictions']
                model_stats[model]['total_quality'] += trend['avg_quality'] * trend['predictions']
                model_stats[model]['total_confidence'] += trend['avg_confidence'] * trend['predictions']
                model_stats[model]['total_time'] += trend['avg_processing_time'] * trend['predictions']
                model_stats[model]['days'] += 1
            
            comparison_html += "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; margin: 15px 0;'>"
            
            for model, stats in model_stats.items():
                if stats['total_predictions'] > 0:
                    avg_quality = stats['total_quality'] / stats['total_predictions']
                    avg_confidence = stats['total_confidence'] / stats['total_predictions'] if stats['total_predictions'] > 0 else 0.0
                    avg_time = stats['total_time'] / stats['total_predictions'] if stats['total_predictions'] > 0 else 0.0
                    
                    # Color based on quality
                    if avg_quality >= 4:
                        color = "#4CAF50"
                    elif avg_quality >= 3:
                        color = "#FF9800"
                    else:
                        color = "#F44336"
                    
                    # Format confidence and time display
                    confidence_display = f"{avg_confidence:.2f}" if avg_confidence > 0 else "N/A"
                    time_display = f"{avg_time:.2f}s" if avg_time > 0 else "N/A"
                    
                    comparison_html += f"""
                    <div style="border: 2px solid {color}; border-radius: 10px; padding: 15px; background: linear-gradient(135deg, {color}22 0%, {color}11 100%);">
                        <h5 style="margin: 0 0 10px 0; color: {color};">{model}</h5>
                        <p><strong>Quality:</strong> {avg_quality:.2f}/5</p>
                        <p><strong>Predictions:</strong> {stats['total_predictions']}</p>
                        <p><strong>Confidence:</strong> {confidence_display}</p>
                        <p><strong>Avg Time:</strong> {time_display}</p>
                    </div>
                    """
            
            comparison_html += "</div>"
        else:
            comparison_html += "<p style='color: #666; font-style: italic;'>No model comparison data available</p>"
        
        # Format recommendations
        recommendations = analytics.get('recommendations', [])
        recommendations_html = "<h4>💡 Recommendations</h4><ul>"
        for rec in recommendations:
            recommendations_html += f"<li style='margin: 8px 0; padding: 8px; background: #f0f8ff; border-left: 4px solid #2196F3; border-radius: 4px;'>{rec}</li>"
        recommendations_html += "</ul>"
        
        return overview_html, comparison_html, recommendations_html
        
    except Exception as e:
        print(f"❌ Error generating dashboard: {e}")
        return f"❌ Error generating dashboard: {str(e)}", "", ""


def get_monai_active_learning_dashboard(user_info):
    """Generate MONAI Active Learning dashboard for researchers."""
    if not ACTIVE_LEARNING_ENABLED:
        return "⚠️ Active learning not enabled"
    
    # Only researchers can view MONAI AL dashboard
    if user_info.get("role") != "researcher":
        return "❌ Only researchers can access MONAI Active Learning dashboard"
    
    try:
        # Get active learning status
        al_status = active_learning_orchestrator.get_active_learning_status()
        
        # Get expert review queue
        pending_reviews = active_learning_orchestrator.get_expert_review_queue("pending")
        completed_reviews = active_learning_orchestrator.get_expert_review_queue("completed")
        
        # Format the dashboard
        orchestrator_info = al_status.get('orchestrator_info', {})
        queue_info = al_status.get('expert_review_queue', {})
        system_health = al_status.get('system_health', {})
        
        dashboard_html = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
            <h3>🧠 MONAI Active Learning Dashboard</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; text-align: center;">
                    <h4>{'✅ Enabled' if orchestrator_info.get('monai_enabled') else '❌ Disabled'}</h4>
                    <p>MONAI Status</p>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; text-align: center;">
                    <h4>{orchestrator_info.get('current_iteration', 0)}</h4>
                    <p>AL Iterations</p>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; text-align: center;">
                    <h4>{queue_info.get('pending', 0)}</h4>
                    <p>Pending Reviews</p>
                </div>
                <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; text-align: center;">
                    <h4>{orchestrator_info.get('cached_uncertainties', 0)}</h4>
                    <p>Cached Analyses</p>
                </div>
            </div>
        </div>
        
        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h4>📋 Expert Review Queue</h4>
            <div style="margin-top: 15px;">
        """
        
        if pending_reviews:
            dashboard_html += "<h5>🔍 Pending Reviews:</h5><ul>"
            for review in pending_reviews[:5]:  # Show top 5
                uncertainty_score = review.get('uncertainty_analysis', {}).get('final_uncertainty_score', 0)
                confidence_level = review.get('uncertainty_analysis', {}).get('confidence_level', 'Unknown')
                dashboard_html += f"""
                <li style="margin-bottom: 10px; padding: 10px; background: #fff3cd; border-radius: 5px;">
                    <strong>ID:</strong> {review['id'][:8]}... | 
                    <strong>Priority:</strong> {review['priority'].title()} | 
                    <strong>Confidence:</strong> {confidence_level} |
                    <strong>Uncertainty:</strong> {uncertainty_score:.2f}
                    <br><small>Image: {os.path.basename(review['image_path'])}</small>
                </li>
                """
            dashboard_html += "</ul>"
        else:
            dashboard_html += "<p>✅ No pending expert reviews</p>"
        
        dashboard_html += "</div></div>"
        
        # Add MONAI-specific information if available
        monai_info = al_status.get('monai_info', {})
        if monai_info:
            dashboard_html += f"""
            <div style="background: #e7f3ff; padding: 20px; border-radius: 10px;">
                <h4>🧠 MONAI Configuration</h4>
                <ul>
                    <li><strong>Model Type:</strong> {monai_info.get('model_type', 'Unknown')}</li>
                    <li><strong>Uncertainty Threshold:</strong> {monai_info.get('uncertainty_threshold', 0.3)}</li>
                    <li><strong>Confidence Threshold:</strong> {monai_info.get('confidence_threshold', 0.8)}</li>
                </ul>
            </div>
            """
        
        # Add recent activity if available
        if orchestrator_info.get('current_iteration', 0) > 0:
            dashboard_html += f"""
            <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; margin-top: 20px;">
                <h5>📈 Recent Activity</h5>
                <p>Active Learning has completed {orchestrator_info.get('current_iteration', 0)} iteration(s).</p>
                <p>System Health: <strong>{system_health.get('status', 'Unknown').title()}</strong></p>
            </div>
            """
        
        return dashboard_html
        
    except Exception as e:
        print(f"❌ Error generating MONAI AL dashboard: {e}")
        return f"❌ Error generating MONAI Active Learning dashboard: {str(e)}"


def run_active_learning_iteration(user_info, strategy="hybrid"):
    """Run a new active learning iteration."""
    if not ACTIVE_LEARNING_ENABLED:
        return "⚠️ Active learning not enabled"
    
    # Only researchers can run AL iterations
    if user_info.get("role") != "researcher":
        return "❌ Only researchers can run active learning iterations"
    
    if not MONAI_AL_ENABLED:
        return "⚠️ MONAI Active Learning not available"
    
    try:
        print(f"🚀 Starting Active Learning iteration (strategy: {strategy})")
        results = active_learning_orchestrator.run_active_learning_iteration(strategy=strategy)
        
        # Format results
        iteration_num = results.get('iteration', 0)
        recommendations = results.get('recommendations', [])
        
        result_html = f"""
        <div style="background: #d4edda; padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h4>🚀 Active Learning Iteration #{iteration_num} Complete</h4>
            <p><strong>Strategy Used:</strong> {strategy.title()}</p>
            <p><strong>Timestamp:</strong> {results.get('timestamp', 'Unknown')}</p>
            
            <h5>💡 Recommendations:</h5>
            <ul>
        """
        
        for rec in recommendations:
            result_html += f"<li>{rec}</li>"
        
        result_html += """
            </ul>
        </div>
        """
        
        # Add MONAI results if available
        monai_results = results.get('monai_results', {})
        if monai_results:
            result_html += f"""
            <div style="background: #e7f3ff; padding: 15px; border-radius: 8px; margin-top: 10px;">
                <h5>🧠 MONAI Results:</h5>
                <pre>{monai_results}</pre>
            </div>
            """
        
        return result_html
        
    except Exception as e:
        print(f"❌ Error running AL iteration: {e}")
        return f"❌ Error running active learning iteration: {str(e)}"


def get_analysis_instructions(mode):
    """Get instructions based on the selected analysis mode."""
    if mode == "Text Prompt":
        return """
        <div style='padding: 10px; background: #e8f4f8; border-radius: 8px; margin: 10px 0;'>
            <strong>📝 Text Prompt Mode:</strong><br>
            <small>Use descriptive medical terms to guide the segmentation. Examples: 'brain tumor', 'lung nodule', 'heart ventricle'</small>
        </div>
        """
    elif mode == "Point Annotations":
        return """
        <div style='padding: 10px; background: #f0f8e8; border-radius: 8px; margin: 10px 0;'>
            <strong>🎯 Point Annotations Mode:</strong><br>
            <small>Click points on the uploaded image above to guide the segmentation. Your clicks will be used as prompts for SAM.</small>
        </div>
        """
    elif mode == "Bounding Box":
        return """
        <div style='padding: 10px; background: #fff0e8; border-radius: 8px; margin: 10px 0;'>
            <strong>📦 Bounding Box Guidance Mode:</strong><br>
            <small><strong>Architecture:</strong> Text → GDINO (detects objects) → User Box (filters detections) → SAM (segments)<br>
            Enter coordinates (x1,y1,x2,y2) to focus GDINO detections within a specific region. Still requires text prompt!</small>
        </div>
        """
    elif mode == "Combined (Text + Points)":
        return """
        <div style='padding: 10px; background: #f8f0e8; border-radius: 8px; margin: 10px 0;'>
            <strong>🔄 Combined Text + Points:</strong><br>
            <small>Use both text descriptions AND point annotations for the most precise segmentation results.</small>
        </div>
        """
    else:  # Combined (Text + Box)
        return """
        <div style='padding: 10px; background: #f0e8f8; border-radius: 8px; margin: 10px 0;'>
            <strong>🎯 Combined Text + Box Guidance:</strong><br>
            <small><strong>Full Pipeline:</strong> Text → GDINO (detects) → User Box (filters) → SAM (segments)<br>
            Use text descriptions with bounding box coordinates to filter GDINO detections within a specific region.</small>
        </div>
        """


# Admin Management Functions
def create_new_user(username, password, role, full_name, email, session_id):
    """Create a new user (admin only)."""
    # Validate session and admin permissions
    user_info = user_manager.validate_session(session_id)
    if not user_info or not check_permission(user_info, "can_manage_users"):
        return "❌ Access denied. Administrator privileges required.", get_user_list(session_id)
    
    if not username or not password:
        return "❌ Username and password are required.", get_user_list(session_id)
    
    success = user_manager.create_user(username, password, role, full_name, email)
    if success:
        user_manager.save_users()
        return f"✅ User '{username}' created successfully with role '{role}'.", get_user_list(session_id)
    else:
        return f"❌ Failed to create user. Username '{username}' may already exist.", get_user_list(session_id)


def get_user_list(session_id):
    """Get list of all users (admin only)."""
    user_info = user_manager.validate_session(session_id)
    if not user_info or not check_permission(user_info, "can_manage_users"):
        return []
    
    user_data = []
    for username, data in user_manager.users.items():
        user_data.append([
            username,
            data["role"],
            data.get("full_name", ""),
            data.get("email", ""),
            data.get("active", True),
            data.get("last_login", "Never") or "Never"
        ])
    
    return user_data


def toggle_user_status(username, session_id):
    """Toggle user active status (admin only)."""
    user_info = user_manager.validate_session(session_id)
    if not user_info or not check_permission(user_info, "can_manage_users"):
        return "❌ Access denied. Administrator privileges required.", get_user_list(session_id)
    
    if not username:
        return "❌ Please enter a username.", get_user_list(session_id)
    
    if username not in user_manager.users:
        return f"❌ User '{username}' not found.", get_user_list(session_id)
    
    # Prevent admin from deactivating themselves
    if username == user_info["username"]:
        return "❌ You cannot deactivate your own account.", get_user_list(session_id)
    
    current_status = user_manager.users[username].get("active", True)
    user_manager.users[username]["active"] = not current_status
    user_manager.save_users()
    
    status_text = "activated" if not current_status else "deactivated"
    return f"✅ User '{username}' has been {status_text}.", get_user_list(session_id)


def get_role_statistics(session_id):
    """Get role-based statistics (admin only)."""
    user_info = user_manager.validate_session(session_id)
    if not user_info or not check_permission(user_info, "can_manage_users"):
        return "❌ Access denied"
    
    # Count users by role
    total_users = len(user_manager.users)
    role_counts = {}
    active_users = 0
    
    for username, data in user_manager.users.items():
        role = data["role"]
        role_counts[role] = role_counts.get(role, 0) + 1
        if data.get("active", True):
            active_users += 1
    
    stats_html = f"""
    <div style='padding: 15px; background: #e8f5e8; border-radius: 8px; border-left: 4px solid #28a745;'>
        <strong>📊 User Statistics:</strong><br>
        <small style='color: #155724;'>
        • <b>Total Users:</b> {total_users}<br>
        • <b>Active Users:</b> {active_users}<br>
        • <b>Inactive Users:</b> {total_users - active_users}<br><br>
        <b>By Role:</b><br>
        {''.join([f"• <b>{role.title()}:</b> {count}<br>" for role, count in role_counts.items()])}
        </small>
    </div>
    """
    return stats_html


def get_permission_matrix(session_id):
    """Get permission matrix for all roles (admin only)."""
    user_info = user_manager.validate_session(session_id)
    if not user_info or not check_permission(user_info, "can_manage_users"):
        return "❌ Access denied"
    
    roles = ["administrator", "radiologist", "researcher", "guest"]
    permissions = [
        "can_analyze", "can_view_all_results", "can_manage_users", 
        "can_export_data", "can_modify_settings", "can_view_system_logs"
    ]
    
    matrix_html = """
    <div style='padding: 15px; background: #f0f8ff; border-radius: 8px; border-left: 4px solid #007bff;'>
        <strong>🔐 Permission Matrix:</strong><br>
        <table style='width: 100%; margin-top: 10px; border-collapse: collapse;'>
            <tr style='background: #e9ecef;'>
                <th style='padding: 8px; border: 1px solid #dee2e6; text-align: left;'>Role</th>
                <th style='padding: 8px; border: 1px solid #dee2e6; text-align: center;'>Analyze</th>
                <th style='padding: 8px; border: 1px solid #dee2e6; text-align: center;'>View Results</th>
                <th style='padding: 8px; border: 1px solid #dee2e6; text-align: center;'>Manage Users</th>
                <th style='padding: 8px; border: 1px solid #dee2e6; text-align: center;'>Export</th>
                <th style='padding: 8px; border: 1px solid #dee2e6; text-align: center;'>Settings</th>
                <th style='padding: 8px; border: 1px solid #dee2e6; text-align: center;'>Logs</th>
            </tr>
    """
    
    for role in roles:
        role_permissions = user_manager.get_role_permissions(role)
        matrix_html += f"""
            <tr>
                <td style='padding: 8px; border: 1px solid #dee2e6; font-weight: bold;'>{role.title()}</td>
                <td style='padding: 8px; border: 1px solid #dee2e6; text-align: center;'>{'✅' if role_permissions.get('can_analyze') else '❌'}</td>
                <td style='padding: 8px; border: 1px solid #dee2e6; text-align: center;'>{'✅' if role_permissions.get('can_view_all_results') else '❌'}</td>
                <td style='padding: 8px; border: 1px solid #dee2e6; text-align: center;'>{'✅' if role_permissions.get('can_manage_users') else '❌'}</td>
                <td style='padding: 8px; border: 1px solid #dee2e6; text-align: center;'>{'✅' if role_permissions.get('can_export_data') else '❌'}</td>
                <td style='padding: 8px; border: 1px solid #dee2e6; text-align: center;'>{'✅' if role_permissions.get('can_modify_settings') else '❌'}</td>
                <td style='padding: 8px; border: 1px solid #dee2e6; text-align: center;'>{'✅' if role_permissions.get('can_view_system_logs') else '❌'}</td>
            </tr>
        """
    
    matrix_html += """
        </table>
    </div>
    """
    return matrix_html


# Create the Gradio interface
with gr.Blocks(
    title="AI-Enabled GUI for Medical Image Analysis",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1400px !important;
        margin: auto !important;
        padding: 20px !important;
    }
    .title-text {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        background: linear-gradient(45deg, #4a90e2 0%, #2c3e50 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .subtitle-text {
        text-align: center;
        font-size: 1.2em;
        color: #666;
        margin-bottom: 30px;
    }
    .login-container {
        max-width: 400px;
        margin: 50px auto;
        padding: 30px;
        background: #f8fafb;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #4a90e2;
    }
    .main-button {
        background: linear-gradient(45deg, #4a90e2 0%, #2c3e50 100%) !important;
        border: none !important;
        border-radius: 25px !important;
        color: white !important;
        font-size: 1.1em !important;
        font-weight: bold !important;
        padding: 15px 30px !important;
        margin: 20px auto !important;
        display: block !important;
        min-width: 200px !important;
        box-shadow: 0 4px 15px rgba(74, 144, 226, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    .main-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(74, 144, 226, 0.4) !important;
    }
    .control-panel {
        background: #f8fafb;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #4a90e2;
    }
    .image-panel {
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        overflow: hidden;
        border: 2px solid #e8f4f8;
        min-height: 300px;
    }
    .image-panel img {
        max-width: 100%;
        height: auto;
        cursor: crosshair;
    }
    .user-info {
        background: linear-gradient(45deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 10px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .logout-btn {
        background: #dc3545 !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 5px 15px !important;
        font-size: 0.9em !important;
    }
    """
) as blocks:
    
    # Session state variables
    session_id = gr.State(None)
    current_user = gr.State(None)
    current_role = gr.State(None)
    current_full_name = gr.State(None)
    
    # Header
    gr.HTML("""
        <div class="title-text">🏥 AI-Enabled GUI for Medical Image Analysis</div>
        <div class="subtitle-text">Advanced Medical Image Analysis & AI-Powered Segmentation</div>
    """)
    
    # Login Section
    with gr.Column(visible=True, elem_classes="login-container") as login_section:
        gr.HTML("<h2 style='text-align: center; color: #2c3e50; margin-top: 0;'>🏥 Medical Professional Login</h2>")
        
        gr.HTML("""
            <div style='padding: 15px; background: #e8f4f8; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #4a90e2;'>
                <strong>� Secure Access:</strong><br>
                <small style='color: #444;'>
                Please enter your authorized medical professional credentials to access the AI segmentation platform.
                </small>
            </div>
        """)
        
        username_input = gr.Textbox(
            label="👤 Username",
            placeholder="Enter your username",
            scale=1
        )
        password_input = gr.Textbox(
            label="🔒 Password", 
            type="password",
            placeholder="Enter your password",
            scale=1
        )
        
        login_btn = gr.Button(
            "🔓 Login to Medical Platform",
            variant="primary",
            elem_classes=["main-button"]
        )
        
        login_status = gr.Textbox(
            label="Status",
            interactive=False,
            value="Please enter your credentials to access the medical annotation platform"
        )
    
    # Main Application Section (initially hidden)
    with gr.Column(visible=False) as main_section:
        
        # User Info Bar
        with gr.Row(elem_classes="user-info"):
            with gr.Column(scale=4):
                user_greeting = gr.HTML("Welcome!")
            with gr.Column(scale=1):
                logout_btn = gr.Button(
                    "🚪 Logout",
                    variant="secondary",
                    elem_classes=["logout-btn"]
                )
        
        # Admin Controls Section (visible only to administrators)
        with gr.Accordion("👨‍💼 Admin Controls", open=False, visible=False) as admin_section:
            gr.HTML("""
                <div style='padding: 15px; background: #fff3cd; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #ffc107;'>
                    <strong>⚠️ Administrator Access:</strong><br>
                    <small style='color: #856404;'>
                    This section is only visible to administrators. Use these controls to manage users and assign role-based permissions.
                    </small>
                </div>
            """)
            
            with gr.Tabs():
                # User Management Tab
                with gr.Tab("👥 User Management"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.HTML("<h4>➕ Create New User</h4>")
                            new_username = gr.Textbox(label="Username", placeholder="Enter username")
                            new_password = gr.Textbox(label="Password", type="password", placeholder="Enter password")
                            new_role = gr.Dropdown(
                                choices=["administrator", "radiologist", "researcher", "guest"],
                                label="Role",
                                value="researcher"
                            )
                            new_full_name = gr.Textbox(label="Full Name", placeholder="Enter full name")
                            new_email = gr.Textbox(label="Email", placeholder="Enter email")
                            create_user_btn = gr.Button("➕ Create User", variant="primary")
                            create_user_status = gr.Textbox(label="Status", interactive=False)
                        
                        with gr.Column(scale=2):
                            gr.HTML("<h4>👥 Manage Existing Users</h4>")
                            user_list = gr.Dataframe(
                                headers=["Username", "Role", "Full Name", "Email", "Active", "Last Login"],
                                datatype=["str", "str", "str", "str", "bool", "str"],
                                interactive=False,
                                label="User List"
                            )
                            with gr.Row():
                                refresh_users_btn = gr.Button("🔄 Refresh List", variant="secondary")
                                selected_user = gr.Textbox(label="Username to Toggle", placeholder="Enter username")
                                toggle_user_btn = gr.Button("🔄 Toggle Active Status", variant="secondary")
                            user_management_status = gr.Textbox(label="Status", interactive=False)
                
                # Group Management Tab  
                with gr.Tab("🏢 Group Management"):
                    gr.HTML("""
                        <div style='padding: 15px; background: #e8f4f8; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #4a90e2;'>
                            <strong>🏢 Role-Based Access Control:</strong><br>
                            <small style='color: #444;'>
                            Configure permissions and access levels for different user roles in the medical annotation system.
                            </small>
                        </div>
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            gr.HTML("<h4>📊 Role Statistics</h4>")
                            role_stats = gr.HTML()
                            
                        with gr.Column():
                            gr.HTML("<h4>🔐 Permission Matrix</h4>")
                            permission_matrix = gr.HTML()
                    
                    refresh_stats_btn = gr.Button("🔄 Refresh Statistics", variant="secondary")
        
        # Project Architecture Section
        with gr.Accordion("🏗️ Project Architecture & How It Works", open=False):
            gr.HTML("""
                <div style='padding: 20px; background: #f8f9fa; border-radius: 10px; margin: 10px 0;'>
                    <h3 style='color: #2c3e50; margin-top: 0;'>🎯 Brain Tumor Detection Pipeline</h3>
                    
                    <div style='margin-bottom: 20px; padding: 15px; background: #e8f4f8; border-radius: 8px; border-left: 4px solid #4a90e2;'>
                        <strong>🔄 Core Architecture (3-Step Pipeline):</strong><br>
                        <div style='margin: 10px 0; padding: 10px; background: #f0f8ff; border-radius: 5px;'>
                            <strong>Step 1:</strong> 📝 <b>User Input</b> → Image + Text Prompt (e.g., "brain tumor")<br>
                            <strong>Step 2:</strong> 🎯 <b>GDINO</b> → Text → Object Detection → Bounding Boxes<br>
                            <strong>Step 3:</strong> ✂️ <b>SAM</b> → Image + GDINO Boxes → Precise Segmentation Masks
                        </div>
                    </div>
                    
                    <div style='margin-bottom: 20px; padding: 15px; background: #e8f5e8; border-radius: 8px; border-left: 4px solid #28a745;'>
                        <strong>🔐 User Access Control:</strong><br>
                        <small style='color: #444;'>
                        • <b>Administrator:</b> Full system access, user management, system logs<br>
                        • <b>Radiologist:</b> Medical analysis, view all results, export data<br>
                        • <b>Researcher:</b> Research analysis, limited result access<br>
                        • <b>Audit Trail:</b> All actions logged with user attribution
                        </small>
                    </div>
                </div>
            """)
        
        # Rest of your existing UI components go here...
        # Control Panel
        with gr.Group(elem_classes="control-panel"):
            gr.HTML("<h3 style='margin-top: 0; color: #333; text-align: center;'>⚙️ Medical Analysis Settings</h3>")
            
            with gr.Row():
                with gr.Column(scale=1):
                    sam_model_choices = gr.Dropdown(
                        choices=list(SAM_MODELS.keys()), 
                        label="🤖 SAM Model", 
                        value="brain_tumour_sam2",
                        info="Choose the SAM model for medical image segmentation"
                    )
                with gr.Column(scale=1):
                    box_threshold = gr.Slider(
                        minimum=0.0, 
                        maximum=1.0, 
                        value=0.25, 
                        label="🎯 Detection Threshold",
                        info="Sensitivity for feature detection (lower = more sensitive)"
                    )
                with gr.Column(scale=1):
                    text_threshold = gr.Slider(
                        minimum=0.0, 
                        maximum=1.0, 
                        value=0.2, 
                        label="🔍 Text Matching Threshold",
                        info="Confidence threshold for medical term matching"
                    )
        
        # Main Content Area
        gr.HTML("<h3 style='text-align: center; margin: 30px 0 20px 0; color: #333;'>🔬 Medical Image Analysis Workspace</h3>")
        
        with gr.Row():
            # Left side - Image Processing
            with gr.Column(scale=3):
                # Image Upload Section
                with gr.Group(elem_classes="control-panel"):
                    gr.HTML("<h4 style='margin-top: 0; color: #333; text-align: center;'>📋 Step 1: Upload Medical Image & Annotate ROI</h4>")
                    
                    image_input = gr.Image(
                        type="filepath", 
                        label="Upload Medical Image (JPEG, PNG) - Click on image to annotate ROI",
                        elem_classes="image-panel",
                        interactive=True
                    )
                    
                    # Bounding Box Creation Tools
                    with gr.Row(visible=False) as bbox_tools:
                        bbox_mode_btn = gr.Button(
                            "📦 Enable Bounding Box Mode",
                            variant="secondary",
                            scale=1,
                            visible=False,
                            interactive=False
                        )
                        clear_bbox_btn = gr.Button(
                            "🗑️ Clear Bounding Box",
                            variant="secondary",
                            visible=False,
                            scale=1
                        )
                    
                    # Bounding Box Display
                    bbox_display = gr.Textbox(
                        label="📦 Created Bounding Box",
                        value="No bounding box created yet",
                        interactive=False,
                        visible=False,
                        placeholder="Bounding box coordinates will appear here (x1,y1,x2,y2)"
                    )
                    
                    # Instructions for bounding box creation
                    bbox_instructions = gr.HTML(
                        "<div style='padding: 10px; background-color: #f0f8ff; border-radius: 5px; margin: 5px 0;'>"
                        "<strong>📦 Bounding Box Mode:</strong><br>"
                        "• Click to set first corner (top-left)<br>"
                        "• Click again to set second corner (bottom-right)<br>"
                        "• Box coordinates will auto-fill in the guidance field below"
                        "</div>",
                        visible=False
                    )
                
                # Results Section
                with gr.Group(elem_classes="control-panel"):
                    gr.HTML("<h4 style='margin-top: 0; color: #333; text-align: center;'>📊 Step 2: Segmentation Results</h4>")
                    output_image = gr.Image(
                        type="pil", 
                        label="🎯 Medical Segmentation Result",
                        elem_classes="image-panel"
                    )
                
                # Active Learning Section (Researchers & Radiologists)
                with gr.Accordion("🧠 Active Learning & Performance Analytics", open=False) as al_accordion:
                    gr.HTML("<div style='text-align: center; margin: 10px 0; color: #555;'>AI-Powered Analysis Feedback & Model Performance Monitoring</div>")
                    
                    # Feedback Section (For Radiologists and Researchers)
                    with gr.Accordion("📝 Quality Feedback", open=False) as feedback_accordion:
                        with gr.Row():
                            with gr.Column():
                                feedback_quality = gr.Radio(
                                    choices=["Excellent (5)", "Good (4)", "Fair (3)", "Poor (2)", "Very Poor (1)"],
                                    label="⭐ Overall Quality Rating",
                                    info="Rate the segmentation accuracy and clinical usefulness"
                                )
                                feedback_type = gr.Radio(
                                    choices=["General", "False Positive", "False Negative", "Boundary Issue", "Clinical Relevance"],
                                    label="🏷️ Feedback Type",
                                    value="General"
                                )
                            with gr.Column():
                                clinical_notes = gr.Textbox(
                                    lines=4,
                                    label="📋 Clinical Notes",
                                    placeholder="Provide specific feedback about segmentation quality, missed regions, or clinical accuracy..."
                                )
                                submit_feedback = gr.Button("💾 Submit Feedback", variant="primary")
                        
                        feedback_status = gr.HTML()
                    
                    # Performance Analytics Section (For Radiologists and Researchers)
                    with gr.Accordion("📊 Performance Analytics", open=False) as analytics_accordion:
                        with gr.Row():
                            days_filter = gr.Dropdown(
                                choices=[7, 14, 30, 90],
                                value=30,
                                label="📅 Time Period (Days)",
                                info="Filter analytics data"
                            )
                            refresh_analytics = gr.Button("🔄 Refresh Analytics", variant="secondary")
                        
                        analytics_overview = gr.HTML()
                        analytics_comparison = gr.HTML()
                        analytics_recommendations = gr.HTML()
                    
                    # MONAI Active Learning Section (For Researchers Only)
                    with gr.Accordion("🔬 MONAI Active Learning", open=False) as monai_accordion:
                        gr.HTML("""
                            <div style='padding: 15px; background: #f0f8ff; border-radius: 8px; margin: 10px 0; border-left: 4px solid #4a90e2;'>
                                <h4 style='color: #2c3e50; margin-top: 0;'>🧠 Advanced MONAI Active Learning</h4>
                                <p style='margin: 5px 0; color: #555;'>
                                    <strong>Researcher-only features:</strong> Uncertainty analysis, model training suggestions, 
                                    and expert review queue management powered by MONAI framework.
                                </p>
                            </div>
                        """)
                        
                        with gr.Row():
                            with gr.Column():
                                expert_queue = gr.HTML()
                                
                                with gr.Row():
                                    review_image_btn = gr.Button("📋 Review Next Image", variant="secondary")
                                    run_al_iteration = gr.Button("🔄 Run AL Iteration", variant="primary")
                            
                            with gr.Column():
                                al_strategy = gr.Dropdown(
                                    choices=["uncertainty", "diversity", "hybrid"],
                                    value="hybrid",
                                    label="🎯 Active Learning Strategy",
                                    info="Choose sampling strategy for next iteration"
                                )
                        
                        monai_dashboard = gr.HTML()
                        al_iteration_results = gr.HTML()
                        
                        # Initialize expert queue on load
                        def load_expert_queue(session_id, username, role):
                            if role == "researcher":
                                return get_monai_active_learning_dashboard({"username": username, "role": role})
                            return ""
                        
                        # Load expert queue when accordion opens
                        session_id.change(
                            fn=load_expert_queue,
                            inputs=[session_id, current_user, current_role],
                            outputs=[expert_queue]
                        )
            
            # Right side - Medical Prompt Assistant
            with gr.Column(scale=1, elem_classes="control-panel"):
                gr.HTML("<h3 style='margin-top: 0; color: #333; text-align: center;'>🩺 Medical Analysis Assistant</h3>")
                
                # Analysis Mode Selection
                analysis_mode = gr.Radio(
                    choices=["Text Prompt", "Point Annotations", "Bounding Box", "Combined (Text + Points)", "Combined (Text + Box)"],
                    value="Text Prompt",
                    label="🎯 Analysis Mode",
                    info="Choose how to guide the segmentation"
                )
                
                text_prompt = gr.Textbox(
                    lines=6, 
                    label="What medical features to detect?", 
                    placeholder="Describe the anatomical structures or abnormalities to segment...\n\nMedical Examples:\n• brain tumor\n• lung nodule\n• heart ventricle\n• liver lesion\n• bone fracture\n• kidney stone\n• blood vessel\n• abnormal tissue\n\nSeparate multiple targets with periods.",
                    info="🏥 Use medical terminology for better accuracy",
                    value="tumor"
                )
                
                # Bounding box coordinates input
                bbox_coords = gr.Textbox(
                    label="📦 Bounding Box Guidance Coordinates",
                    placeholder="Format: x1,y1,x2,y2 or (x1,y1,x2,y2) - Example: 100,150,200,250",
                    info="🎯 Guide GDINO detections: Filters object detections to focus only within this region. Still needs text prompt above!",
                    visible=False,
                    interactive=True
                )
                
                # Dynamic instructions based on analysis mode
                mode_instructions = gr.HTML(
                    get_analysis_instructions("Text Prompt"),
                    label="Instructions"
                )
                
                # Progress indicator and status
                with gr.Row():
                    analyze_btn = gr.Button(
                        "🔍 Analyze Medical Image",
                        variant="primary",
                        elem_classes=["main-button"],
                        size="lg"
                    )
                    analysis_status = gr.Textbox(
                        label="Analysis Status",
                        value="Ready to analyze",
                        interactive=False,
                        scale=1
                    )
    
    # State variables for point-based annotations
    clicked_points = gr.State([])
    bbox_points = gr.State([])  # State for bounding box creation
    feedback_submitted = gr.State(False)  # Track if feedback has been submitted for current image
    
    # Event handlers
    def update_interactive_image(image):
        """Update the annotation controls when a new image is uploaded."""
        if image is not None:
            # Reset feedback form when new image is uploaded
            return (
                image,
                gr.Row(visible=True),  # Make bbox_tools (entire row) visible
                None,  # Reset feedback_quality
                None,  # Reset feedback_type  
                "",    # Reset clinical_notes
                "",    # Reset feedback_status
                False  # Reset feedback_submitted state
            )
        else:
            return (
                None,
                gr.Row(visible=False),  # Keep bbox_tools hidden
                None,  # Reset feedback_quality
                None,  # Reset feedback_type
                "",    # Reset clinical_notes
                "",    # Reset feedback_status
                False  # Reset feedback_submitted state
            )
    
    def clear_annotations_handler(image_path):
        """Clear all annotation points."""
        result = clear_points(image_path)
        return result[0], result[2], []
    
    def handle_analysis_mode_change(mode):
        instructions = get_analysis_instructions(mode)
        # Only show bbox button and textbox for relevant modes
        if mode == "Bounding Box":
            # Only bbox guidance
            return (
                gr.Textbox(visible=False, interactive=False),
                gr.Textbox(visible=True, interactive=True),
                instructions,
                gr.Button(visible=True, interactive=True),  # bbox_mode_btn
            )
        elif mode == "Combined (Text + Box)":
            # Both text and bbox guidance
            return (
                gr.Textbox(visible=True, interactive=True),
                gr.Textbox(visible=True, interactive=True),
                instructions,
                gr.Button(visible=True, interactive=True),  # bbox_mode_btn
            )
        elif mode == "Text Prompt":
            return (
                gr.Textbox(visible=True, interactive=True),
                gr.Textbox(visible=False, interactive=False),
                instructions,
                gr.Button(visible=False, interactive=False),  # bbox_mode_btn
            )
        elif mode == "Point Annotations":
            return (
                gr.Textbox(visible=False, interactive=False),
                gr.Textbox(visible=False, interactive=False),
                instructions,
                gr.Button(visible=False, interactive=False),  # bbox_mode_btn
            )
        else:  # Combined (Text + Points)
            return (
                gr.Textbox(visible=True, interactive=True),
                gr.Textbox(visible=False, interactive=False),
                instructions,
                gr.Button(visible=False, interactive=False),  # bbox_mode_btn
            )

    def update_user_greeting(full_name, role):
        """Update the user greeting display."""
        if full_name and role:
            role_emojis = {
                "administrator": "👨‍💼",
                "radiologist": "👩‍⚕️", 
                "researcher": "👨‍🔬"
            }
            emoji = role_emojis.get(role, "👤")
            return f"{emoji} Welcome, {full_name} ({role.title()})"
        return "Welcome!"

    # Wire up login functionality
    login_btn.click(
        fn=login_user,
        inputs=[username_input, password_input],
        outputs=[login_section, main_section, login_status, session_id, current_user, current_role, current_full_name, admin_section]
    ).then(
        fn=update_user_greeting,
        inputs=[current_full_name, current_role],
        outputs=[user_greeting]
    )
    
    # Wire up logout functionality
    logout_btn.click(
        fn=logout_user,
        inputs=[session_id],
        outputs=[login_section, main_section, login_status, session_id, current_user, current_role, current_full_name, admin_section]
    )
    
    # Wire up the main analysis button
    analyze_btn.click(
        fn=inference,
        inputs=[sam_model_choices, box_threshold, text_threshold, image_input, text_prompt, clicked_points, image_input, bbox_coords, analysis_mode, session_id],
        outputs=[output_image, analysis_status]
    )

    # Wire up the event handlers for point-based annotation
    image_input.change(
        fn=update_interactive_image,
        inputs=[image_input],
        outputs=[image_input, bbox_tools, feedback_quality, feedback_type, clinical_notes, feedback_status, feedback_submitted]
    )
    
    image_input.select(
        fn=unified_click_handler,
        inputs=[image_input, clicked_points, bbox_points, bbox_mode_btn],
        outputs=[image_input, clicked_points, bbox_points, bbox_display, bbox_coords]
    )
    
    # Wire up bounding box functionality
    bbox_mode_btn.click(
        fn=toggle_bbox_mode,
        inputs=[bbox_mode_btn, image_input],
        outputs=[bbox_mode_btn, clear_bbox_btn, bbox_instructions, bbox_display]
    )
    
    clear_bbox_btn.click(
        fn=clear_bbox,
        inputs=[image_input],
        outputs=[image_input, bbox_points, bbox_display, bbox_coords]
    )
    
    analysis_mode.change(
        fn=handle_analysis_mode_change,
        inputs=[analysis_mode],
        outputs=[text_prompt, bbox_coords, mode_instructions, bbox_mode_btn]
    )
    
    # Wire up admin functionality
    create_user_btn.click(
        fn=create_new_user,
        inputs=[new_username, new_password, new_role, new_full_name, new_email, session_id],
        outputs=[create_user_status, user_list]
    )
    
    refresh_users_btn.click(
        fn=lambda session_id: ("User list refreshed.", get_user_list(session_id)),
        inputs=[session_id],
        outputs=[user_management_status, user_list]
    )
    
    toggle_user_btn.click(
        fn=toggle_user_status,
        inputs=[selected_user, session_id],
        outputs=[user_management_status, user_list]
    )
    
    refresh_stats_btn.click(
        fn=lambda session_id: (get_role_statistics(session_id), get_permission_matrix(session_id)),
        inputs=[session_id],
        outputs=[role_stats, permission_matrix]
    )

    # Active Learning Event Handlers
    if ACTIVE_LEARNING_ENABLED:
        # Show/hide AL tab and sections based on user role
        def update_al_visibility(session_id, role):
            """Update Active Learning tab visibility and sections based on role."""
            if not session_id:
                # Not logged in - hide everything
                if MONAI_AL_ENABLED:
                    return gr.Tab(visible=False), gr.Accordion(visible=False), gr.Accordion(visible=False), gr.Accordion(visible=False)
                else:
                    return gr.Tab(visible=False), gr.Accordion(visible=False), gr.Accordion(visible=False)
            
            # Radiologists and Researchers can see basic AL features
            if role in ["radiologist", "researcher"]:
                tab_visible = True
                feedback_visible = True
                analytics_visible = True
                # Only researchers can see MONAI features
                monai_visible = (role == "researcher") and MONAI_AL_ENABLED
            else:
                # Admin and guests cannot see AL features
                tab_visible = False
                feedback_visible = False
                analytics_visible = False
                monai_visible = False
            
            if MONAI_AL_ENABLED:
                return (
                    gr.Tab(visible=tab_visible),
                    gr.Accordion(visible=feedback_visible),
                    gr.Accordion(visible=analytics_visible),
                    gr.Accordion(visible=monai_visible)
                )
            else:
                return (
                    gr.Tab(visible=tab_visible),
                    gr.Accordion(visible=feedback_visible),
                    gr.Accordion(visible=analytics_visible)
                )
        
        # Update AL visibility when user logs in
        def reset_interface_on_login(session_id, role):
            """Reset all interface components when user logs in/changes."""
            # Reset images
            clean_input_image = None
            clean_output_image = None
            
            # Reset feedback form
            clean_feedback_quality = None
            clean_feedback_type = None
            clean_clinical_notes = ""
            clean_feedback_status = ""
            clean_feedback_submitted = False
            
            # Reset analysis status
            clean_analysis_status = "Ready to analyze"
            
            # Reset text inputs
            clean_text_prompt = "tumor"
            clean_bbox_coords = ""
            
            # Reset clicked points
            clean_clicked_points = []
            
            return (
                clean_input_image, clean_output_image, 
                clean_feedback_quality, clean_feedback_type, clean_clinical_notes, clean_feedback_status, clean_feedback_submitted,
                clean_analysis_status, clean_text_prompt, clean_bbox_coords, clean_clicked_points
            )
        
        def update_al_visibility(session_id, role):
            """Update Active Learning accordion visibility based on role."""
            if not ACTIVE_LEARNING_ENABLED:
                return (
                    gr.Accordion(visible=False),  # al_accordion
                    gr.Accordion(visible=False),  # feedback_accordion  
                    gr.Accordion(visible=False),  # analytics_accordion
                    gr.Accordion(visible=False)   # monai_accordion
                )
            
            # Show AL accordion for researchers and radiologists
            if role in ["researcher", "radiologist"]:
                feedback_visible = True
                analytics_visible = True
                monai_visible = (role == "researcher" and MONAI_AL_ENABLED)  # Only researchers see MONAI
                
                return (
                    gr.Accordion(visible=True),                    # al_accordion
                    gr.Accordion(visible=feedback_visible),        # feedback_accordion  
                    gr.Accordion(visible=analytics_visible),       # analytics_accordion
                    gr.Accordion(visible=monai_visible)            # monai_accordion
                )
            else:
                return (
                    gr.Accordion(visible=False),  # al_accordion
                    gr.Accordion(visible=False),  # feedback_accordion  
                    gr.Accordion(visible=False),  # analytics_accordion
                    gr.Accordion(visible=False)   # monai_accordion
                )

        # Reset interface on user login/change
        session_id.change(
            fn=reset_interface_on_login,
            inputs=[session_id, current_role],
            outputs=[
                image_input, output_image,
                feedback_quality, feedback_type, clinical_notes, feedback_status, feedback_submitted,
                analysis_status, text_prompt, bbox_coords, clicked_points
            ]
        )

        session_id.change(
            fn=update_al_visibility,
            inputs=[session_id, current_role],
            outputs=[al_accordion, feedback_accordion, analytics_accordion, monai_accordion]
        )
        
        # Submit feedback event handler
        submit_feedback.click(
            fn=lambda quality, ftype, notes, img_path, model, mode, prompt, bbox, session_id, username, role, feedback_state: collect_feedback(
                quality, ftype, notes, img_path, model, mode, prompt, bbox,
                {"username": username, "role": role}, {}, 0.0, 0.0, feedback_state
            ),
            inputs=[
                feedback_quality, feedback_type, clinical_notes, 
                image_input, sam_model_choices, analysis_mode, 
                text_prompt, bbox_coords, session_id, current_user, current_role, feedback_submitted
            ],
            outputs=[feedback_status, feedback_accordion, feedback_submitted]
        )
        
        # Refresh analytics dashboard
        refresh_analytics.click(
            fn=lambda session_id, username, role, days: get_performance_dashboard(
                {"username": username, "role": role}, days
            ),
            inputs=[session_id, current_user, current_role, days_filter],
            outputs=[analytics_overview, analytics_comparison, analytics_recommendations]
        )
        
        # Update analytics when days filter changes
        days_filter.change(
            fn=lambda session_id, username, role, days: get_performance_dashboard(
                {"username": username, "role": role}, days
            ),
            inputs=[session_id, current_user, current_role, days_filter],
            outputs=[analytics_overview, analytics_comparison, analytics_recommendations]
        )
        
        # MONAI Active Learning event handlers (Researchers only)
        if MONAI_AL_ENABLED:
            # Review next image
            review_image_btn.click(
                fn=lambda session_id, username, role: get_monai_active_learning_dashboard(
                    {"username": username, "role": role}
                ),
                inputs=[session_id, current_user, current_role],
                outputs=[expert_queue]
            )
            
            # Run active learning iteration
            run_al_iteration.click(
                fn=lambda session_id, username, role, strategy: run_active_learning_iteration(
                    {"username": username, "role": role}, strategy
                ),
                inputs=[session_id, current_user, current_role, al_strategy],
                outputs=[al_iteration_results]
            )

server.app = gr.mount_gradio_app(server.app, blocks, path="/aisegmentation")

if __name__ == "__main__":
    print(f"🏥 Starting AI Segmentation Platform on port {PORT}...")
    print("🔐 Secure Medical Image Analysis & AI-Powered Segmentation")
    print("� Please contact your system administrator for login credentials")
    print("🌐 Access URL: http://localhost:{PORT}/aisegmentation")
    server.run(port=PORT)
