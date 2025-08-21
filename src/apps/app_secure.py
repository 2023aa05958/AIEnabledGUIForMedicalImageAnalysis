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


@require_auth
def inference(sam_type, box_threshold, text_threshold, image, text_prompt, clicked_points, 
              annotated_image, bbox_coords, analysis_mode, session_id, user_info=None):
    """Gradio function that makes a request to the /predict LitServe endpoint."""
    
    # Check if user has permission to analyze
    if not check_permission(user_info, "can_analyze"):
        return None, "❌ You don't have permission to perform analysis"
    
    url = f"http://localhost:{PORT}/predict"

    # Use the original image file for processing
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
                response = requests.post(url, files=files, data=data)
            except Exception as e:
                print(f"Request failed: {e}")
                return None, f"❌ Server connection failed: {str(e)}"

        if response.status_code == 200:
            try:
                output_image = Image.open(BytesIO(response.content)).convert("RGB")
                
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
                
                return output_image, f"✅ Analysis complete! Saved to: {output_filename}"
                
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
                        value="sam2.1_hiera_small",
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
                    
                    with gr.Row():
                        clear_annotations_btn = gr.Button(
                            "🗑️ Clear All Points",
                            variant="secondary",
                            visible=False
                        )
                        annotation_status = gr.Textbox(
                            label="Annotation Status",
                            value="Upload an image to start annotating",
                            interactive=False,
                            visible=False,
                            scale=2
                        )
                
                # Results Section
                with gr.Group(elem_classes="control-panel"):
                    gr.HTML("<h4 style='margin-top: 0; color: #333; text-align: center;'>📊 Step 2: Segmentation Results</h4>")
                    output_image = gr.Image(
                        type="pil", 
                        label="🎯 Medical Segmentation Result",
                        elem_classes="image-panel"
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
    
    # Event handlers
    def update_interactive_image(image):
        """Update the annotation controls when a new image is uploaded."""
        if image is not None:
            return (
                image,
                gr.Button(visible=True),
                gr.Textbox(value="Ready for annotation - click directly on the image above", visible=True),
                []
            )
        else:
            return (
                None,
                gr.Button(visible=False),
                gr.Textbox(value="Upload an image to start annotating", visible=False),
                []
            )
    
    def clear_annotations_handler(image_path):
        """Clear all annotation points."""
        result = clear_points(image_path)
        return result[0], result[2], []
    
    def handle_analysis_mode_change(mode):
        """Handle changes in analysis mode."""
        instructions = get_analysis_instructions(mode)
        
        if mode == "Text Prompt":
            return gr.Textbox(visible=True, interactive=True), gr.Textbox(visible=False, interactive=False), instructions
        elif mode == "Point Annotations":
            return gr.Textbox(visible=False, interactive=False), gr.Textbox(visible=False, interactive=False), instructions
        elif mode == "Bounding Box":
            return gr.Textbox(visible=False, interactive=False), gr.Textbox(visible=True, interactive=True), instructions
        elif mode == "Combined (Text + Points)":
            return gr.Textbox(visible=True, interactive=True), gr.Textbox(visible=False, interactive=False), instructions
        else:  # Combined (Text + Box)
            return gr.Textbox(visible=True, interactive=True), gr.Textbox(visible=True, interactive=True), instructions

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
        outputs=[image_input, clear_annotations_btn, annotation_status, clicked_points]
    )
    
    image_input.select(
        fn=process_click,
        inputs=[image_input, clicked_points],
        outputs=[image_input, clicked_points, annotation_status]
    )
    
    clear_annotations_btn.click(
        fn=clear_annotations_handler,
        inputs=[image_input],
        outputs=[image_input, annotation_status, clicked_points]
    )
    
    analysis_mode.change(
        fn=handle_analysis_mode_change,
        inputs=[analysis_mode],
        outputs=[text_prompt, bbox_coords, mode_instructions]
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

server.app = gr.mount_gradio_app(server.app, blocks, path="/aisegmentation")

if __name__ == "__main__":
    print(f"🏥 Starting AI Segmentation Platform on port {PORT}...")
    print("🔐 Secure Medical Image Analysis & AI-Powered Segmentation")
    print("� Please contact your system administrator for login credentials")
    print("🌐 Access URL: http://localhost:{PORT}/aisegmentation")
    server.run(port=PORT)
