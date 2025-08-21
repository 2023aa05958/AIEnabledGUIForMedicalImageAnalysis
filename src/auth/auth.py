"""
Authentication module for AI Medical Image Annotation Tool
Provides user authentication, session management, and access control
"""

import json
import hashlib
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import secrets


class UserManager:
    """Manages user authentication and session handling."""
    
    def __init__(self, users_file: str = "users.json"):
        self.users_file = users_file
        self.sessions = {}  # In-memory session storage
        self.session_timeout = timedelta(hours=8)  # 8-hour session timeout
        self.load_users()
    
    def load_users(self):
        """Load users from JSON file or create default admin user."""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r') as f:
                    self.users = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self.users = {}
                self.create_default_users()
        else:
            self.users = {}
            self.create_default_users()
    
    def save_users(self):
        """Save users to JSON file."""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def create_default_users(self):
        """Create default users for the medical annotation system."""
        default_users = [
            {
                "username": "admin",
                "password": "admin123",
                "role": "administrator",
                "full_name": "System Administrator",
                "email": "admin@medical-ai.com"
            },
            {
                "username": "doctor",
                "password": "doctor123", 
                "role": "radiologist",
                "full_name": "Dr. Medical Professional",
                "email": "doctor@medical-ai.com"
            },
            {
                "username": "researcher",
                "password": "research123",
                "role": "researcher", 
                "full_name": "AI Researcher",
                "email": "researcher@medical-ai.com"
            }
        ]
        
        for user_data in default_users:
            self.create_user(
                username=user_data["username"],
                password=user_data["password"],
                role=user_data["role"],
                full_name=user_data["full_name"],
                email=user_data["email"]
            )
        
        self.save_users()
        print("🔐 Default users created:")
        print("   👨‍💼 admin/admin123 (Administrator)")
        print("   👩‍⚕️ doctor/doctor123 (Radiologist)")
        print("   👨‍🔬 researcher/research123 (Researcher)")
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username: str, password: str, role: str, 
                   full_name: str = "", email: str = "") -> bool:
        """Create a new user."""
        if username in self.users:
            return False
        
        self.users[username] = {
            "password_hash": self.hash_password(password),
            "role": role,
            "full_name": full_name,
            "email": email,
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "active": True
        }
        return True
    
    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user and return user info if successful."""
        if username not in self.users:
            return None
        
        user = self.users[username]
        if not user.get("active", True):
            return None
        
        if user["password_hash"] == self.hash_password(password):
            # Update last login
            user["last_login"] = datetime.now().isoformat()
            self.save_users()
            
            return {
                "username": username,
                "role": user["role"],
                "full_name": user["full_name"],
                "email": user["email"]
            }
        return None
    
    def create_session(self, user_info: Dict[str, Any]) -> str:
        """Create a new session for authenticated user."""
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {
            "user_info": user_info,
            "created_at": datetime.now(),
            "last_activity": datetime.now()
        }
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Validate session and return user info if valid."""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Check if session has expired
        if datetime.now() - session["last_activity"] > self.session_timeout:
            del self.sessions[session_id]
            return None
        
        # Update last activity
        session["last_activity"] = datetime.now()
        return session["user_info"]
    
    def logout(self, session_id: str) -> bool:
        """Logout user by removing session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def get_role_permissions(self, role: str) -> Dict[str, bool]:
        """Get permissions based on user role."""
        permissions = {
            "administrator": {
                "can_analyze": True,
                "can_view_all_results": True,
                "can_manage_users": True,
                "can_export_data": True,
                "can_modify_settings": True,
                "can_view_system_logs": True
            },
            "radiologist": {
                "can_analyze": True,
                "can_view_all_results": True,
                "can_manage_users": False,
                "can_export_data": True,
                "can_modify_settings": False,
                "can_view_system_logs": False
            },
            "researcher": {
                "can_analyze": True,
                "can_view_all_results": False,
                "can_manage_users": False,
                "can_export_data": True,
                "can_modify_settings": False,
                "can_view_system_logs": False
            },
            "guest": {
                "can_analyze": True,
                "can_view_all_results": False,
                "can_manage_users": False,
                "can_export_data": False,
                "can_modify_settings": False,
                "can_view_system_logs": False
            }
        }
        return permissions.get(role, permissions["guest"])


# Global user manager instance
users_json_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "users.json")
user_manager = UserManager(users_json_path)


def require_auth(func):
    """Decorator to require authentication for Gradio functions."""
    def wrapper(*args, **kwargs):
        # Extract session from kwargs or args
        session_id = kwargs.get('session_id') or (args[-1] if args else None)
        
        if not session_id:
            return None, "❌ Authentication required"
        
        user_info = user_manager.validate_session(session_id)
        if not user_info:
            return None, "❌ Session expired. Please login again."
        
        # Add user_info to kwargs
        kwargs['user_info'] = user_info
        return func(*args, **kwargs)
    
    return wrapper


def get_user_greeting(user_info: Dict[str, Any]) -> str:
    """Generate personalized greeting for user."""
    role_emojis = {
        "administrator": "👨‍💼",
        "radiologist": "👩‍⚕️", 
        "researcher": "👨‍🔬",
        "guest": "👤"
    }
    
    emoji = role_emojis.get(user_info["role"], "👤")
    role_title = user_info["role"].title()
    name = user_info.get("full_name", user_info["username"])
    
    return f"{emoji} Welcome, {name} ({role_title})"


def check_permission(user_info: Dict[str, Any], permission: str) -> bool:
    """Check if user has specific permission."""
    permissions = user_manager.get_role_permissions(user_info["role"])
    return permissions.get(permission, False)
