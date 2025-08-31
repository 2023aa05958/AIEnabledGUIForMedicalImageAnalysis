"""
Active Learning Module for Medical Image Annotation System

This module provides comprehensive active learning capabilities for medical AI models,
combining traditional feedback collection with advanced MONAI-based active learning.

Components:
- feedback_manager: Database and feedback collection system
- performance_analytics: Performance tracking and analysis
- monai_active_learning: MONAI-based uncertainty sampling and active learning
- al_orchestrator: Integration orchestrator combining all AL approaches

Features:
- Uncertainty-based sample selection using Monte Carlo dropout
- Diversity-based sampling for comprehensive training data
- Expert review queue management
- Performance analytics and model comparison
- Real-time uncertainty assessment during inference
"""

from .feedback_manager import feedback_manager
from .performance_analytics import performance_analytics

try:
    from .monai_active_learning import MonaiActiveLearning, create_monai_active_learning_config
    from .al_orchestrator import active_learning_orchestrator
    MONAI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ MONAI Active Learning not available: {e}")
    MONAI_AVAILABLE = False

__version__ = "1.0.0"
__author__ = "Medical AI Research Team"

__all__ = [
    'feedback_manager',
    'performance_analytics',
    'MONAI_AVAILABLE'
]

if MONAI_AVAILABLE:
    __all__.extend([
        'MonaiActiveLearning',
        'create_monai_active_learning_config',
        'active_learning_orchestrator'
    ])

print(f"🤖 Active Learning Module v{__version__} loaded")
print(f"   📊 Basic Analytics: Available")
print(f"   🧠 MONAI Integration: {'Available' if MONAI_AVAILABLE else 'Not Available'}")
