"""
Performance Analytics for Active Learning
Provides detailed analysis and visualization data for model performance tracking
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import json


class PerformanceAnalytics:
    """Advanced analytics for model performance and feedback analysis."""
    
    def __init__(self, db_path: str = "active_learning.db"):
        self.db_path = db_path
    
    def get_comprehensive_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive analytics for the dashboard."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            analytics = {
                'overview': self._get_overview_stats(cursor, days),
                'model_trends': self._get_model_trends(cursor, days),
                'user_insights': self._get_user_insights(cursor, days),
                'quality_analysis': self._get_quality_analysis(cursor, days),
                'recommendations': self._generate_recommendations(cursor, days)
            }
            
            return analytics
            
        except Exception as e:
            print(f"❌ Error generating analytics: {e}")
            return {}
        finally:
            conn.close()
    
    def _get_overview_stats(self, cursor, days: int) -> Dict[str, Any]:
        """Get high-level overview statistics."""
        # Total feedback count
        cursor.execute('''
            SELECT COUNT(*) FROM prediction_feedback 
            WHERE date(timestamp) >= date('now', '-{} days')
        '''.format(days))
        total_feedback = cursor.fetchone()[0]
        
        # Average quality score
        cursor.execute('''
            SELECT AVG(feedback_quality) FROM prediction_feedback 
            WHERE date(timestamp) >= date('now', '-{} days')
            AND feedback_quality > 0
        '''.format(days))
        avg_quality = cursor.fetchone()[0] or 0
        
        # Unique users providing feedback
        cursor.execute('''
            SELECT COUNT(DISTINCT user_id) FROM prediction_feedback 
            WHERE date(timestamp) >= date('now', '-{} days')
        '''.format(days))
        active_users = cursor.fetchone()[0]
        
        # Models used
        cursor.execute('''
            SELECT COUNT(DISTINCT model_used) FROM prediction_feedback 
            WHERE date(timestamp) >= date('now', '-{} days')
        '''.format(days))
        models_used = cursor.fetchone()[0]
        
        # Calculate improvement trend (compare with previous period)
        cursor.execute('''
            SELECT AVG(feedback_quality) FROM prediction_feedback 
            WHERE date(timestamp) BETWEEN date('now', '-{} days') AND date('now', '-{} days')
            AND feedback_quality > 0
        '''.format(days * 2, days))
        prev_avg_quality = cursor.fetchone()[0] or 0
        
        trend = "improving" if avg_quality > prev_avg_quality else "declining" if avg_quality < prev_avg_quality else "stable"
        
        return {
            'total_feedback': total_feedback,
            'avg_quality_score': round(avg_quality, 2),
            'active_users': active_users,
            'models_evaluated': models_used,
            'quality_trend': trend,
            'trend_change': round(avg_quality - prev_avg_quality, 2)
        }
    
    def _get_model_trends(self, cursor, days: int) -> List[Dict[str, Any]]:
        """Get model performance trends over time."""
        cursor.execute('''
            SELECT model_used, date(timestamp) as day, 
                   COUNT(*) as predictions,
                   AVG(feedback_quality) as avg_quality,
                   AVG(CASE WHEN confidence_score > 0 THEN confidence_score ELSE NULL END) as avg_confidence,
                   AVG(CASE WHEN processing_time > 0 THEN processing_time ELSE NULL END) as avg_time
            FROM prediction_feedback 
            WHERE date(timestamp) >= date('now', '-{} days')
            AND feedback_quality > 0
            GROUP BY model_used, date(timestamp)
            ORDER BY day DESC, model_used
        '''.format(days))
        
        results = cursor.fetchall()
        trends = []
        
        for row in results:
            model, day, predictions, avg_quality, avg_confidence, avg_time = row
            trends.append({
                'model': model,
                'date': day,
                'predictions': predictions,
                'avg_quality': round(avg_quality, 2),
                'avg_confidence': round(avg_confidence or 0.0, 2),
                'avg_processing_time': round(avg_time or 0.0, 2)
            })
        
        return trends
    
    def _get_user_insights(self, cursor, days: int) -> List[Dict[str, Any]]:
        """Get user feedback insights."""
        cursor.execute('''
            SELECT user_id, user_role,
                   COUNT(*) as total_feedback,
                   AVG(feedback_quality) as avg_rating,
                   COUNT(DISTINCT model_used) as models_tried,
                   COUNT(DISTINCT date(timestamp)) as active_days
            FROM prediction_feedback 
            WHERE date(timestamp) >= date('now', '-{} days')
            AND feedback_quality > 0
            GROUP BY user_id, user_role
            ORDER BY total_feedback DESC
        '''.format(days))
        
        results = cursor.fetchall()
        insights = []
        
        for row in results:
            user_id, user_role, total_feedback, avg_rating, models_tried, active_days = row
            insights.append({
                'user_id': user_id,
                'user_role': user_role,
                'total_feedback': total_feedback,
                'avg_rating': round(avg_rating, 2),
                'models_tried': models_tried,
                'active_days': active_days,
                'engagement_score': round((total_feedback * avg_rating) / max(active_days, 1), 2)
            })
        
        return insights
    
    def _get_quality_analysis(self, cursor, days: int) -> Dict[str, Any]:
        """Analyze quality distribution and patterns."""
        # Quality distribution
        cursor.execute('''
            SELECT feedback_quality, COUNT(*) as count
            FROM prediction_feedback 
            WHERE date(timestamp) >= date('now', '-{} days')
            AND feedback_quality > 0
            GROUP BY feedback_quality
            ORDER BY feedback_quality DESC
        '''.format(days))
        
        quality_dist = {}
        quality_labels = {5: "Excellent", 4: "Good", 3: "Fair", 2: "Poor", 1: "Very Poor"}
        
        for quality, count in cursor.fetchall():
            quality_dist[quality_labels.get(quality, f"Rating {quality}")] = count
        
        # Analysis mode performance
        cursor.execute('''
            SELECT analysis_mode, 
                   COUNT(*) as total,
                   AVG(feedback_quality) as avg_quality,
                   AVG(confidence_score) as avg_confidence
            FROM prediction_feedback 
            WHERE date(timestamp) >= date('now', '-{} days')
            AND feedback_quality > 0
            GROUP BY analysis_mode
            ORDER BY avg_quality DESC
        '''.format(days))
        
        mode_performance = []
        for row in cursor.fetchall():
            mode, total, avg_quality, avg_confidence = row
            mode_performance.append({
                'mode': mode,
                'total_uses': total,
                'avg_quality': round(avg_quality, 2),
                'avg_confidence': round(avg_confidence, 2)
            })
        
        return {
            'quality_distribution': quality_dist,
            'mode_performance': mode_performance
        }
    
    def _generate_recommendations(self, cursor, days: int) -> List[str]:
        """Generate actionable recommendations based on data analysis."""
        recommendations = []
        
        # Find best performing model
        cursor.execute('''
            SELECT model_used, AVG(feedback_quality) as avg_quality, COUNT(*) as usage_count
            FROM prediction_feedback 
            WHERE date(timestamp) >= date('now', '-{} days')
            AND feedback_quality > 0
            GROUP BY model_used
            HAVING usage_count >= 5
            ORDER BY avg_quality DESC
            LIMIT 1
        '''.format(days))
        
        best_model = cursor.fetchone()
        if best_model:
            model_name, quality, usage = best_model
            recommendations.append(f"🏆 {model_name} shows best performance (avg: {quality:.2f}/5) - consider as default")
        
        # Find underperforming models
        cursor.execute('''
            SELECT model_used, AVG(feedback_quality) as avg_quality, COUNT(*) as usage_count
            FROM prediction_feedback 
            WHERE date(timestamp) >= date('now', '-{} days')
            AND feedback_quality > 0
            GROUP BY model_used
            HAVING usage_count >= 5 AND avg_quality < 3.0
            ORDER BY avg_quality ASC
        '''.format(days))
        
        poor_models = cursor.fetchall()
        for model_name, quality, usage in poor_models:
            recommendations.append(f"⚠️ {model_name} needs improvement (avg: {quality:.2f}/5) - consider retraining")
        
        # Check user engagement
        cursor.execute('''
            SELECT COUNT(DISTINCT user_id) as active_users,
                   COUNT(*) as total_feedback
            FROM prediction_feedback 
            WHERE date(timestamp) >= date('now', '-{} days')
        '''.format(days))
        
        engagement = cursor.fetchone()
        if engagement and engagement[0] > 0:
            feedback_per_user = engagement[1] / engagement[0]
            if feedback_per_user < 5:
                recommendations.append(f"📈 Low user engagement ({feedback_per_user:.1f} feedback/user) - consider feedback incentives")
        
        # Analysis mode recommendations
        cursor.execute('''
            SELECT analysis_mode, AVG(feedback_quality) as avg_quality, COUNT(*) as usage
            FROM prediction_feedback 
            WHERE date(timestamp) >= date('now', '-{} days')
            AND feedback_quality > 0
            GROUP BY analysis_mode
            ORDER BY avg_quality DESC
            LIMIT 1
        '''.format(days))
        
        best_mode = cursor.fetchone()
        if best_mode:
            mode, quality, usage = best_mode
            recommendations.append(f"🎯 '{mode}' mode performs best (avg: {quality:.2f}/5) - promote to users")
        
        if not recommendations:
            recommendations.append("📊 Collect more feedback data to generate meaningful recommendations")
        
        return recommendations
    
    def get_model_comparison_data(self, days: int = 30) -> Dict[str, Any]:
        """Get data specifically formatted for model comparison charts."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                SELECT model_used,
                       COUNT(*) as total_predictions,
                       AVG(feedback_quality) as avg_quality,
                       AVG(CASE WHEN confidence_score > 0 THEN confidence_score ELSE NULL END) as avg_confidence,
                       AVG(CASE WHEN processing_time > 0 THEN processing_time ELSE NULL END) as avg_time,
                       SUM(CASE WHEN feedback_quality >= 4 THEN 1 ELSE 0 END) as good_predictions
                FROM prediction_feedback 
                WHERE date(timestamp) >= date('now', '-{} days')
                AND feedback_quality > 0
                GROUP BY model_used
                ORDER BY avg_quality DESC
            '''.format(days))
            
            results = cursor.fetchall()
            
            comparison_data = {
                'models': [],
                'quality_scores': [],
                'confidence_scores': [],
                'processing_times': [],
                'success_rates': [],
                'total_uses': []
            }
            
            for row in results:
                model, total, avg_quality, avg_confidence, avg_time, good_predictions = row
                success_rate = (good_predictions / total * 100) if total > 0 else 0
                
                comparison_data['models'].append(model)
                comparison_data['quality_scores'].append(round(avg_quality, 2))
                comparison_data['confidence_scores'].append(round(avg_confidence or 0.0, 2))
                comparison_data['processing_times'].append(round(avg_time or 0.0, 2))
                comparison_data['success_rates'].append(round(success_rate, 1))
                comparison_data['total_uses'].append(total)
            
            return comparison_data
            
        except Exception as e:
            print(f"❌ Error generating comparison data: {e}")
            return {}
        finally:
            conn.close()


# Initialize analytics
performance_analytics = PerformanceAnalytics()
