"""
Active Learning Feedback Manager
Handles feedback collection, storage, and analysis for model improvement
"""

import json
import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid


class FeedbackManager:
    """Manages feedback collection and storage for active learning."""
    
    def __init__(self, db_path: str = "active_learning.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the feedback database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_feedback (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                user_role TEXT NOT NULL,
                image_path TEXT NOT NULL,
                model_used TEXT NOT NULL,
                analysis_mode TEXT NOT NULL,
                text_prompt TEXT,
                bounding_box TEXT,
                prediction_results TEXT,
                feedback_quality INTEGER,
                feedback_type TEXT,
                clinical_notes TEXT,
                confidence_score REAL,
                processing_time REAL,
                timestamp TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create model performance tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                date TEXT NOT NULL,
                total_predictions INTEGER DEFAULT 0,
                excellent_count INTEGER DEFAULT 0,
                good_count INTEGER DEFAULT 0,
                fair_count INTEGER DEFAULT 0,
                poor_count INTEGER DEFAULT 0,
                very_poor_count INTEGER DEFAULT 0,
                avg_quality_score REAL DEFAULT 0.0,
                avg_confidence REAL DEFAULT 0.0,
                avg_processing_time REAL DEFAULT 0.0,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create user feedback summary table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_feedback_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                date TEXT NOT NULL,
                total_feedback INTEGER DEFAULT 0,
                avg_quality_rating REAL DEFAULT 0.0,
                most_used_model TEXT,
                feedback_trend TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Active Learning database initialized successfully")
    
    def store_feedback(self, feedback_data: Dict[str, Any]) -> str:
        """Store user feedback in the database."""
        feedback_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO prediction_feedback (
                    id, user_id, user_role, image_path, model_used, analysis_mode,
                    text_prompt, bounding_box, prediction_results, feedback_quality,
                    feedback_type, clinical_notes, confidence_score, processing_time, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback_id,
                feedback_data.get('user_id', ''),
                feedback_data.get('user_role', ''),
                feedback_data.get('image_path', ''),
                feedback_data.get('model_used', ''),
                feedback_data.get('analysis_mode', ''),
                feedback_data.get('text_prompt', ''),
                feedback_data.get('bounding_box', ''),
                json.dumps(feedback_data.get('prediction_results', {})),
                feedback_data.get('feedback_quality', 0),
                feedback_data.get('feedback_type', ''),
                feedback_data.get('clinical_notes', ''),
                feedback_data.get('confidence_score', 0.0),
                feedback_data.get('processing_time', 0.0),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            print(f"✅ Feedback stored with ID: {feedback_id}")
            
            # Update performance metrics
            self._update_model_performance(feedback_data)
            
            return feedback_id
            
        except Exception as e:
            print(f"❌ Error storing feedback: {e}")
            return ""
        finally:
            conn.close()
    
    def _update_model_performance(self, feedback_data: Dict[str, Any]):
        """Update model performance metrics based on new feedback."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            model_name = feedback_data.get('model_used', '')
            today = datetime.now().strftime('%Y-%m-%d')
            quality = feedback_data.get('feedback_quality', 0)
            confidence = feedback_data.get('confidence_score', 0.0)
            processing_time = feedback_data.get('processing_time', 0.0)
            
            # Check if record exists for today
            cursor.execute('''
                SELECT id, total_predictions, excellent_count, good_count, fair_count, 
                       poor_count, very_poor_count, avg_quality_score, avg_confidence, avg_processing_time
                FROM model_performance 
                WHERE model_name = ? AND date = ?
            ''', (model_name, today))
            
            result = cursor.fetchone()
            
            if result:
                # Update existing record
                record_id, total, excellent, good, fair, poor, very_poor, avg_quality, avg_conf, avg_time = result
                
                # Update counters
                total += 1
                quality_mapping = {5: 'excellent', 4: 'good', 3: 'fair', 2: 'poor', 1: 'very_poor'}
                quality_name = quality_mapping.get(quality, 'fair')
                
                if quality_name == 'excellent':
                    excellent += 1
                elif quality_name == 'good':
                    good += 1
                elif quality_name == 'fair':
                    fair += 1
                elif quality_name == 'poor':
                    poor += 1
                else:
                    very_poor += 1
                
                # Calculate new averages
                new_avg_quality = (avg_quality * (total - 1) + quality) / total
                new_avg_conf = (avg_conf * (total - 1) + confidence) / total
                new_avg_time = (avg_time * (total - 1) + processing_time) / total
                
                cursor.execute('''
                    UPDATE model_performance 
                    SET total_predictions = ?, excellent_count = ?, good_count = ?, 
                        fair_count = ?, poor_count = ?, very_poor_count = ?,
                        avg_quality_score = ?, avg_confidence = ?, avg_processing_time = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (total, excellent, good, fair, poor, very_poor, 
                      new_avg_quality, new_avg_conf, new_avg_time, record_id))
            else:
                # Create new record
                excellent = 1 if quality == 5 else 0
                good = 1 if quality == 4 else 0
                fair = 1 if quality == 3 else 0
                poor = 1 if quality == 2 else 0
                very_poor = 1 if quality == 1 else 0
                
                cursor.execute('''
                    INSERT INTO model_performance (
                        model_name, date, total_predictions, excellent_count, good_count,
                        fair_count, poor_count, very_poor_count, avg_quality_score,
                        avg_confidence, avg_processing_time
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (model_name, today, 1, excellent, good, fair, poor, very_poor,
                      quality, confidence, processing_time))
            
            conn.commit()
            
        except Exception as e:
            print(f"❌ Error updating model performance: {e}")
        finally:
            conn.close()
    
    def get_model_performance_report(self, days: int = 30) -> Dict[str, Any]:
        """Get model performance report for the last N days."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get performance data for last N days
            cursor.execute('''
                SELECT model_name, date, total_predictions, excellent_count, good_count,
                       fair_count, poor_count, very_poor_count, avg_quality_score,
                       avg_confidence, avg_processing_time
                FROM model_performance 
                WHERE date >= date('now', '-{} days')
                ORDER BY date DESC, model_name
            '''.format(days))
            
            results = cursor.fetchall()
            
            # Process results into report format
            report = {
                'summary': {},
                'daily_performance': [],
                'model_comparison': {},
                'recommendations': []
            }
            
            model_totals = {}
            
            for row in results:
                model_name, date, total, excellent, good, fair, poor, very_poor, avg_quality, avg_conf, avg_time = row
                
                daily_data = {
                    'model_name': model_name,
                    'date': date,
                    'total_predictions': total,
                    'quality_distribution': {
                        'excellent': excellent,
                        'good': good,
                        'fair': fair,
                        'poor': poor,
                        'very_poor': very_poor
                    },
                    'avg_quality_score': round(avg_quality, 2),
                    'avg_confidence': round(avg_conf, 2),
                    'avg_processing_time': round(avg_time, 2)
                }
                
                report['daily_performance'].append(daily_data)
                
                # Aggregate by model
                if model_name not in model_totals:
                    model_totals[model_name] = {
                        'total_predictions': 0,
                        'total_quality_score': 0,
                        'total_confidence': 0,
                        'total_time': 0,
                        'excellent': 0,
                        'good': 0,
                        'fair': 0,
                        'poor': 0,
                        'very_poor': 0
                    }
                
                model_totals[model_name]['total_predictions'] += total
                model_totals[model_name]['total_quality_score'] += avg_quality * total
                model_totals[model_name]['total_confidence'] += avg_conf * total
                model_totals[model_name]['total_time'] += avg_time * total
                model_totals[model_name]['excellent'] += excellent
                model_totals[model_name]['good'] += good
                model_totals[model_name]['fair'] += fair
                model_totals[model_name]['poor'] += poor
                model_totals[model_name]['very_poor'] += very_poor
            
            # Calculate model comparison
            for model_name, totals in model_totals.items():
                if totals['total_predictions'] > 0:
                    report['model_comparison'][model_name] = {
                        'total_predictions': totals['total_predictions'],
                        'avg_quality': round(totals['total_quality_score'] / totals['total_predictions'], 2),
                        'avg_confidence': round(totals['total_confidence'] / totals['total_predictions'], 2),
                        'avg_processing_time': round(totals['total_time'] / totals['total_predictions'], 2),
                        'success_rate': round((totals['excellent'] + totals['good']) / totals['total_predictions'] * 100, 1)
                    }
            
            # Generate recommendations
            if report['model_comparison']:
                best_quality_model = max(report['model_comparison'].items(), 
                                       key=lambda x: x[1]['avg_quality'])
                fastest_model = min(report['model_comparison'].items(), 
                                  key=lambda x: x[1]['avg_processing_time'])
                
                report['recommendations'] = [
                    f"🏆 Best Quality: {best_quality_model[0]} (avg: {best_quality_model[1]['avg_quality']}/5)",
                    f"⚡ Fastest: {fastest_model[0]} ({fastest_model[1]['avg_processing_time']:.2f}s)",
                    f"📊 Total feedback collected: {sum(m['total_predictions'] for m in report['model_comparison'].values())}"
                ]
            
            return report
            
        except Exception as e:
            print(f"❌ Error generating performance report: {e}")
            return {}
        finally:
            conn.close()
    
    def get_feedback_history(self, user_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get feedback history, optionally filtered by user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            if user_id:
                cursor.execute('''
                    SELECT * FROM prediction_feedback 
                    WHERE user_id = ?
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (user_id, limit))
            else:
                cursor.execute('''
                    SELECT * FROM prediction_feedback 
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (limit,))
            
            columns = [description[0] for description in cursor.description]
            results = cursor.fetchall()
            
            feedback_list = []
            for row in results:
                feedback_dict = dict(zip(columns, row))
                if feedback_dict['prediction_results']:
                    try:
                        feedback_dict['prediction_results'] = json.loads(feedback_dict['prediction_results'])
                    except json.JSONDecodeError:
                        feedback_dict['prediction_results'] = {}
                feedback_list.append(feedback_dict)
            
            return feedback_list
            
        except Exception as e:
            print(f"❌ Error retrieving feedback history: {e}")
            return []
        finally:
            conn.close()


# Initialize the feedback manager
feedback_manager = FeedbackManager()
