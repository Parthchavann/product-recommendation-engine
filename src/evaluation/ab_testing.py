import hashlib
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
from scipy import stats
import json

from ..utils.logger import get_logger


logger = get_logger(__name__)


class ExperimentVariant(Enum):
    """Experiment variant types"""
    CONTROL = "control"
    TREATMENT = "treatment"


class ExperimentStatus(Enum):
    """Experiment status"""
    DRAFT = "draft"
    RUNNING = "running" 
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class ABTestManager:
    """Manage A/B tests for recommendation system"""
    
    def __init__(self, min_sample_size: int = 1000, significance_level: float = 0.05):
        self.experiments = {}
        self.results = {}
        self.min_sample_size = min_sample_size
        self.significance_level = significance_level
        
        logger.info(f"Initialized A/B Test Manager (min_samples={min_sample_size}, alpha={significance_level})")
    
    def create_experiment(self,
                         experiment_id: str,
                         name: str,
                         description: str,
                         treatment_percentage: float = 0.5,
                         min_sample_size: Optional[int] = None,
                         target_metrics: List[str] = None,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Create a new A/B test experiment
        
        Args:
            experiment_id: Unique identifier for the experiment
            name: Human-readable name
            description: Description of what's being tested
            treatment_percentage: Percentage of users in treatment group
            min_sample_size: Minimum sample size per group
            target_metrics: Primary metrics to track
            start_date: When experiment should start
            end_date: When experiment should end
            
        Returns:
            Experiment configuration
        """
        
        if experiment_id in self.experiments:
            raise ValueError(f"Experiment {experiment_id} already exists")
        
        if not 0 < treatment_percentage < 1:
            raise ValueError("Treatment percentage must be between 0 and 1")
        
        experiment = {
            'id': experiment_id,
            'name': name,
            'description': description,
            'treatment_percentage': treatment_percentage,
            'min_sample_size': min_sample_size or self.min_sample_size,
            'target_metrics': target_metrics or ['ctr', 'conversion_rate'],
            'start_date': start_date or datetime.now(),
            'end_date': end_date,
            'status': ExperimentStatus.DRAFT.value,
            'created_at': datetime.now(),
            'metrics': {
                'control': {
                    'users': set(),
                    'impressions': 0,
                    'clicks': 0,
                    'conversions': 0,
                    'revenue': 0.0,
                    'engagement_time': [],
                    'bounce_rate': [],
                    'custom_metrics': {}
                },
                'treatment': {
                    'users': set(),
                    'impressions': 0,
                    'clicks': 0,
                    'conversions': 0,
                    'revenue': 0.0,
                    'engagement_time': [],
                    'bounce_rate': [],
                    'custom_metrics': {}
                }
            }
        }
        
        self.experiments[experiment_id] = experiment
        
        logger.info(f"Created experiment: {experiment_id} ({name})")
        
        return experiment
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment"""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if experiment['status'] != ExperimentStatus.DRAFT.value:
            raise ValueError(f"Experiment {experiment_id} is not in draft status")
        
        experiment['status'] = ExperimentStatus.RUNNING.value
        experiment['actual_start_date'] = datetime.now()
        
        logger.info(f"Started experiment: {experiment_id}")
        
        return True
    
    def get_variant(self, experiment_id: str, user_id: int, 
                   force_assignment: Optional[bool] = None,
                   segment_overrides: Optional[Dict[str, str]] = None) -> ExperimentVariant:
        """
        Deterministically assign user to variant with dynamic allocation
        
        Args:
            experiment_id: Experiment identifier
            user_id: User identifier
            force_assignment: Force specific assignment (bypass normal logic)
            segment_overrides: Segment-specific allocation overrides
            
        Returns:
            Assigned variant
        """
        
        if experiment_id not in self.experiments:
            return ExperimentVariant.CONTROL
        
        experiment = self.experiments[experiment_id]
        
        # Check if experiment is running
        if experiment['status'] != ExperimentStatus.RUNNING.value:
            return ExperimentVariant.CONTROL
        
        # Check date bounds
        now = datetime.now()
        if now < experiment['start_date']:
            return ExperimentVariant.CONTROL
        
        if experiment['end_date'] and now > experiment['end_date']:
            return ExperimentVariant.CONTROL
        
        # Check for forced assignment
        if force_assignment is not None:
            return ExperimentVariant.TREATMENT if force_assignment else ExperimentVariant.CONTROL
        
        # Check for segment overrides
        if segment_overrides:
            user_segment = self._get_user_segment(user_id)
            if user_segment in segment_overrides:
                if segment_overrides[user_segment] == 'treatment':
                    return ExperimentVariant.TREATMENT
                else:
                    return ExperimentVariant.CONTROL
        
        # Dynamic allocation based on current sample sizes
        treatment_percentage = self._get_dynamic_allocation_percentage(experiment_id)
        
        # Hash user_id for consistent assignment
        hash_input = f"{experiment_id}:{user_id}".encode()
        hash_val = int(hashlib.md5(hash_input).hexdigest(), 16)
        
        # Determine variant based on hash and treatment percentage
        if (hash_val % 10000) / 10000 < treatment_percentage:
            return ExperimentVariant.TREATMENT
        return ExperimentVariant.CONTROL
    
    def record_impression(self,
                         experiment_id: str,
                         user_id: int,
                         variant: Optional[ExperimentVariant] = None) -> bool:
        """Record an impression (recommendation shown)"""
        
        if experiment_id not in self.experiments:
            return False
        
        if variant is None:
            variant = self.get_variant(experiment_id, user_id)
        
        experiment = self.experiments[experiment_id]
        metrics = experiment['metrics'][variant.value]
        
        metrics['users'].add(user_id)
        metrics['impressions'] += 1
        
        return True
    
    def record_click(self,
                    experiment_id: str,
                    user_id: int,
                    variant: Optional[ExperimentVariant] = None) -> bool:
        """Record a click (user clicked on recommendation)"""
        
        if experiment_id not in self.experiments:
            return False
        
        if variant is None:
            variant = self.get_variant(experiment_id, user_id)
        
        experiment = self.experiments[experiment_id]
        metrics = experiment['metrics'][variant.value]
        
        metrics['users'].add(user_id)
        metrics['clicks'] += 1
        
        return True
    
    def record_conversion(self,
                         experiment_id: str,
                         user_id: int,
                         revenue: float = 0.0,
                         variant: Optional[ExperimentVariant] = None) -> bool:
        """Record a conversion (user purchased/engaged deeply)"""
        
        if experiment_id not in self.experiments:
            return False
        
        if variant is None:
            variant = self.get_variant(experiment_id, user_id)
        
        experiment = self.experiments[experiment_id]
        metrics = experiment['metrics'][variant.value]
        
        metrics['users'].add(user_id)
        metrics['conversions'] += 1
        metrics['revenue'] += revenue
        
        return True
    
    def record_engagement(self,
                         experiment_id: str,
                         user_id: int,
                         engagement_time: float,
                         bounced: bool = False,
                         variant: Optional[ExperimentVariant] = None) -> bool:
        """Record engagement metrics"""
        
        if experiment_id not in self.experiments:
            return False
        
        if variant is None:
            variant = self.get_variant(experiment_id, user_id)
        
        experiment = self.experiments[experiment_id]
        metrics = experiment['metrics'][variant.value]
        
        metrics['users'].add(user_id)
        metrics['engagement_time'].append(engagement_time)
        metrics['bounce_rate'].append(1.0 if bounced else 0.0)
        
        return True
    
    def record_custom_metric(self,
                            experiment_id: str,
                            user_id: int,
                            metric_name: str,
                            metric_value: float,
                            variant: Optional[ExperimentVariant] = None) -> bool:
        """Record custom metric"""
        
        if experiment_id not in self.experiments:
            return False
        
        if variant is None:
            variant = self.get_variant(experiment_id, user_id)
        
        experiment = self.experiments[experiment_id]
        metrics = experiment['metrics'][variant.value]
        
        if metric_name not in metrics['custom_metrics']:
            metrics['custom_metrics'][metric_name] = []
        
        metrics['users'].add(user_id)
        metrics['custom_metrics'][metric_name].append(metric_value)
        
        return True
    
    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Analyze experiment results and determine statistical significance
        
        Args:
            experiment_id: Experiment to analyze
            
        Returns:
            Analysis results with statistics and recommendations
        """
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        control_metrics = experiment['metrics']['control']
        treatment_metrics = experiment['metrics']['treatment']
        
        # Check sample sizes
        control_users = len(control_metrics['users'])
        treatment_users = len(treatment_metrics['users'])
        min_sample_size = experiment['min_sample_size']
        
        if control_users < min_sample_size or treatment_users < min_sample_size:
            return {
                'status': 'insufficient_data',
                'control_users': control_users,
                'treatment_users': treatment_users,
                'min_required': min_sample_size,
                'message': f'Need at least {min_sample_size} users per group'
            }
        
        # Calculate basic metrics
        results = {
            'status': 'complete',
            'experiment_id': experiment_id,
            'name': experiment['name'],
            'analyzed_at': datetime.now().isoformat(),
            'sample_sizes': {
                'control': control_users,
                'treatment': treatment_users
            },
            'metrics': {},
            'statistical_tests': {},
            'recommendations': {}
        }
        
        # CTR Analysis
        if control_metrics['impressions'] > 0 and treatment_metrics['impressions'] > 0:
            control_ctr = control_metrics['clicks'] / control_metrics['impressions']
            treatment_ctr = treatment_metrics['clicks'] / treatment_metrics['impressions']
            
            # Chi-square test for CTR
            contingency_table = [
                [control_metrics['clicks'], control_metrics['impressions'] - control_metrics['clicks']],
                [treatment_metrics['clicks'], treatment_metrics['impressions'] - treatment_metrics['clicks']]
            ]
            
            chi2, p_value_ctr = stats.chi2_contingency(contingency_table)[:2]
            
            ctr_lift = (treatment_ctr - control_ctr) / control_ctr * 100 if control_ctr > 0 else 0
            
            results['metrics']['ctr'] = {
                'control': control_ctr,
                'treatment': treatment_ctr,
                'lift_percentage': ctr_lift,
                'absolute_difference': treatment_ctr - control_ctr
            }
            
            results['statistical_tests']['ctr'] = {
                'test_type': 'chi_square',
                'statistic': chi2,
                'p_value': p_value_ctr,
                'is_significant': p_value_ctr < self.significance_level,
                'confidence_level': (1 - self.significance_level) * 100
            }
        
        # Conversion Rate Analysis
        control_conv_rate = control_metrics['conversions'] / control_users if control_users > 0 else 0
        treatment_conv_rate = treatment_metrics['conversions'] / treatment_users if treatment_users > 0 else 0
        
        if control_users > 0 and treatment_users > 0:
            # Two-proportion z-test
            p1, p2 = control_conv_rate, treatment_conv_rate
            n1, n2 = control_users, treatment_users
            
            p_pooled = (control_metrics['conversions'] + treatment_metrics['conversions']) / (n1 + n2)
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
            
            if se > 0:
                z_score = (p2 - p1) / se
                p_value_conv = 2 * (1 - stats.norm.cdf(abs(z_score)))
            else:
                z_score = 0
                p_value_conv = 1.0
            
            conv_lift = (treatment_conv_rate - control_conv_rate) / control_conv_rate * 100 if control_conv_rate > 0 else 0
            
            results['metrics']['conversion_rate'] = {
                'control': control_conv_rate,
                'treatment': treatment_conv_rate,
                'lift_percentage': conv_lift,
                'absolute_difference': treatment_conv_rate - control_conv_rate
            }
            
            results['statistical_tests']['conversion_rate'] = {
                'test_type': 'two_proportion_z_test',
                'statistic': z_score,
                'p_value': p_value_conv,
                'is_significant': p_value_conv < self.significance_level,
                'confidence_level': (1 - self.significance_level) * 100
            }
        
        # Revenue Analysis
        control_revenue_per_user = control_metrics['revenue'] / control_users if control_users > 0 else 0
        treatment_revenue_per_user = treatment_metrics['revenue'] / treatment_users if treatment_users > 0 else 0
        
        revenue_lift = ((treatment_revenue_per_user - control_revenue_per_user) / 
                       control_revenue_per_user * 100) if control_revenue_per_user > 0 else 0
        
        results['metrics']['revenue_per_user'] = {
            'control': control_revenue_per_user,
            'treatment': treatment_revenue_per_user,
            'lift_percentage': revenue_lift,
            'absolute_difference': treatment_revenue_per_user - control_revenue_per_user
        }
        
        # Engagement Time Analysis
        if control_metrics['engagement_time'] and treatment_metrics['engagement_time']:
            control_engagement = np.array(control_metrics['engagement_time'])
            treatment_engagement = np.array(treatment_metrics['engagement_time'])
            
            # T-test for engagement time
            t_stat, p_value_engagement = stats.ttest_ind(treatment_engagement, control_engagement)
            
            control_avg_engagement = np.mean(control_engagement)
            treatment_avg_engagement = np.mean(treatment_engagement)
            engagement_lift = ((treatment_avg_engagement - control_avg_engagement) / 
                             control_avg_engagement * 100) if control_avg_engagement > 0 else 0
            
            results['metrics']['avg_engagement_time'] = {
                'control': control_avg_engagement,
                'treatment': treatment_avg_engagement,
                'lift_percentage': engagement_lift,
                'absolute_difference': treatment_avg_engagement - control_avg_engagement
            }
            
            results['statistical_tests']['engagement_time'] = {
                'test_type': 't_test',
                'statistic': t_stat,
                'p_value': p_value_engagement,
                'is_significant': p_value_engagement < self.significance_level,
                'confidence_level': (1 - self.significance_level) * 100
            }
        
        # Overall recommendation
        significant_improvements = []
        significant_degradations = []
        
        for metric_name, test_result in results['statistical_tests'].items():
            if test_result['is_significant']:
                metric_data = results['metrics'][metric_name]
                if metric_data['lift_percentage'] > 0:
                    significant_improvements.append(metric_name)
                else:
                    significant_degradations.append(metric_name)
        
        # Make recommendation
        if significant_improvements and not significant_degradations:
            recommendation = 'deploy_treatment'
            confidence = 'high'
            reason = f"Significant improvements in: {', '.join(significant_improvements)}"
        elif significant_improvements and significant_degradations:
            recommendation = 'investigate_further'
            confidence = 'medium'
            reason = f"Mixed results - improvements in {', '.join(significant_improvements)} but degradations in {', '.join(significant_degradations)}"
        elif significant_degradations:
            recommendation = 'keep_control'
            confidence = 'high'  
            reason = f"Significant degradations in: {', '.join(significant_degradations)}"
        else:
            recommendation = 'no_clear_winner'
            confidence = 'low'
            reason = "No statistically significant differences found"
        
        results['recommendations'] = {
            'action': recommendation,
            'confidence': confidence,
            'reason': reason,
            'significant_improvements': significant_improvements,
            'significant_degradations': significant_degradations
        }
        
        # Store results
        self.results[experiment_id] = results
        
        logger.info(f"Analyzed experiment {experiment_id}: {recommendation} ({confidence} confidence)")
        
        return results
    
    def stop_experiment(self, experiment_id: str, reason: str = "") -> bool:
        """Stop a running experiment"""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if experiment['status'] != ExperimentStatus.RUNNING.value:
            raise ValueError(f"Experiment {experiment_id} is not running")
        
        experiment['status'] = ExperimentStatus.COMPLETED.value
        experiment['ended_at'] = datetime.now()
        experiment['stop_reason'] = reason
        
        # Final analysis
        final_results = self.analyze_experiment(experiment_id)
        
        logger.info(f"Stopped experiment: {experiment_id} - {reason}")
        
        return True
    
    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment summary"""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        summary = {
            'id': experiment_id,
            'name': experiment['name'],
            'status': experiment['status'],
            'treatment_percentage': experiment['treatment_percentage'],
            'created_at': experiment['created_at'].isoformat(),
            'start_date': experiment['start_date'].isoformat(),
        }
        
        # Add runtime information
        if experiment['status'] == ExperimentStatus.RUNNING.value:
            summary['days_running'] = (datetime.now() - experiment['start_date']).days
        
        # Add sample sizes
        control_users = len(experiment['metrics']['control']['users'])
        treatment_users = len(experiment['metrics']['treatment']['users'])
        
        summary['sample_sizes'] = {
            'control': control_users,
            'treatment': treatment_users,
            'total': control_users + treatment_users
        }
        
        # Add latest results if available
        if experiment_id in self.results:
            latest_results = self.results[experiment_id]
            summary['latest_analysis'] = {
                'analyzed_at': latest_results['analyzed_at'],
                'recommendation': latest_results['recommendations']['action'],
                'confidence': latest_results['recommendations']['confidence']
            }
        
        return summary
    
    def list_experiments(self, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all experiments with optional status filter"""
        
        experiments = []
        
        for exp_id, exp_data in self.experiments.items():
            if status_filter is None or exp_data['status'] == status_filter:
                summary = self.get_experiment_summary(exp_id)
                experiments.append(summary)
        
        # Sort by creation date (newest first)
        experiments.sort(key=lambda x: x['created_at'], reverse=True)
        
        return experiments
    
    def export_results(self, experiment_id: str, format: str = 'json') -> str:
        """Export experiment results"""
        
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Get full experiment data
        experiment = self.experiments[experiment_id]
        
        # Convert sets to lists for serialization
        export_data = experiment.copy()
        for variant in ['control', 'treatment']:
            export_data['metrics'][variant]['users'] = list(export_data['metrics'][variant]['users'])
        
        # Add analysis results if available
        if experiment_id in self.results:
            export_data['analysis_results'] = self.results[experiment_id]
        
        # Convert datetime objects to strings
        for key, value in export_data.items():
            if isinstance(value, datetime):
                export_data[key] = value.isoformat()
        
        if format == 'json':
            return json.dumps(export_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")


class ExperimentalRecommendationEngine:
    """Recommendation engine with integrated A/B testing"""
    
    def __init__(self, base_engine, ab_manager: ABTestManager):
        self.base_engine = base_engine
        self.ab_manager = ab_manager
        
        logger.info("Initialized Experimental Recommendation Engine")
    
    async def get_recommendations(self,
                                user_id: int,
                                num_recommendations: int = 10,
                                **kwargs) -> Dict[str, Any]:
        """Get recommendations with A/B testing"""
        
        # Check active experiments
        active_experiments = []
        for exp_id, experiment in self.ab_manager.experiments.items():
            if experiment['status'] == ExperimentStatus.RUNNING.value:
                variant = self.ab_manager.get_variant(exp_id, user_id)
                active_experiments.append({
                    'experiment_id': exp_id,
                    'variant': variant.value
                })
        
        # Determine which model to use based on experiments
        model_version = 'control'  # Default
        experiment_context = {}
        
        for exp in active_experiments:
            if exp['variant'] == 'treatment':
                model_version = 'treatment'
                experiment_context[exp['experiment_id']] = exp['variant']
                
                # Record impression
                self.ab_manager.record_impression(exp['experiment_id'], user_id)
        
        # Get recommendations from appropriate model
        if model_version == 'treatment':
            # Use experimental model/parameters
            recommendations = await self._get_treatment_recommendations(user_id, num_recommendations, **kwargs)
        else:
            # Use control model
            recommendations = await self.base_engine.get_recommendations(user_id, num_recommendations, **kwargs)
        
        # Add experiment metadata
        response = {
            'user_id': user_id,
            'recommendations': recommendations,
            'experiment_context': experiment_context,
            'model_version': model_version,
            'active_experiments': active_experiments
        }
        
        return response
    
    async def _get_treatment_recommendations(self,
                                          user_id: int,
                                          num_recommendations: int,
                                          **kwargs) -> List[Dict[str, Any]]:
        """Get recommendations using treatment model"""
        
        # Example: Use different parameters for treatment
        kwargs_treatment = kwargs.copy()
        kwargs_treatment['use_reranking'] = True  # Always use re-ranking in treatment
        kwargs_treatment['num_recommendations'] = num_recommendations * 2  # Over-fetch more
        
        recommendations = await self.base_engine.get_recommendations(
            user_id, 
            num_recommendations, 
            **kwargs_treatment
        )
        
        return recommendations[:num_recommendations]
    
    def record_interaction(self,
                          user_id: int,
                          item_id: int,
                          action: str,
                          **kwargs):
        """Record user interaction for all active experiments"""
        
        for exp_id, experiment in self.ab_manager.experiments.items():
            if experiment['status'] == ExperimentStatus.RUNNING.value:
                variant = self.ab_manager.get_variant(exp_id, user_id)
                
                if action == 'click':
                    self.ab_manager.record_click(exp_id, user_id, variant)
                elif action == 'purchase':
                    revenue = kwargs.get('revenue', 0.0)
                    self.ab_manager.record_conversion(exp_id, user_id, revenue, variant)
                elif action == 'engagement':
                    engagement_time = kwargs.get('engagement_time', 0.0)
                    bounced = kwargs.get('bounced', False)
                    self.ab_manager.record_engagement(exp_id, user_id, engagement_time, bounced, variant)


def simulate_ab_test_data(ab_manager: ABTestManager,
                         experiment_id: str,
                         num_users: int = 10000,
                         days: int = 14) -> Dict[str, Any]:
    """
    Simulate A/B test data for demonstration
    
    Args:
        ab_manager: A/B test manager
        experiment_id: Experiment to simulate
        num_users: Number of users to simulate
        days: Duration in days
        
    Returns:
        Simulation summary
    """
    
    logger.info(f"Simulating A/B test data for {experiment_id}")
    
    # Simulation parameters
    base_ctr = 0.05  # 5% base CTR
    base_conversion = 0.02  # 2% base conversion rate
    treatment_lift_ctr = 0.15  # 15% improvement in CTR
    treatment_lift_conversion = 0.25  # 25% improvement in conversion
    
    simulated_data = {
        'experiment_id': experiment_id,
        'num_users': num_users,
        'days': days,
        'control_users': [],
        'treatment_users': []
    }
    
    for user_id in range(1, num_users + 1):
        variant = ab_manager.get_variant(experiment_id, user_id)
        
        # Simulate user journey over time
        for day in range(days):
            # Each user has a chance to interact each day
            if random.random() < 0.3:  # 30% chance of interaction per day
                
                # Record impression
                ab_manager.record_impression(experiment_id, user_id, variant)
                
                # Determine if user clicks
                if variant == ExperimentVariant.CONTROL:
                    click_prob = base_ctr
                    conv_prob = base_conversion
                else:  # Treatment
                    click_prob = base_ctr * (1 + treatment_lift_ctr)
                    conv_prob = base_conversion * (1 + treatment_lift_conversion)
                
                if random.random() < click_prob:
                    ab_manager.record_click(experiment_id, user_id, variant)
                    
                    # Record engagement
                    engagement_time = random.exponential(30)  # Average 30 seconds
                    bounced = random.random() < 0.4  # 40% bounce rate
                    ab_manager.record_engagement(experiment_id, user_id, engagement_time, bounced, variant)
                    
                    # Determine if user converts
                    if random.random() < conv_prob:
                        revenue = random.exponential(50)  # Average $50 revenue
                        ab_manager.record_conversion(experiment_id, user_id, revenue, variant)
        
        # Track which users were in which variant
        if variant == ExperimentVariant.CONTROL:
            simulated_data['control_users'].append(user_id)
        else:
            simulated_data['treatment_users'].append(user_id)
    
    logger.info(f"Simulation completed: {len(simulated_data['control_users'])} control, {len(simulated_data['treatment_users'])} treatment users")
    
    return simulated_data