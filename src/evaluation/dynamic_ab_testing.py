"""
Dynamic A/B Testing Extensions
Extends the base ABTestManager with dynamic allocation capabilities
"""

from typing import Dict, Any, Optional
from datetime import datetime
from .ab_testing import ABTestManager, ExperimentVariant
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DynamicABTestManager(ABTestManager):
    """Enhanced A/B Test Manager with dynamic allocation capabilities"""
    
    def _get_user_segment(self, user_id: int) -> str:
        """Determine user segment for targeted allocation"""
        # Simple segmentation based on user_id
        # In production, this would use actual user data
        if user_id % 10 == 0:
            return "power_user"
        elif user_id % 5 == 0:
            return "regular_user"
        else:
            return "new_user"
    
    def _get_dynamic_allocation_percentage(self, experiment_id: str) -> float:
        """Calculate dynamic allocation percentage based on current sample sizes"""
        experiment = self.experiments[experiment_id]
        
        # Get current sample sizes
        control_users = len(experiment['metrics']['control']['users'])
        treatment_users = len(experiment['metrics']['treatment']['users'])
        
        # If very early in experiment, use original percentage
        if control_users + treatment_users < 100:
            return experiment['treatment_percentage']
        
        # Check if sample sizes are severely imbalanced
        total_users = control_users + treatment_users
        control_ratio = control_users / total_users
        treatment_ratio = treatment_users / total_users
        
        target_control_ratio = 1 - experiment['treatment_percentage']
        target_treatment_ratio = experiment['treatment_percentage']
        
        # If treatment group is significantly under-represented, increase allocation
        if treatment_ratio < target_treatment_ratio * 0.8:
            adjusted_percentage = min(experiment['treatment_percentage'] * 1.2, 0.7)
        # If control group is under-represented, decrease treatment allocation
        elif control_ratio < target_control_ratio * 0.8:
            adjusted_percentage = max(experiment['treatment_percentage'] * 0.8, 0.3)
        else:
            adjusted_percentage = experiment['treatment_percentage']
        
        return adjusted_percentage
    
    def get_variant(self, experiment_id: str, user_id: int, 
                   force_assignment: Optional[bool] = None,
                   segment_overrides: Optional[Dict[str, str]] = None) -> ExperimentVariant:
        """
        Enhanced variant assignment with dynamic allocation
        
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
        if experiment['status'] != 'running':
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
        
        # Check experiment-level segment overrides
        experiment_segment_overrides = experiment.get('segment_overrides', {})
        if experiment_segment_overrides:
            user_segment = self._get_user_segment(user_id)
            if user_segment in experiment_segment_overrides:
                if experiment_segment_overrides[user_segment] == 'treatment':
                    return ExperimentVariant.TREATMENT
                else:
                    return ExperimentVariant.CONTROL
        
        # Dynamic allocation based on current sample sizes
        treatment_percentage = self._get_dynamic_allocation_percentage(experiment_id)
        
        # Hash user_id for consistent assignment
        import hashlib
        hash_input = f"{experiment_id}:{user_id}".encode()
        hash_val = int(hashlib.md5(hash_input).hexdigest(), 16)
        
        # Determine variant based on hash and treatment percentage
        if (hash_val % 10000) / 10000 < treatment_percentage:
            return ExperimentVariant.TREATMENT
        return ExperimentVariant.CONTROL
    
    def update_experiment_allocation(self, 
                                   experiment_id: str, 
                                   new_treatment_percentage: float,
                                   reason: str = "") -> bool:
        """Dynamically update experiment allocation"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        if not 0 < new_treatment_percentage < 1:
            raise ValueError("Treatment percentage must be between 0 and 1")
        
        experiment = self.experiments[experiment_id]
        old_percentage = experiment['treatment_percentage']
        experiment['treatment_percentage'] = new_treatment_percentage
        
        # Log the change
        change_log = {
            'timestamp': datetime.now(),
            'old_percentage': old_percentage,
            'new_percentage': new_treatment_percentage,
            'reason': reason
        }
        
        if 'allocation_changes' not in experiment:
            experiment['allocation_changes'] = []
        experiment['allocation_changes'].append(change_log)
        
        logger.info(f"Updated allocation for {experiment_id}: {old_percentage:.1%} -> {new_treatment_percentage:.1%}")
        
        return True
    
    def set_segment_override(self, 
                           experiment_id: str,
                           segment: str,
                           variant: str) -> bool:
        """Set segment-specific allocation override"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        if variant not in ['control', 'treatment']:
            raise ValueError("Variant must be 'control' or 'treatment'")
        
        experiment = self.experiments[experiment_id]
        
        if 'segment_overrides' not in experiment:
            experiment['segment_overrides'] = {}
        
        experiment['segment_overrides'][segment] = variant
        
        logger.info(f"Set segment override for {experiment_id}: {segment} -> {variant}")
        
        return True
    
    def remove_segment_override(self, experiment_id: str, segment: str) -> bool:
        """Remove segment-specific allocation override"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        if 'segment_overrides' in experiment and segment in experiment['segment_overrides']:
            del experiment['segment_overrides'][segment]
            logger.info(f"Removed segment override for {experiment_id}: {segment}")
            return True
        
        return False
    
    def get_allocation_summary(self, experiment_id: str) -> Dict[str, Any]:
        """Get detailed allocation summary for experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        control_users = len(experiment['metrics']['control']['users'])
        treatment_users = len(experiment['metrics']['treatment']['users'])
        total_users = control_users + treatment_users
        
        summary = {
            'experiment_id': experiment_id,
            'target_allocation': {
                'control': 1 - experiment['treatment_percentage'],
                'treatment': experiment['treatment_percentage']
            },
            'actual_allocation': {
                'control': control_users / total_users if total_users > 0 else 0,
                'treatment': treatment_users / total_users if total_users > 0 else 0
            },
            'sample_sizes': {
                'control': control_users,
                'treatment': treatment_users,
                'total': total_users
            },
            'current_treatment_percentage': self._get_dynamic_allocation_percentage(experiment_id),
            'segment_overrides': experiment.get('segment_overrides', {}),
            'allocation_changes': experiment.get('allocation_changes', [])
        }
        
        # Calculate balance score (how close actual is to target)
        if total_users > 0:
            target_control = 1 - experiment['treatment_percentage']
            actual_control = control_users / total_users
            balance_score = 1 - abs(target_control - actual_control)
            summary['balance_score'] = balance_score
        
        return summary
    
    def auto_balance_experiment(self, experiment_id: str, min_imbalance_threshold: float = 0.1) -> bool:
        """Automatically balance experiment allocation if imbalanced"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        summary = self.get_allocation_summary(experiment_id)
        
        # Only balance if we have sufficient sample size
        if summary['sample_sizes']['total'] < 200:
            return False
        
        balance_score = summary.get('balance_score', 1.0)
        
        # If balance score is below threshold, adjust allocation
        if balance_score < (1 - min_imbalance_threshold):
            target_treatment = summary['target_allocation']['treatment']
            actual_treatment = summary['actual_allocation']['treatment']
            
            # Adjust allocation towards balance
            if actual_treatment < target_treatment:
                # Increase treatment allocation
                new_percentage = min(target_treatment * 1.1, 0.7)
            else:
                # Decrease treatment allocation
                new_percentage = max(target_treatment * 0.9, 0.3)
            
            reason = f"Auto-balance: balance_score={balance_score:.3f}"
            self.update_experiment_allocation(experiment_id, new_percentage, reason)
            
            logger.info(f"Auto-balanced experiment {experiment_id}: {target_treatment:.1%} -> {new_percentage:.1%}")
            return True
        
        return False
    
    def multi_arm_bandit_allocation(self, experiment_id: str, 
                                  epsilon: float = 0.1, 
                                  min_samples_per_arm: int = 100) -> Dict[str, float]:
        """
        Implement epsilon-greedy multi-arm bandit allocation
        
        Args:
            experiment_id: Experiment to optimize
            epsilon: Exploration rate (0.1 = 10% exploration)
            min_samples_per_arm: Minimum samples before optimizing
            
        Returns:
            Updated allocation percentages
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        control_metrics = experiment['metrics']['control']
        treatment_metrics = experiment['metrics']['treatment']
        
        control_users = len(control_metrics['users'])
        treatment_users = len(treatment_metrics['users'])
        
        # Need minimum samples to start optimization
        if control_users < min_samples_per_arm or treatment_users < min_samples_per_arm:
            return {
                'control': 1 - experiment['treatment_percentage'],
                'treatment': experiment['treatment_percentage']
            }
        
        # Calculate conversion rates
        control_conversions = control_metrics['conversions']
        treatment_conversions = treatment_metrics['conversions']
        
        control_rate = control_conversions / control_users if control_users > 0 else 0
        treatment_rate = treatment_conversions / treatment_users if treatment_users > 0 else 0
        
        # Epsilon-greedy: exploit best arm most of the time, explore randomly
        if treatment_rate > control_rate:
            # Treatment is better
            treatment_allocation = 1 - epsilon + (epsilon / 2)
            control_allocation = epsilon / 2
        else:
            # Control is better
            control_allocation = 1 - epsilon + (epsilon / 2)
            treatment_allocation = epsilon / 2
        
        # Update experiment allocation
        reason = f"Multi-arm bandit: control_rate={control_rate:.3f}, treatment_rate={treatment_rate:.3f}"
        self.update_experiment_allocation(experiment_id, treatment_allocation, reason)
        
        return {
            'control': control_allocation,
            'treatment': treatment_allocation
        }


class AdaptiveExperimentManager:
    """Manager for adaptive experiments that adjust based on results"""
    
    def __init__(self, ab_manager: DynamicABTestManager):
        self.ab_manager = ab_manager
        self.adaptation_rules = {}
        
    def add_adaptation_rule(self, 
                          experiment_id: str,
                          metric: str,
                          threshold: float,
                          action: str,
                          min_sample_size: int = 1000):
        """
        Add automatic adaptation rule for experiment
        
        Args:
            experiment_id: Experiment to adapt
            metric: Metric to monitor ('ctr', 'conversion_rate', etc.)
            threshold: Threshold for triggering action
            action: Action to take ('stop', 'increase_treatment', 'decrease_treatment')
            min_sample_size: Minimum samples before rule activates
        """
        if experiment_id not in self.adaptation_rules:
            self.adaptation_rules[experiment_id] = []
        
        rule = {
            'metric': metric,
            'threshold': threshold,
            'action': action,
            'min_sample_size': min_sample_size,
            'triggered': False
        }
        
        self.adaptation_rules[experiment_id].append(rule)
        
        logger.info(f"Added adaptation rule for {experiment_id}: {metric} {threshold} -> {action}")
    
    def check_adaptation_rules(self, experiment_id: str) -> bool:
        """Check if any adaptation rules should be triggered"""
        if experiment_id not in self.adaptation_rules:
            return False
        
        # Get current experiment results
        try:
            results = self.ab_manager.analyze_experiment(experiment_id)
            if results['status'] != 'complete':
                return False
        except:
            return False
        
        experiment = self.ab_manager.experiments[experiment_id]
        control_users = len(experiment['metrics']['control']['users'])
        treatment_users = len(experiment['metrics']['treatment']['users'])
        
        rules_triggered = False
        
        for rule in self.adaptation_rules[experiment_id]:
            if rule['triggered']:
                continue
            
            # Check minimum sample size
            if control_users < rule['min_sample_size'] or treatment_users < rule['min_sample_size']:
                continue
            
            # Check if threshold is met
            metric_value = self._get_metric_value(results, rule['metric'])
            if metric_value is None:
                continue
            
            if self._should_trigger_rule(rule, metric_value):
                self._execute_adaptation_action(experiment_id, rule)
                rule['triggered'] = True
                rules_triggered = True
        
        return rules_triggered
    
    def _get_metric_value(self, results: Dict[str, Any], metric: str) -> Optional[float]:
        """Extract metric value from results"""
        if metric in results.get('metrics', {}):
            metric_data = results['metrics'][metric]
            return metric_data.get('lift_percentage', 0)
        return None
    
    def _should_trigger_rule(self, rule: Dict[str, Any], metric_value: float) -> bool:
        """Check if rule should be triggered based on metric value"""
        return abs(metric_value) >= rule['threshold']
    
    def _execute_adaptation_action(self, experiment_id: str, rule: Dict[str, Any]):
        """Execute the adaptation action"""
        action = rule['action']
        metric = rule['metric']
        
        if action == 'stop':
            reason = f"Adaptation rule triggered: {metric} exceeded threshold {rule['threshold']}"
            self.ab_manager.stop_experiment(experiment_id, reason)
            
        elif action == 'increase_treatment':
            current_percentage = self.ab_manager.experiments[experiment_id]['treatment_percentage']
            new_percentage = min(current_percentage * 1.2, 0.8)
            reason = f"Adaptation rule: increase treatment due to {metric} performance"
            self.ab_manager.update_experiment_allocation(experiment_id, new_percentage, reason)
            
        elif action == 'decrease_treatment':
            current_percentage = self.ab_manager.experiments[experiment_id]['treatment_percentage']
            new_percentage = max(current_percentage * 0.8, 0.2)
            reason = f"Adaptation rule: decrease treatment due to {metric} performance"
            self.ab_manager.update_experiment_allocation(experiment_id, new_percentage, reason)
        
        logger.info(f"Executed adaptation action for {experiment_id}: {action} due to {metric}")