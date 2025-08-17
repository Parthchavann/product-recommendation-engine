import time
import psutil
import threading
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from datetime import datetime, timedelta

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


class PrometheusMetrics:
    """Prometheus-compatible metrics collector"""
    
    def __init__(self, namespace: str = "recommendation_engine"):
        self.namespace = namespace
        self.metrics = defaultdict(list)
        self.counters = defaultdict(float)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(lambda: defaultdict(list))
        self.summaries = defaultdict(lambda: deque(maxlen=1000))
        
        # System metrics collection
        self._system_metrics_enabled = False
        self._system_metrics_thread = None
        self._stop_event = threading.Event()
        
        logger.info(f"Initialized Prometheus metrics with namespace: {namespace}")
    
    def counter_inc(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment counter metric"""
        full_name = f"{self.namespace}_{name}_total"
        labels = labels or {}
        
        key = (full_name, tuple(sorted(labels.items())))
        self.counters[key] += value
        
        # Store metric point
        metric_point = MetricPoint(
            name=full_name,
            value=self.counters[key],
            timestamp=time.time(),
            labels=labels
        )
        self.metrics[full_name].append(metric_point)
    
    def gauge_set(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set gauge metric value"""
        full_name = f"{self.namespace}_{name}"
        labels = labels or {}
        
        key = (full_name, tuple(sorted(labels.items())))
        self.gauges[key] = value
        
        # Store metric point
        metric_point = MetricPoint(
            name=full_name,
            value=value,
            timestamp=time.time(),
            labels=labels
        )
        self.metrics[full_name].append(metric_point)
    
    def histogram_observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe value in histogram"""
        full_name = f"{self.namespace}_{name}"
        labels = labels or {}
        
        key = (full_name, tuple(sorted(labels.items())))
        self.histograms[key]['observations'].append(value)
        
        # Store metric point
        metric_point = MetricPoint(
            name=f"{full_name}_bucket",
            value=value,
            timestamp=time.time(),
            labels=labels
        )
        self.metrics[full_name].append(metric_point)
    
    def summary_observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe value in summary"""
        full_name = f"{self.namespace}_{name}"
        labels = labels or {}
        
        key = (full_name, tuple(sorted(labels.items())))
        self.summaries[key].append(value)
        
        # Store metric point
        metric_point = MetricPoint(
            name=full_name,
            value=value,
            timestamp=time.time(),
            labels=labels
        )
        self.metrics[full_name].append(metric_point)
    
    def timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager for timing operations"""
        return TimerContext(self, name, labels)
    
    def start_system_metrics_collection(self, interval: float = 10.0):
        """Start collecting system metrics in background"""
        if self._system_metrics_enabled:
            return
        
        self._system_metrics_enabled = True
        self._stop_event.clear()
        
        self._system_metrics_thread = threading.Thread(
            target=self._collect_system_metrics,
            args=(interval,),
            daemon=True
        )
        self._system_metrics_thread.start()
        
        logger.info(f"Started system metrics collection (interval: {interval}s)")
    
    def stop_system_metrics_collection(self):
        """Stop collecting system metrics"""
        if not self._system_metrics_enabled:
            return
        
        self._system_metrics_enabled = False
        self._stop_event.set()
        
        if self._system_metrics_thread:
            self._system_metrics_thread.join(timeout=5.0)
        
        logger.info("Stopped system metrics collection")
    
    def _collect_system_metrics(self, interval: float):
        """Collect system metrics periodically"""
        while not self._stop_event.wait(interval):
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                self.gauge_set("system_cpu_usage_percent", cpu_percent)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self.gauge_set("system_memory_usage_bytes", memory.used)
                self.gauge_set("system_memory_total_bytes", memory.total)
                self.gauge_set("system_memory_usage_percent", memory.percent)
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                self.gauge_set("system_disk_usage_bytes", disk.used)
                self.gauge_set("system_disk_total_bytes", disk.total)
                self.gauge_set("system_disk_usage_percent", (disk.used / disk.total) * 100)
                
                # Network metrics (if available)
                try:
                    net_io = psutil.net_io_counters()
                    self.counter_inc("system_network_bytes_sent", net_io.bytes_sent)
                    self.counter_inc("system_network_bytes_recv", net_io.bytes_recv)
                except:
                    pass
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
    
    def get_metrics_text(self) -> str:
        """Generate Prometheus text format metrics"""
        lines = []
        
        # Add help and type information
        metric_types = set()
        
        for metric_name in self.metrics.keys():
            if metric_name not in metric_types:
                lines.append(f"# HELP {metric_name} Generated metric")
                
                if "_total" in metric_name:
                    lines.append(f"# TYPE {metric_name} counter")
                elif "_bucket" in metric_name:
                    lines.append(f"# TYPE {metric_name.replace('_bucket', '')} histogram")
                else:
                    lines.append(f"# TYPE {metric_name} gauge")
                
                metric_types.add(metric_name)
        
        # Add counter metrics
        for key, value in self.counters.items():
            metric_name, labels_tuple = key
            labels_str = ""
            if labels_tuple:
                labels_list = [f'{k}="{v}"' for k, v in labels_tuple]
                labels_str = "{" + ",".join(labels_list) + "}"
            
            lines.append(f"{metric_name}{labels_str} {value}")
        
        # Add gauge metrics
        for key, value in self.gauges.items():
            metric_name, labels_tuple = key
            labels_str = ""
            if labels_tuple:
                labels_list = [f'{k}="{v}"' for k, v in labels_tuple]
                labels_str = "{" + ",".join(labels_list) + "}"
            
            lines.append(f"{metric_name}{labels_str} {value}")
        
        # Add histogram metrics
        for key, observations in self.histograms.items():
            metric_name, labels_tuple = key
            base_labels = dict(labels_tuple) if labels_tuple else {}
            
            obs_list = observations.get('observations', [])
            if obs_list:
                # Generate buckets
                buckets = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')]
                
                for bucket in buckets:
                    count = sum(1 for obs in obs_list if obs <= bucket)
                    bucket_labels = base_labels.copy()
                    bucket_labels['le'] = str(bucket)
                    
                    labels_list = [f'{k}="{v}"' for k, v in sorted(bucket_labels.items())]
                    labels_str = "{" + ",".join(labels_list) + "}"
                    
                    lines.append(f"{metric_name}_bucket{labels_str} {count}")
                
                # Add count and sum
                base_labels_str = ""
                if base_labels:
                    labels_list = [f'{k}="{v}"' for k, v in sorted(base_labels.items())]
                    base_labels_str = "{" + ",".join(labels_list) + "}"
                
                lines.append(f"{metric_name}_count{base_labels_str} {len(obs_list)}")
                lines.append(f"{metric_name}_sum{base_labels_str} {sum(obs_list)}")
        
        # Add summary metrics
        for key, observations in self.summaries.items():
            metric_name, labels_tuple = key
            base_labels = dict(labels_tuple) if labels_tuple else {}
            
            if observations:
                obs_array = np.array(list(observations))
                
                # Calculate quantiles
                quantiles = [0.5, 0.9, 0.95, 0.99]
                
                for q in quantiles:
                    value = np.percentile(obs_array, q * 100)
                    quant_labels = base_labels.copy()
                    quant_labels['quantile'] = str(q)
                    
                    labels_list = [f'{k}="{v}"' for k, v in sorted(quant_labels.items())]
                    labels_str = "{" + ",".join(labels_list) + "}"
                    
                    lines.append(f"{metric_name}{labels_str} {value}")
                
                # Add count and sum
                base_labels_str = ""
                if base_labels:
                    labels_list = [f'{k}="{v}"' for k, v in sorted(base_labels.items())]
                    base_labels_str = "{" + ",".join(labels_list) + "}"
                
                lines.append(f"{metric_name}_count{base_labels_str} {len(observations)}")
                lines.append(f"{metric_name}_sum{base_labels_str} {sum(observations)}")
        
        return "\n".join(lines) + "\n"
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary"""
        metrics_dict = {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'histograms': {},
            'summaries': {}
        }
        
        # Process histograms
        for key, observations in self.histograms.items():
            obs_list = observations.get('observations', [])
            if obs_list:
                metrics_dict['histograms'][key] = {
                    'count': len(obs_list),
                    'sum': sum(obs_list),
                    'mean': np.mean(obs_list),
                    'percentiles': {
                        'p50': np.percentile(obs_list, 50),
                        'p90': np.percentile(obs_list, 90),
                        'p95': np.percentile(obs_list, 95),
                        'p99': np.percentile(obs_list, 99),
                    }
                }
        
        # Process summaries
        for key, observations in self.summaries.items():
            if observations:
                obs_array = np.array(list(observations))
                metrics_dict['summaries'][key] = {
                    'count': len(observations),
                    'sum': float(np.sum(obs_array)),
                    'mean': float(np.mean(obs_array)),
                    'percentiles': {
                        'p50': float(np.percentile(obs_array, 50)),
                        'p90': float(np.percentile(obs_array, 90)),
                        'p95': float(np.percentile(obs_array, 95)),
                        'p99': float(np.percentile(obs_array, 99)),
                    }
                }
        
        return metrics_dict
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
        self.summaries.clear()
        
        logger.info("Reset all metrics")


class TimerContext:
    """Context manager for timing operations"""
    
    def __init__(self, metrics: PrometheusMetrics, name: str, labels: Optional[Dict[str, str]] = None):
        self.metrics = metrics
        self.name = name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics.histogram_observe(f"{self.name}_duration_seconds", duration, self.labels)


# Global metrics instance
metrics = PrometheusMetrics()


def get_metrics() -> PrometheusMetrics:
    """Get global metrics instance"""
    return metrics


class ModelMetrics:
    """Specialized metrics for ML models"""
    
    def __init__(self, metrics: PrometheusMetrics):
        self.metrics = metrics
    
    def record_inference_time(self, model_name: str, duration: float):
        """Record model inference time"""
        self.metrics.histogram_observe(
            "model_inference_duration_seconds",
            duration,
            {"model": model_name}
        )
    
    def record_prediction_count(self, model_name: str, count: int = 1):
        """Record number of predictions made"""
        self.metrics.counter_inc(
            "model_predictions",
            count,
            {"model": model_name}
        )
    
    def record_model_accuracy(self, model_name: str, accuracy: float):
        """Record model accuracy metric"""
        self.metrics.gauge_set(
            "model_accuracy",
            accuracy,
            {"model": model_name}
        )
    
    def record_cache_hit(self, cache_type: str):
        """Record cache hit"""
        self.metrics.counter_inc(
            "cache_hits",
            1,
            {"type": cache_type}
        )
    
    def record_cache_miss(self, cache_type: str):
        """Record cache miss"""
        self.metrics.counter_inc(
            "cache_misses",
            1,
            {"type": cache_type}
        )
    
    def record_recommendation_served(self, user_id: str, num_items: int):
        """Record recommendation served to user"""
        self.metrics.counter_inc(
            "recommendations_served",
            1,
            {"user_type": "registered" if user_id.isdigit() else "anonymous"}
        )
        
        self.metrics.histogram_observe(
            "recommendation_list_size",
            num_items
        )


class APIMetrics:
    """Specialized metrics for API endpoints"""
    
    def __init__(self, metrics: PrometheusMetrics):
        self.metrics = metrics
    
    def record_request(self, endpoint: str, method: str, status_code: int, duration: float):
        """Record API request metrics"""
        labels = {
            "endpoint": endpoint,
            "method": method,
            "status": str(status_code)
        }
        
        # Request count
        self.metrics.counter_inc("api_requests", 1, labels)
        
        # Request duration
        self.metrics.histogram_observe("api_request_duration_seconds", duration, labels)
        
        # Error rate tracking
        if status_code >= 400:
            self.metrics.counter_inc("api_errors", 1, labels)
    
    def record_active_connections(self, count: int):
        """Record number of active connections"""
        self.metrics.gauge_set("api_active_connections", count)
    
    def record_queue_size(self, queue_name: str, size: int):
        """Record queue size"""
        self.metrics.gauge_set(
            "queue_size",
            size,
            {"queue": queue_name}
        )