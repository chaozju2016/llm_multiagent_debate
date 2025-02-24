import pandas as pd
from typing import Callable, Dict, Any

class MetricCalculator:
    def __init__(self):
        self.metric = {}
    
    def register_metric(self, metric_name: str, metric_func: Callable, **kwargs):
        """Register a metric function with its name and optional arguments.

        Args:
            metric_name (str): The name of the metric.
            metric_func (Callable): The function to calculate the metric.
            **kwargs: Optional arguments for the metric function.
        """
        self.metric[metric_name] = (metric_func, kwargs)

    def calculate_single_metric(self, data: pd.DataFrame, metric_name: str) -> float:
        """Calculate a single metric from the data.

        Args:
            data (pd.DataFrame): The data to calculate the metric from.
            metric_name (str): The name of the metric to calculate.
            **kwargs: Optional arguments for the metric function.
        """
        assert metric_name in self.metric, f"Metric {metric_name} not found."
        return self.metric[metric_name][0](data, **self.metric[metric_name][1])
    
    def calculate_all_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate all metrics from the data.

        Args:
            data (pd.DataFrame): The data to calculate the metrics from the data.

        Returns:
            Dict[str, Any]: A dictionary of metric names and their calculated values.
        """
        return {metric_name: self.calculate_single_metric(data, metric_name) for metric_name in self.metric.keys()}
    