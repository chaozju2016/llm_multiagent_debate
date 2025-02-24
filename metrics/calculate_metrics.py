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


if __name__ == "__main__":
    # run in parent folder of metrics with command "python3 -m metrics.calculate_metrics"
    from .semantic_convergence import calculate_semantic_convergence
    from .influence_score import calculate_influence_score
    from .organize_data import process_file

    metrics = {
        "semantic_similarity": calculate_semantic_convergence,
        "influence_matrix": calculate_influence_score,
    }

    metric_calcultor = MetricCalculator()
    for key, func in metrics.items():
        metric_calcultor.register_metric(
            metric_name=key,
            metric_func=func
        )

    file_name = "/Users/tanghuaze/llm_multiagent_debate/data/multi_mmlu_results_er100_agents6_dr5_ratio0.0_range30.p"
    data_df = process_file(file_name)
    metric_dict = metric_calcultor.calculate_all_metrics(
        data=data_df
    )

    print(f"metric_dict ({type(metric_dict)}):")
    for key, metric in metric_dict.items():
        print(f"{key}: {type(metric)}, \n{metric}")
