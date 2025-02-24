import pandas as pd
import numpy as np
from typing import Dict, Any
from collections import defaultdict

from .organize_data import (
    PROBLEM_INDEX_KEY,
    ROUND_KEY,
    SIM_MATRIX_KEY,
)

MEAN_SIMI_KEY = 'mean_similarity'
STD_SIMI_KEY = 'std_similarity'


def calculate_semantic_convergence(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the semantic convergence of the data.

    Args:
        data (pd.DataFrame): The data to calculate the diffusion rate from the data. Input DataFrame containing:
          - problem_index: Problem identifier
          - round: Round number
          - sim_matrix: Similarity matrix between agent responses

    Returns:
        pd.DataFrame: The data with the semantic convergence calculated, including mean and std of similarity scores.
    """
    # check whether key column in data
    assert PROBLEM_INDEX_KEY in data.columns, f"Semantic convergence error: {PROBLEM_INDEX_KEY} not in data columns"
    assert ROUND_KEY in data.columns, f"Semantic convergence error: {ROUND_KEY} not in data columns"
    assert SIM_MATRIX_KEY in data.columns, f"Semantic convergence error: {SIM_MATRIX_KEY} not in data columns"
    
    results = []
    
    # group by problems and rounds
    for (problem_idx, round_num), group in data.groupby([PROBLEM_INDEX_KEY, ROUND_KEY]):
        # get similarity matrix for this group
        sim_matrix = group[SIM_MATRIX_KEY].iloc[0]
        
        # calculate mean and std of similarity scores
        # exclude diagonal elements (self-similarity)
        mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
        similarities = sim_matrix[mask]
        
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        results.append({
            PROBLEM_INDEX_KEY: problem_idx,
            ROUND_KEY: round_num,
            MEAN_SIMI_KEY: mean_sim,
            STD_SIMI_KEY: std_sim,
        })
    
    return pd.DataFrame(results)