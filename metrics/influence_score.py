import torch
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .organize_data import (
    PROBLEM_INDEX_KEY,
    ROUND_KEY,
    CONTEXT_KEY
)

INFLUENCE_SCORE_KEY = 'influence_score'
INFLUENCE_MATRIX_KEY = 'influence_matrix'
ROUND_START_KEY = 'round_start'
ROUND_END_KEY = 'round_end'

def get_device() -> torch.device:
    """Automatically select the best available device (CUDA, MPS, or CPU).
    
    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
 
# embeddings = embedding_model.encode(context, normalize_embeddings=True)

def calculate_influence_score(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate the semantic convergence of the data.

    Args:
        data (pd.DataFrame): The data to calculate the diffusion rate from the data. Input DataFrame containing:
          - problem_index: Problem identifier
          - round: Round number
          - text_answer (list[str]): output context of each agent in this round, say [context_0, context_1, ...]

    Returns:
        pd.DataFrame: The data with the semantic convergence calculated, including mean and std of similarity scores.
    """
    # check whether key column in data
    assert PROBLEM_INDEX_KEY in data.columns, f"Semantic convergence error: {PROBLEM_INDEX_KEY} not in data columns"
    assert ROUND_KEY in data.columns, f"Semantic convergence error: {ROUND_KEY} not in data columns"
    assert CONTEXT_KEY in data.columns, f"Semantic convergence error: {CONTEXT_KEY} not in data columns"

    model_path = str(Path(__file__).parent.parent / "nomic-ai" / "nomic-embed-text-v1")
    embedding_model = SentenceTransformer(
        model_name_or_path=model_path, 
        trust_remote_code=True,
        device=get_device()
    )

    influence_scores_all = []

    # Process each problem separately
    for problem_index, problem_data in data.groupby(PROBLEM_INDEX_KEY):
        rounds = sorted(problem_data[ROUND_KEY].unique())
        
        # Precompute embeddings for all rounds and agents
        embeddings = {}
        for r in rounds:
            round_data = problem_data[problem_data[ROUND_KEY] == r]
            embeddings[r] = [
                torch.from_numpy(embedding_model.encode(
                    text if text is not None else '',  # replace None as ''
                    normalize_embeddings=True
                ))
                for text in round_data.iloc[0][CONTEXT_KEY]
            ]
        
        # Calculate influence scores
        for i in range(len(rounds) - 1):
            for j in range(i + 1, len(rounds)):
                round_i = rounds[i]
                round_j = rounds[j]

                # Convert numpy arrays to torch tensors
                embeddings_i = torch.stack(embeddings[round_i])
                embeddings_j = torch.stack(embeddings[round_j])
                
                # Calculate similarity matrix between round i and j
                sim_matrix = cosine_similarity(embeddings_i, embeddings_j)
                
                # Calculate baseline similarity (same round)
                baseline_sim = cosine_similarity(embeddings_i, embeddings_i)
                
                # Calculate influence score
                influence_score = sim_matrix - baseline_sim
                influence_scores_all.append({
                    ROUND_START_KEY: round_i,
                    ROUND_END_KEY: round_j,
                    PROBLEM_INDEX_KEY: problem_index,
                    INFLUENCE_MATRIX_KEY: influence_score
                })
        
    return pd.DataFrame(influence_scores_all)
    