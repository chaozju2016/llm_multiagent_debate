"""
Module for decision-making mechanisms in multi-agent debate frameworks.
Implements various strategies to determine agent participation and information filtering.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import re

class DecisionMaker:
    """Base class for decision-making mechanisms in multi-agent debate"""
    
    def __init__(self, num_agents: int, strategy: str = "always_participate"):
        """
        Initialize decision maker
        
        Args:
            num_agents: Number of agents in the debate
            strategy: Decision strategy to use
        """
        self.num_agents = num_agents
        self.strategy = strategy
        self.answer_history = []
    
    def should_participate(self, agent_idx: int, sim_matrix: np.ndarray, 
                          current_round: int, total_rounds: int) -> bool:
        """
        Determine if an agent should participate in the current debate round
        
        Args:
            agent_idx: Index of the agent making the decision
            sim_matrix: Similarity matrix between agents
            current_round: Current debate round
            total_rounds: Total number of debate rounds
            
        Returns:
            bool: True if agent should participate, False otherwise
        """
        if self.strategy == "always_participate":
            return True
        elif self.strategy == "s2mad":
            return self._s2mad_decision(agent_idx, sim_matrix, current_round, total_rounds)
        else:
            return True
    
    def _s2mad_decision(self, agent_idx: int, sim_matrix: np.ndarray, 
                      current_round: int, total_rounds: int) -> bool:
        """
        Implementation of S²-MAD decision mechanism
        Only participate if there are differing viewpoints
        
        Args:
            agent_idx: Index of the agent making the decision
            sim_matrix: Similarity matrix between agents
            current_round: Current debate round
            total_rounds: Total number of debate rounds
            
        Returns:
            bool: True if agent should participate, False otherwise
        """
        if current_round == 0:
            return True  # Always participate in first round
        
        # Get similarity scores with other agents
        similarities = sim_matrix[agent_idx]
        
        # Determine threshold - typically around 0.6-0.7
        # Can be adapted based on debate progress
        threshold = 0.65
        
        # Check if there are any agents with significantly different opinions
        different_opinions = np.any(similarities < threshold)
        
        return different_opinions
    
    def filter_redundant_information(self, agent_idx: int, 
                                   messages: List[Dict[str, Any]], 
                                   sim_matrix: np.ndarray,
                                   threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Filter out redundant information from other agents
        
        Args:
            agent_idx: Index of the agent receiving information
            messages: List of messages from other agents
            sim_matrix: Similarity matrix between agents
            threshold: Similarity threshold for filtering
            
        Returns:
            List[Dict[str, Any]]: Filtered list of messages
        """
        if not messages:
            return []
            
        # Get indices of other agents
        other_indices = [i for i in range(self.num_agents) if i != agent_idx]
        
        # Filter messages based on similarity
        filtered_messages = []
        for idx, msg in zip(other_indices, messages):
            # If similarity is below threshold, keep the message
            if sim_matrix[agent_idx, idx] < threshold:
                filtered_messages.append(msg)
        
        return filtered_messages
    
    def update_answer_history(self, answers: List[str]):
        """
        Update history of agent answers
        
        Args:
            answers: List of current answers from all agents
        """
        self.answer_history.append(answers)
    
    def compute_stability_scores(self) -> np.ndarray:
        """
        Compute stability scores for all agents based on answer history
        
        Returns:
            np.ndarray: Array of stability scores for each agent
        """
        if len(self.answer_history) < 2:
            return np.zeros(self.num_agents)
            
        stability_scores = np.zeros(self.num_agents)
        
        for i in range(self.num_agents):
            # Count how many times agent changed their answer
            changes = 0
            for t in range(1, len(self.answer_history)):
                prev_answer = self.answer_history[t-1][i] if i < len(self.answer_history[t-1]) else None
                curr_answer = self.answer_history[t][i] if i < len(self.answer_history[t]) else None
                
                if prev_answer is not None and curr_answer is not None and prev_answer != curr_answer:
                    changes += 1
            
            # Convert to stability score (0-1, where 1 is most stable)
            if len(self.answer_history) > 1:
                stability_scores[i] = 1 - (changes / (len(self.answer_history) - 1))
        
        return stability_scores

    @staticmethod
    def parse_answer(response: str) -> Optional[str]:
        """Extract answer from response text"""
        # Look for pattern like "(A)" or "(B)" etc.
        matches = re.findall(r'[\(\s]+([A-D])[\)\s]+', response)
        
        # Return the last matching answer if found
        if matches:
            return matches[-1].upper()
        
        # Additional pattern matching for answers in other formats
        # Look for single letter answers
        matches = re.findall(r'\b([A-D])\b', response)
        if matches:
            return matches[-1].upper()
            
        return None
    
    @staticmethod
    def find_majority_answer(answers: List[str]) -> Optional[str]:
        """Find the majority answer from a list of answers"""
        if not answers:
            return None
            
        # Filter out None values
        valid_answers = [a for a in answers if a is not None]
        if not valid_answers:
            return None
            
        # Count occurrences
        counts = {}
        for answer in valid_answers:
            counts[answer] = counts.get(answer, 0) + 1
            
        # Find answer with highest count
        max_count = 0
        majority_answer = None
        for answer, count in counts.items():
            if count > max_count:
                max_count = count
                majority_answer = answer
                
        return majority_answer

class OpinionDynamicsDecisionMaker(DecisionMaker):
    """Decision maker based on opinion dynamics models like HK and Deffuant"""
    
    def __init__(self, num_agents: int, trust_radius_init: float = 0.0, 
                trust_radius_final: float = 0.8, outlier_threshold: float = 0.5):
        """
        Initialize opinion dynamics decision maker
        
        Args:
            num_agents: Number of agents in the debate
            trust_radius_init: Initial trust radius
            trust_radius_final: Final trust radius
            outlier_threshold: Threshold for identifying outliers
        """
        super().__init__(num_agents, strategy="opinion_dynamics")
        self.trust_radius_init = trust_radius_init
        self.trust_radius_final = trust_radius_final
        self.outlier_threshold = outlier_threshold
    
    def compute_trust_radius(self, current_round: int, total_rounds: int) -> float:
        """
        Compute trust radius based on debate progress
        
        Args:
            current_round: Current debate round
            total_rounds: Total number of debate rounds
            
        Returns:
            float: Current trust radius
        """
        if total_rounds <= 1:
            return self.trust_radius_final
            
        progress = min(1.0, current_round / (total_rounds - 1))
        return self.trust_radius_init + progress * (self.trust_radius_final - self.trust_radius_init)
    
    def identify_outliers(self, sim_matrix: np.ndarray) -> np.ndarray:
        """
        Identify outlier agents based on similarity matrix
        
        Args:
            sim_matrix: Similarity matrix between agents
            
        Returns:
            np.ndarray: Boolean array where True indicates an outlier
        """
        # Calculate mean similarity for each agent
        mean_sims = np.mean(sim_matrix, axis=1)
        
        # Identify outliers - agents with significantly lower mean similarity
        outliers = (mean_sims < self.outlier_threshold) & (mean_sims < np.mean(mean_sims) - np.std(mean_sims))
        
        return outliers
    
    def compute_influence_weights(self, sim_matrix: np.ndarray, 
                                current_round: int, total_rounds: int) -> np.ndarray:
        """
        Compute influence weights between agents based on similarity and debate progress
        
        Args:
            sim_matrix: Similarity matrix between agents
            current_round: Current debate round
            total_rounds: Total number of debate rounds
            
        Returns:
            np.ndarray: Matrix of influence weights
        """
        n = self.num_agents
        weights = np.zeros((n, n))
        
        # Get current trust radius
        radius = self.compute_trust_radius(current_round, total_rounds)
        
        # Get stability scores
        stability = self.compute_stability_scores()
        
        # Compute progress factor (0-1)
        progress = min(1.0, current_round / max(1, total_rounds - 1))
        
        # Base self-influence increases with progress
        base_self_influence = 0.3 + 0.5 * progress
        
        for i in range(n):
            # Self-influence adjusted by stability
            self_weight = base_self_influence * (1 + 0.2 * stability[i])
            weights[i, i] = min(0.8, max(0.3, self_weight))
            
            for j in range(n):
                if i == j:
                    continue
                
                # In early rounds, prioritize diversity
                if current_round < total_rounds // 2:
                    if sim_matrix[i, j] > radius:
                        # base_weight = 0.7 * (1 - sim_matrix[i, j])
                        base_weight = 0.7 * sim_matrix[i, j]
                    else:
                        base_weight = 0.1
                # In later rounds, prioritize consensus
                else:
                    if sim_matrix[i, j] > radius:
                        base_weight = 0.7 * sim_matrix[i, j]
                    else:
                        base_weight = 0.1
                
                # Adjust weight based on stability of other agent
                weights[i, j] = min(0.8, max(0.1, base_weight * (1 + 0.2 * stability[j])))
        
        # Row-normalize the weights
        row_sums = weights.sum(axis=1, keepdims=True)
        normalized_weights = weights / np.maximum(row_sums, 1e-10)
        
        return normalized_weights