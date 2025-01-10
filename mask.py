"""
Module for generating and managing visibility masks for multi-agent communication.
Provides different strategies for mask generation with controlled visibility ratios.
"""

import numpy as np
from typing import Optional, Literal, Union
from dataclasses import dataclass

@dataclass
class MaskConfig:
    """Configuration for mask generation"""
    num_agents: int
    visibility_ratio: float
    strategy: Literal['fixed_ratio', 'independent', 'symmetric'] = 'fixed_ratio'
    
    def __post_init__(self):
        if not 0 <= self.visibility_ratio <= 1:
            raise ValueError("visibility_ratio must be between 0 and 1")
        if self.num_agents < 1:
            raise ValueError("num_agents must be positive")

class MaskGenerator:
    """Generates visibility masks for agent communication"""
    
    @staticmethod
    def generate(config: MaskConfig) -> np.ndarray:
        """
        Generate visibility mask based on specified configuration
        
        Args:
            config: MaskConfig object containing generation parameters
            
        Returns:
            np.ndarray: Boolean mask matrix of shape (num_agents, num_agents)
        """
        if config.strategy == 'fixed_ratio':
            return MaskGenerator._fixed_ratio_mask(config)
        elif config.strategy == 'independent':
            return MaskGenerator._independent_mask(config)
        elif config.strategy == 'symmetric':
            return MaskGenerator._symmetric_mask(config)
        else:
            raise ValueError(f"Unknown strategy: {config.strategy}")
    
    @staticmethod
    def _fixed_ratio_mask(config: MaskConfig) -> np.ndarray:
        """
        Generate mask with exactly visibility_ratio * N * N visible elements
        """
        total_elements = config.num_agents * config.num_agents
        num_visible = int(total_elements * config.visibility_ratio)
        
        # Create 1D array with exact number of True values
        mask = np.zeros(total_elements, dtype=bool)
        mask[:num_visible] = True
        np.random.shuffle(mask)
        
        return mask.reshape((config.num_agents, config.num_agents))
    
    @staticmethod
    def _independent_mask(config: MaskConfig) -> np.ndarray:
        """
        Generate mask where each element is independently sampled
        """
        return np.random.random((config.num_agents, config.num_agents)) < config.visibility_ratio
    
    @staticmethod
    def _symmetric_mask(config: MaskConfig) -> np.ndarray:
        """
        Generate symmetric mask where if agent i can see agent j, 
        then agent j can also see agent i
        """
        # Generate upper triangular mask
        upper = np.random.random(
            (config.num_agents, config.num_agents)
        ) < config.visibility_ratio
        
        # Make it symmetric by copying upper triangle to lower triangle
        mask = np.logical_or(upper, upper.T)
        return mask

def apply_mask(messages: list, mask: np.ndarray, agent_idx: int) -> list:
    """
    Apply visibility mask to agent messages
    
    Args:
        messages: List of messages from other agents
        mask: Visibility mask matrix
        agent_idx: Index of current agent
        
    Returns:
        list: Masked messages where invisible messages are replaced with None
    """
    agent_mask = mask[agent_idx]
    return [
        msg if visible else None 
        for msg, visible in zip(messages, agent_mask)
    ]

# Example usage:
if __name__ == "__main__":
    # Example configuration
    config = MaskConfig(
        num_agents=3,
        visibility_ratio=0.5,
        strategy='fixed_ratio'
    )
    
    # Generate mask
    generator = MaskGenerator()
    mask = generator.generate(config)
    
    print("Generated mask:")
    print(mask)
    
    # Example of applying mask
    messages = ["Agent 1 msg", "Agent 2 msg", "Agent 3 msg"]
    masked_messages = apply_mask(messages, mask, agent_idx=0)
    print(f"Masked messages for agent 0:{apply_mask(messages, mask, agent_idx=0)}")
    print(f"Masked messages for agent 1:{apply_mask(messages, mask, agent_idx=1)}")
    print(f"Masked messages for agent 2:{apply_mask(messages, mask, agent_idx=2)}")
