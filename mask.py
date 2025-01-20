"""
Module for generating and managing visibility masks for multi-agent communication.
Provides different strategies for mask generation with controlled visibility ratios.
"""

import numpy as np
from typing import Optional, Literal, Union, Tuple
from dataclasses import dataclass

@dataclass
class MaskConfig:
    """Configuration for mask generation"""
    num_agents: int
    visibility_ratio: float
    strategy: Literal['fixed_ratio', 'independent', 'symmetric', 'similarity'] = 'fixed_ratio'
    similarity_visible_range: Optional[Tuple[float, float]] = None
    similarity_outlier_threshold: Optional[float] = 0.5
    
    def __post_init__(self):
        if not 0 <= self.visibility_ratio <= 1:
            raise ValueError("visibility_ratio must be between 0 and 1")
        if self.num_agents < 1:
            raise ValueError("num_agents must be positive")

class MaskGenerator:
    """Generates visibility masks for agent communication"""
    
    @staticmethod
    def generate(config: MaskConfig, sim_matrix: np.ndarray =None, range_index: int = None) -> np.ndarray:
        """
        Generate visibility mask based on specified configuration
        
        Args:
            config: MaskConfig object containing generation parameters
            
        Returns:
            np.ndarray: Boolean mask matrix of shape (num_agents, num_agents)
        """
        if config.strategy == 'fixed_ratio':
            mask = MaskGenerator._fixed_ratio_mask(config)
#            mask = np.logical_or(np.eye(mask.shape[0],mask.shape[1]), mask)
            return mask
        elif config.strategy == 'independent':
            return MaskGenerator._independent_mask(config)
        elif config.strategy == 'symmetric':
            return MaskGenerator._symmetric_mask(config)
        elif config.strategy == 'similarity' and sim_matrix is not None:
            return MaskGenerator._similarity_mask(config, sim_matrix, range_index)
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
    
    @staticmethod
    def _similarity_mask(
        config: MaskConfig,
        sim_matrix: np.ndarray, 
        range_index: int,
        enable_asymmetric: bool = True
        ) -> np.ndarray:
        """
        根据相似度矩阵生成mask matrix
        
        Args:
            sim_matrix: 相似度矩阵 shape=(num_agents, num_agents)
            enable_asymmetric: 是否启用非对称mask
        
        Returns:
            mask_matrix: bool矩阵,True表示可见
        """
        low, high = config.similarity_visible_range[min(len(config.similarity_visible_range),range_index)]
        mask = (sim_matrix >= low) & (sim_matrix <= high)
        
        if range_index >=2 and enable_asymmetric:
            # 计算每个agent的平均相似度
            mean_sims = np.mean(sim_matrix, axis=1)
            # 找出离群者(平均相似度显著低于整体)
            outliers = (mean_sims < config.similarity_outlier_threshold) & (mean_sims < np.mean(mean_sims) - np.std(mean_sims))
            
            # 让离群者可以看到所有人
            mask[outliers] = True
            # 但其他人暂时看不到离群者
            mask[:, outliers] = False
        
        # 确保对角线为True(自己总是可以看到自己)
        np.fill_diagonal(mask, True)
        
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
