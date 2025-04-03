"""
Module for generating and managing various communication topologies for multi-agent debate systems.
Implements different strategies including fully-connected, ring, star, and group-based topologies.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field

@dataclass
class TopologyConfig:
    """Configuration for communication topology generation"""
    num_agents: int
    # General configuration
    topology_type: str = "fully_connected"  # fully_connected, ring, star, group, s2mad, asmad
    
    # Group configuration (for 'group' and 's2mad' topologies)
    group_structure: List[List[int]] = None
    intra_group_rounds: int = 2
    inter_group_rounds: int = 1
    
    # S2-MAD configuration
    decision_threshold: float = 0.6
    
    # ASMAD configuration
    trust_radius_init: float = 0.0
    trust_radius_final: float = 0.8
    similarity_outlier_threshold: float = 0.5
    
    def __post_init__(self):
        # Initialize default group structure if none provided
        if self.group_structure is None and self.topology_type in ['group', 's2mad']:
            # Default to roughly equal-sized groups
            agents_per_group = max(2, self.num_agents // 3)
            self.group_structure = []
            for i in range(0, self.num_agents, agents_per_group):
                end = min(i + agents_per_group, self.num_agents)
                self.group_structure.append(list(range(i, end)))


class TopologyManager:
    """Manages communication topology between agents based on different strategies"""
    
    def __init__(self, config: TopologyConfig):
        self.config = config
        self.current_round = 0
        self.total_rounds = 0
        self.stage = "intra"  # 'intra' or 'inter'
        
        # Initialize group information
        self._init_groups()
        
    def _init_groups(self):
        """Initialize group membership information"""
        self.agent_to_group = {}
        if self.config.group_structure:
            for group_idx, group in enumerate(self.config.group_structure):
                for agent_idx in group:
                    self.agent_to_group[agent_idx] = group_idx
    
    def set_total_rounds(self, total_rounds: int):
        """Set the total number of debate rounds"""
        self.total_rounds = total_rounds
    
    def next_round(self):
        """Advance to the next round and update internal state"""
        self.current_round += 1
        
        # For group-based topologies, determine if we're in intra or inter group phase
        if self.config.topology_type in ['group', 's2mad']:
            round_in_cycle = (self.current_round - 1) % (self.config.intra_group_rounds + self.config.inter_group_rounds)
            if round_in_cycle < self.config.intra_group_rounds:
                self.stage = "intra"
            else:
                self.stage = "inter"
    
    def get_trust_radius(self) -> float:
        """Calculate the adaptive trust radius based on debate progress"""
        if self.total_rounds <= 1:
            return self.config.trust_radius_final
        
        progress = min(1.0, self.current_round / (self.total_rounds - 1))
        return self.config.trust_radius_init + progress * (self.config.trust_radius_final - self.config.trust_radius_init)
    
    def generate_mask(self, sim_matrix: np.ndarray = None) -> np.ndarray:
        """
        Generate visibility mask based on the specified topology and current state
        
        Args:
            sim_matrix: Similarity matrix between agents (for similarity-based strategies)
            
        Returns:
            np.ndarray: Boolean mask matrix of shape (num_agents, num_agents)
                        where True indicates visibility
        """
        n = self.config.num_agents
        
        # Generate appropriate mask based on topology type
        if self.config.topology_type == "fully_connected":
            return self._fully_connected_mask()
        elif self.config.topology_type == "ring":
            return self._ring_mask()
        elif self.config.topology_type == "star":
            return self._star_mask()
        elif self.config.topology_type == "group":
            return self._group_mask()
        elif self.config.topology_type == "s2mad":
            if sim_matrix is None:
                raise ValueError("Similarity matrix required for S2-MAD topology")
            return self._s2mad_mask(sim_matrix)
        elif self.config.topology_type == "asmad":
            if sim_matrix is None:
                raise ValueError("Similarity matrix required for ASMAD topology")
            return self._asmad_mask(sim_matrix)
        else:
            raise ValueError(f"Unknown topology type: {self.config.topology_type}")
    
    def _fully_connected_mask(self) -> np.ndarray:
        """Generate fully connected mask where all agents can see all others"""
        n = self.config.num_agents
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, True)  # Ensure agents can see themselves
        return mask
    
    def _ring_mask(self) -> np.ndarray:
        """Generate ring topology mask where agents can only see adjacent agents"""
        n = self.config.num_agents
        mask = np.zeros((n, n), dtype=bool)
        
        # Each agent can see itself and its neighbors
        for i in range(n):
            mask[i, i] = True  # Self
            mask[i, (i - 1) % n] = True  # Left neighbor
            mask[i, (i + 1) % n] = True  # Right neighbor
            
        return mask
    
    def _star_mask(self) -> np.ndarray:
        """Generate star topology mask where agents can only see the central agent"""
        n = self.config.num_agents
        mask = np.zeros((n, n), dtype=bool)
        
        # Define central agent (usually the first one)
        central_agent = 0
        
        # All agents can see central agent
        mask[:, central_agent] = True
        
        # Central agent can see all agents
        mask[central_agent, :] = True
        
        # All agents can see themselves
        np.fill_diagonal(mask, True)
        
        return mask
    
    def _group_mask(self) -> np.ndarray:
        """
        Generate group topology mask based on current stage (intra or inter group)
        """
        n = self.config.num_agents
        mask = np.zeros((n, n), dtype=bool)
        
        # Ensure all agents can see themselves
        np.fill_diagonal(mask, True)
        
        if self.stage == "intra":
            # Intra-group: Agents can see all other agents in same group
            for group in self.config.group_structure:
                for i in group:
                    for j in group:
                        mask[i, j] = True
        else:
            # Inter-group: All agents can see other groups' agents
            # This is a key change from the original implementation that used representatives
            for group_i in self.config.group_structure:
                for group_j in self.config.group_structure:
                    if group_i != group_j:  # Only cross-group visibility
                        for i in group_i:
                            for j in group_j:
                                mask[i, j] = True
                                
            # Agents still see members of their own group
            for group in self.config.group_structure:
                for i in group:
                    for j in group:
                        mask[i, j] = True
                        
        return mask
    
    def _s2mad_mask(self, sim_matrix: np.ndarray) -> np.ndarray:
        """
        Generate mask for Selective Sparse MAD (S2-MAD)
        Uses a decision mechanism based on similarity to determine interactions
        """
        n = self.config.num_agents
        mask = np.zeros((n, n), dtype=bool)
        
        # Ensure all agents can see themselves
        np.fill_diagonal(mask, True)
        
        # Create base mask according to current stage (intra or inter)
        if self.stage == "intra":
            # First create group visibility as in GroupDebate
            for group in self.config.group_structure:
                for i in group:
                    for j in group:
                        # Check similarity threshold for selective participation
                        # Only allow interaction if opinions differ enough
                        if i != j and sim_matrix[i, j] < self.config.decision_threshold:
                            mask[i, j] = True
        else:
            # For inter-group communication, use representatives
            # For each group, find agent with most distinct view (lowest avg similarity)
            representatives = []
            for group in self.config.group_structure:
                if len(group) == 1:
                    representatives.append(group[0])
                    continue
                    
                # Calculate average similarity for each agent in group
                avg_similarities = [np.mean([sim_matrix[i, j] for j in group if j != i]) for i in group]
                # Agent with lowest similarity to others represents most diverse opinion
                rep_idx = group[np.argmin(avg_similarities)]
                representatives.append(rep_idx)
            
            # Representatives can see representatives from other groups
            for i, rep_i in enumerate(representatives):
                for j, rep_j in enumerate(representatives):
                    if i != j and sim_matrix[rep_i, rep_j] < self.config.decision_threshold:
                        mask[rep_i, rep_j] = True
                        
            # Other agents can still see their group members
            for group in self.config.group_structure:
                for i in group:
                    for j in group:
                        if i != j:
                            mask[i, j] = True
                            
        return mask
    
    def _asmad_mask(self, sim_matrix: np.ndarray) -> np.ndarray:
        """
        Generate mask for Adaptive Sparse MAD (ASMAD)
        Implements trust boundary and outlier mechanisms from opinion dynamics
        """
        n = self.config.num_agents
        
        # Get current trust radius
        radius = self.get_trust_radius()
        
        # Generate mask based on similarity and trust radius
        mask = (sim_matrix >= 0) & (sim_matrix <= 1.0)  # Base mask - all True
        
        # Apply trust radius - only interact if within confidence bound
        mask = (sim_matrix >= radius) & mask
        
        # Handle outliers using HK model approach
        mean_sims = np.mean(sim_matrix, axis=1)
        outliers = (mean_sims < self.config.similarity_outlier_threshold) & (mean_sims < np.mean(mean_sims) - np.std(mean_sims))
        
        # Let outliers see everyone
        mask[outliers, :] = True
        
        # But temporarily hide outliers from others
        mask[:, outliers] = False
        
        # Ensure agents can see themselves
        np.fill_diagonal(mask, True)
        
        return mask
    
    # These methods are no longer used with the new design, but kept for backward compatibility
    def is_group_representative(self, agent_idx: int, sim_matrix: np.ndarray = None) -> bool:
        """
        Legacy method - not actively used in the new design without representatives
        All group members now participate in inter-group communication
        """
        return True
    
    def get_group_for_agent(self, agent_idx: int) -> Optional[int]:
        """Get the group index for a given agent"""
        if self.config.group_structure is None:
            return None
            
        for group_idx, group in enumerate(self.config.group_structure):
            if agent_idx in group:
                return group_idx
        return None
    
    def get_agent_group(self, agent_idx: int) -> Optional[int]:
        """Get the group index for a given agent"""
        return self.agent_to_group.get(agent_idx)
    
    def get_group_members(self, group_idx: int) -> List[int]:
        """Get all agent indices for a specific group"""
        if self.config.group_structure is None or group_idx >= len(self.config.group_structure):
            return []
        return self.config.group_structure[group_idx]
    
    def is_intra_group_phase(self) -> bool:
        """Check if currently in intra-group discussion phase"""
        return self.stage == "intra"
    
    def is_inter_group_phase(self) -> bool:
        """Check if currently in inter-group communication phase"""
        return self.stage == "inter"