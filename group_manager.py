"""
Module for managing agent groups and group-based communication in multi-agent debate.
Handles group formation, intra/inter-group communication, and summary generation.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json

class GroupManager:
    """Manages agent groups and communication in multi-agent debate"""
    
    def __init__(self, group_structure: List[List[int]] = None, num_agents: int = None):
        """
        Initialize group manager
        
        Args:
            group_structure: List of agent groups, where each group is a list of agent indices
            num_agents: Total number of agents (required if group_structure is None)
        """
        if group_structure is None:
            if num_agents is None:
                raise ValueError("Either group_structure or num_agents must be provided")
                
            # Default grouping: create roughly equal sized groups
            agents_per_group = max(2, num_agents // 3)
            self.group_structure = []
            for i in range(0, num_agents, agents_per_group):
                end = min(i + agents_per_group, num_agents)
                self.group_structure.append(list(range(i, end)))
        else:
            self.group_structure = group_structure
            
        # Validate group structure
        all_agents = [agent for group in self.group_structure for agent in group]
        if len(set(all_agents)) != len(all_agents):
            raise ValueError("Each agent should belong to exactly one group")
            
        # Create mapping from agent to group
        self.agent_to_group = {}
        for group_idx, group in enumerate(self.group_structure):
            for agent_idx in group:
                self.agent_to_group[agent_idx] = group_idx
                
        # Initialize group summaries
        self.group_summaries = [None] * len(self.group_structure)
        
        # Initialize current phase
        self.current_phase = "intra"  # 'intra' or 'inter'
        
        # Track group membership of all agents
        self.num_agents = len([agent for group in self.group_structure for agent in group])
        
    def get_num_groups(self) -> int:
        """Get the number of groups"""
        return len(self.group_structure)
    
    def get_group_for_agent(self, agent_idx: int) -> Optional[int]:
        """Get the group index for a given agent"""
        return self.agent_to_group.get(agent_idx)
    
    def get_group_members(self, group_idx: int) -> List[int]:
        """Get all agent indices for a specific group"""
        if group_idx >= len(self.group_structure):
            return []
        return self.group_structure[group_idx]
    
    def get_other_groups(self, group_idx: int) -> List[List[int]]:
        """Get the agent groups other than the specified group"""
        return [group for i, group in enumerate(self.group_structure) if i != group_idx]

    def generate_group_summary(self, group_idx: int, agent_responses: List[str], client) -> str:
        """
        Generate a summary of responses from a group
        
        Args:
            group_idx: Index of the group
            agent_responses: List of responses from all agents
            client: LLM client for generating summary
            
        Returns:
            str: Summary of group responses
        """
        group = self.group_structure[group_idx]
        if not group:
            return ""
            
        # Collect responses from group members
        group_responses = [agent_responses[i] for i in group if i < len(agent_responses)]
        
        # Create prompt for summary generation
        summary_prompt = [
            # {
            #     "role": "system", 
            #     "content": "Your task is to summarize the key points from multiple responses to the same question. Focus on the main arguments, identify common patterns, and highlight any unique insights. Your summary should be concise (no more than 150 words) while preserving the most important information."
            # },
            {
                "role": "user",
                "content": f"These are the recent/updated and unique opinions from all agents:\n\n" + "\n\n".join([f"Response {i+1}: {resp}" for i, resp in enumerate(group_responses)]) + "Summarize these opinions carefully and completly in no more than 80 words." + "Aggregate and put your final answers in parentheses at the end of your response."
            }
        ]
        
        try:
            # Generate summary
            completion = client.create_chat_completion(
                messages=summary_prompt,
                max_tokens=512,
                temperature=0.0,
            )
            summary = completion["choices"][0]["message"]["content"]
            
            # Store the original summary (don't try to extract from JSON)
            self.group_summaries[group_idx] = summary
            
            return summary
        except Exception as e:
            print(f"Error generating summary for group {group_idx}: {e}")
            # Return concatenated responses as fallback
            return "\n\n".join(group_responses)
            
    def generate_all_summaries(self, agent_responses: List[str], client) -> Dict[int, str]:
        """Generate summaries for all groups"""
        summaries = {}
        for group_idx in range(len(self.group_structure)):
            summary = self.generate_group_summary(group_idx, agent_responses, client)
            summaries[group_idx] = summary
        return summaries
        
    def get_group_summary(self, group_idx: int) -> Optional[str]:
        """Get the stored summary for a group"""
        if group_idx >= len(self.group_summaries):
            return None
        return self.group_summaries[group_idx]
    
    def set_phase(self, phase: str):
        """Set the current communication phase (intra or inter)"""
        if phase not in ["intra", "inter"]:
            raise ValueError("Phase must be 'intra' or 'inter'")
        self.current_phase = phase