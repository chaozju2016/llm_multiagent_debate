"""
Module for constructing messages in multi-agent debate based on different communication strategies.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple

class MessageConstructor:
    """Constructs messages for agents based on debate strategy and topology"""
    
    def __init__(self, system_message: str = "Welcome to the debate!", output_format: str = ""):
        """
        Initialize message constructor
        
        Args:
            system_message: System message to include at the beginning of all messages
            output_format: Format instructions for the output
        """
        self.system_message = system_message
        self.output_format = output_format
    
    def construct_initial_message(self, question: str) -> List[Dict[str, str]]:
        """
        Construct initial message to agent
        
        Args:
            question: The question to answer
            
        Returns:
            List[Dict[str, str]]: Message in the format expected by OpenAI API
        """
        # For MMLU format
        if "A)" in question and "B)" in question and "C)" in question and "D)" in question:
            return [
                {
                    "role": "system",
                    "content": self.system_message,
                },
                {
                    "role": "user",
                    "content": f"Can you answer the following question as accurately as possible? {question}? Explain your answer, {self.output_format}"
                }
            ]
        # For general math format
        else:
            return [
                {
                    "role": "system",
                    "content": self.system_message,
                },
                {
                    "role": "user",
                    "content": f"Can you solve the following math problem? {question} Explain your reasoning. {self.output_format}"
                }
            ]
    
    def construct_reflection_message(self, self_response: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Construct message asking agent to reflect on their own answer
        
        Args:
            self_response: The agent's own response
            
        Returns:
            List[Dict[str, str]]: Message in the format expected by OpenAI API
        """
        response_text = self_response["content"] if isinstance(self_response, dict) else self_response
        return [
            {
                "role": "user", 
                "content": f"Your previous answer is {response_text}. Examine your solution can you provide an updated answer? {self.output_format}"
            }
        ]
    
    def construct_message_with_visible_responses(self, 
                                              visible_responses: List[Dict[str, str]],
                                              response_indices: List[int],
                                              weights: Optional[np.ndarray] = None) -> List[Dict[str, str]]:
        """
        Construct message containing other agents' responses with importance weighting
        
        Args:
            visible_responses: Responses from other agents that are visible
            response_indices: Indices of the responses
            weights: Optional weights for each response (used only for ASMAD)
            
        Returns:
            List[Dict[str, str]]: Messages in the format expected by OpenAI API
        """
        
        # Standard message for intra-group debate (without weighting)
        if weights is None:
            prefix_string = "These are the recent unique opinions from other agents that differ with yours: "
            
            for i, (agent_response, agent_idx) in enumerate(zip(visible_responses, response_indices)):
                response_text = agent_response["content"] if isinstance(agent_response, dict) else agent_response
                prefix_string += f"\n\n Agent {agent_idx} response: ```{response_text}```"
            
            return [
                {
                    "role": "user",
                    "content": prefix_string + f"\n\n Using the opinions carefully as additional advice, can you provide an updated answer? Examine your solution and that other agents step by step. {self.output_format}"
                }
            ]
        
        # ASMAD-specific weighted response construction
        else:
            # Normalize weights
            if len(weights) != len(visible_responses):
                weights = np.ones(len(visible_responses))
            weights = weights / max(weights.max(), 1e-10)  # Normalize to 0-1
            
            prefix_string = "These are the solutions from other agents: "
            
            for i, (agent_response, agent_idx) in enumerate(zip(visible_responses, response_indices)):
                weight = weights[i]
                
                # Format based on weight
                if weight > 0.4:
                    prompt = "\n\n[Critical] Please carefully analyze Agent {}'s response: ```{}```"
                elif weight > 0.25:
                    prompt = "\n\n[Reference] Consider Agent {}'s perspective: ```{}```"
                elif weight > 0.1:
                    prompt = "\n\n[Background] Agent {}'s response was: ```{}```"
                else:
                    continue  # Skip if weight is too low
                    
                response_text = agent_response["content"] if isinstance(agent_response, dict) else agent_response
                prefix_string += prompt.format(agent_idx, response_text)
            
            return [
                {
                    "role": "user",
                    "content": prefix_string + f"\n\n Using the reasoning from other agents as additional advice, can you provide an updated answer? Examine your solution and that of other agents step by step. {self.output_format}"
                }
            ]
    
    def construct_message_with_group_summaries(self,
                                            group_summary: str,
                                            other_group_summaries: List[str],
                                            other_group_indices: List[int]) -> List[Dict[str, str]]:
        """
        Construct message containing summaries from other groups
        
        Args:
            group_summary: Summary from the current group
            other_group_summaries: Summaries from other groups
            other_group_indices: Indices of the other groups
            
        Returns:
            List[Dict[str, str]]: Messages in the format expected by OpenAI API
        """
        prefix_string = "These are the recent opinions from all groups: "
        prefix_string += f"Your group response: ```{group_summary}```"
        prefix_string += "Other groups responses:"
        
        for summary, group_idx in zip(other_group_summaries, other_group_indices):
            if summary:
                prefix_string += f"\n\n[Group {group_idx+1}]: ```{summary}```"
        
        return [
            {
                "role": "user",
                "content": prefix_string + f"\n\n Using the reasoning from all groups as additional advice, can you give an updated answer? Examine your solution and that all groups step by step. {self.output_format}"
            }
        ]
    
    def construct_message_with_mask(self, 
                                 agent_contexts_other: List[List[Dict[str, str]]], 
                                 msg_idx: int,
                                 agent_mask: np.ndarray, 
                                 agent_indices: List[int]) -> List[Dict[str, str]]:
        """
        Construct message based on visibility mask (for ASMAD and similar methods)
        
        Args:
            agent_contexts_other: Contexts from other agents
            msg_idx: Index of the message to extract
            agent_mask: Boolean mask indicating which agents are visible
            agent_indices: Indices of the other agents
            
        Returns:
            List[Dict[str, str]]: Messages in the format expected by OpenAI API
        """
        all_masked = not np.any(agent_mask)
        
        # Use introspection if no agents are visible
        if len(agent_contexts_other) == 0 or all_masked:
            return self.construct_reflection_message()
        
        prefix_string = "These are the recent unique opinions from other agents that differ with yours: "
        
        for agent_idx, agent_context in enumerate(agent_contexts_other):
            if agent_mask[agent_idx]:  # If current agent is visible
                if msg_idx < len(agent_context):
                    agent_response = agent_context[msg_idx]["content"]
                    response = f"\n\n Agent {agent_indices[agent_idx]} response: ```{agent_response}```"
                    prefix_string += response
        
        return [
            {
                "role": "user",
                "content": prefix_string + f"\n\n Using the opinions carefully as additional advice, can you provide an updated answer? Examine your solution and that other agents step by step. {self.output_format}"
            }
        ]