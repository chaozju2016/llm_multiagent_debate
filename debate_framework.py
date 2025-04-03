"""
Main module for multi-agent debate framework with support for various debate strategies.
"""

import numpy as np
import time
import copy
import re
from typing import List, Dict, Any, Optional, Tuple, Callable
from tqdm import tqdm

from topology import TopologyManager, TopologyConfig
from decision_maker import DecisionMaker, OpinionDynamicsDecisionMaker
from group_manager import GroupManager
from message_constructor import MessageConstructor
from similarity import compute_similarities

class DebateFramework:
    """Framework for multi-agent debate with support for various strategies"""
    
    def __init__(self, 
                 num_agents: int,
                 debate_strategy: str = "fully_connected",
                 client_factory: Callable = None,
                 embedding_model = None,
                 system_message: str = "Welcome to the debate!",
                 output_format: str = "Put your final answer in parentheses at the end of your response.",
                 topology_config = None,
                 debug: bool = False,
                 asmad_equality_weight: float = 0.5):
        """
        Initialize debate framework
        
        Args:
            num_agents: Number of agents
            debate_strategy: Strategy for debate topology
            client_factory: Factory function to create LLM clients
            embedding_model: Model for embedding text
            system_message: System message for all agents
            output_format: Format instruction for agent responses
            topology_config: Configuration for debate topology
            debug: Whether to print debug information
        """
        self.num_agents = num_agents
        self.debate_strategy = debate_strategy
        self.client_factory = client_factory
        self.embedding_model = embedding_model
        self.system_message = system_message
        self.output_format = output_format
        self.debug = debug
        self.topology_config = topology_config
        self.asmad_equality_weight = asmad_equality_weight
        
        # Initialize LLM clients
        self.clients = []
        if client_factory:
            for i in range(num_agents):
                self.clients.append(client_factory(i))
        
        # Initialize components based on debate strategy
        self._init_components()
        
    def _init_components(self):
        """Initialize debate components based on strategy"""
        # Initialize topology config if not provided
        if self.topology_config is None:
            from topology import TopologyConfig
            
            # Initialize topology config
            if self.debate_strategy == "fully_connected":
                self.topology_config = TopologyConfig(
                    num_agents=self.num_agents,
                    topology_type="fully_connected"
                )
            elif self.debate_strategy == "ring":
                self.topology_config = TopologyConfig(
                    num_agents=self.num_agents,
                    topology_type="ring"
                )
            elif self.debate_strategy == "star":
                self.topology_config = TopologyConfig(
                    num_agents=self.num_agents,
                    topology_type="star"
                )
            elif self.debate_strategy == "group":
                # Default to roughly equal groups
                agents_per_group = max(2, self.num_agents // 3)
                group_structure = []
                for i in range(0, self.num_agents, agents_per_group):
                    end = min(i + agents_per_group, self.num_agents)
                    group_structure.append(list(range(i, end)))
                    
                self.topology_config = TopologyConfig(
                    num_agents=self.num_agents,
                    topology_type="group",
                    group_structure=group_structure
                )
            elif self.debate_strategy == "s2mad":
                # Same grouping as for 'group'
                agents_per_group = max(2, self.num_agents // 3)
                group_structure = []
                for i in range(0, self.num_agents, agents_per_group):
                    end = min(i + agents_per_group, self.num_agents)
                    group_structure.append(list(range(i, end)))
                    
                self.topology_config = TopologyConfig(
                    num_agents=self.num_agents,
                    topology_type="s2mad",
                    group_structure=group_structure,
                    decision_threshold=0.65
                )
            elif self.debate_strategy == "asmad":
                self.topology_config = TopologyConfig(
                    num_agents=self.num_agents,
                    topology_type="asmad",
                    trust_radius_init=0.0,
                    trust_radius_final=0.8
                )
            else:
                raise ValueError(f"Unknown debate strategy: {self.debate_strategy}")
        
        # Import necessary components
        from topology import TopologyManager
        from decision_maker import DecisionMaker, OpinionDynamicsDecisionMaker
            
        # Initialize topology manager
        self.topology_manager = TopologyManager(self.topology_config)
        
        # Initialize decision maker
        if self.debate_strategy == "s2mad":
            self.decision_maker = DecisionMaker(self.num_agents, strategy="s2mad")
        elif self.debate_strategy == "asmad":
            self.decision_maker = OpinionDynamicsDecisionMaker(
                self.num_agents,
                self.topology_config.trust_radius_init,
                self.topology_config.trust_radius_final,
                self.topology_config.similarity_outlier_threshold
            )
        else:
            self.decision_maker = DecisionMaker(self.num_agents)
            
        # Initialize group manager if needed
        if self.debate_strategy in ["group", "s2mad"]:
            from group_manager import GroupManager
            self.group_manager = GroupManager(self.topology_config.group_structure)
        else:
            self.group_manager = None
            
        # Initialize message constructor
        from message_constructor import MessageConstructor
        self.message_constructor = MessageConstructor(self.system_message, self.output_format)
        
    def run_debate(self, question: str, debate_rounds: int) -> Dict[str, Any]:
        """
        Run multi-agent debate
        
        Args:
            question: Question to debate
            debate_rounds: Number of debate rounds
            
        Returns:
            Dict[str, Any]: Debate results
        """
        # Set total rounds for components that need it
        self.topology_manager.set_total_rounds(debate_rounds)
        
        # Initialize agent contexts
        agent_contexts = [
            self.message_constructor.construct_initial_message(question) for _ in range(self.num_agents)
        ]
        
        # Initialize results
        results = {
            'question': question,
            'states': []
        }
        
        # Track answers for stability calculation
        text_answer_this_round = [None] * self.num_agents
        text_answer_last_round = [None] * self.num_agents
        
        # Main debate loop
        for round in tqdm(range(debate_rounds), desc="Debate Rounds", disable=not self.debug):
            if self.debug:
                print(f"Debate round {round}")
                
            # Initialize round info
            info_of_round = {
                "round": round,
                "text_answer": [],
                "context": [],
                "usage": []
            }
            
            # For group-based strategies, determine phase
            if self.debate_strategy in ["group", "s2mad"]:
                if round == 0:
                    self.group_manager.set_phase("intra")
                else:
                    # Alternate between intra and inter group phases
                    cycle_length = self.topology_config.intra_group_rounds + self.topology_config.inter_group_rounds
                    round_in_cycle = (round - 1) % cycle_length
                    if round_in_cycle < self.topology_config.intra_group_rounds:
                        self.group_manager.set_phase("intra")
                    else:
                        self.group_manager.set_phase("inter")
            
            # Generate mask or weights for this round
            if round > 0:
                # Get similarity matrix from previous round
                sim_matrix = results['states'][-1]['sim_matrix']
                
                # Update topology manager round
                self.topology_manager.next_round()
                
                if self.debate_strategy == "asmad":
                    # Generate mask based on similarity and trust radius
                    mask_matrix = self.topology_manager.generate_mask(sim_matrix)
                    info_of_round["mask_matrix"] = mask_matrix
                    
                    # Compute opinion dynamics weights
                    weights_matrix = self.decision_maker.compute_influence_weights(
                        sim_matrix, round, debate_rounds
                    )
                    info_of_round["weights_matrix"] = weights_matrix
                elif self.debate_strategy == "s2mad":
                    # Generate mask with decision mechanism
                    mask_matrix = self.topology_manager.generate_mask(sim_matrix)
                    info_of_round["mask_matrix"] = mask_matrix
                else:
                    # Generate mask based on topology
                    mask_matrix = self.topology_manager.generate_mask()
                    info_of_round["mask_matrix"] = mask_matrix
            else:
                # First round - diagonal matrix (agents only see themselves)
                mask_matrix = np.eye(self.num_agents, dtype=bool)
                info_of_round["mask_matrix"] = mask_matrix
                
            # Group summaries for inter-group communication
            if self.debate_strategy in ["group", "s2mad"] and round > 0:
                if self.group_manager.current_phase == "inter":
                    # Generate group summaries at the end of intra-group phase
                    previous_responses = results['states'][-1]['context']
                    group_summaries = self.group_manager.generate_all_summaries(previous_responses, self.clients[0])
                    info_of_round["group_summaries"] = group_summaries
                    
                    # Calculate embeddings for group summaries for S2-MAD
                    if self.debate_strategy == "s2mad" and self.embedding_model is not None:
                        summary_embeddings = {}
                        for group_idx, summary in group_summaries.items():
                            if summary:
                                embedding = self.embedding_model.encode([summary], normalize_embeddings=True)[0]
                                summary_embeddings[group_idx] = embedding
                        info_of_round["group_summary_embeddings"] = summary_embeddings
            
            # Agent debate loop
            for i, agent_context in enumerate(agent_contexts):
                if self.debug:
                    print(f"Agent {i}")
                
                # Skip if initial round (already initialized)
                if round > 0:
                    # For group-based strategies
                    if self.debate_strategy in ["group", "s2mad"]:
                        if self.group_manager.current_phase == "intra":
                            # Intra-group debate
                            group_idx = self.group_manager.get_group_for_agent(i)
                            group_members = self.group_manager.get_group_members(group_idx)
                            
                            # Get responses from other group members
                            other_members = [idx for idx in group_members if idx != i]
                            visible_members = [idx for idx in other_members if mask_matrix[i, idx]]
                            
                            if visible_members:
                                # Only include visible members
                                visible_contexts = [agent_contexts[idx] for idx in visible_members]
                                # Construct message with visible responses
                                messages = self.message_constructor.construct_message_with_visible_responses(
                                    [ctx[2*round] for ctx in visible_contexts],
                                    visible_members,
                                    weights_matrix[i, visible_members] if self.debate_strategy == "asmad" else None
                                )
                            else:
                                # Get agent's own response
                                agent_response = agent_contexts[i][2*round]
                                # No visible members, use reflection
                                messages = self.message_constructor.construct_reflection_message(agent_response)
                        else:
                            # Inter-group debate
                            group_idx = self.group_manager.get_group_for_agent(i)
                            # Get summary from the current group
                            group_summary = self.group_manager.get_group_summary(group_idx)
                            # Get summaries from other groups
                            other_groups = [idx for idx in range(len(self.group_manager.group_structure)) if idx != group_idx]
                            
                            # Construct message with group summaries
                            other_group_summaries = [self.group_manager.get_group_summary(idx) for idx in other_groups]
                            messages = self.message_constructor.construct_message_with_group_summaries(
                                group_summary=group_summary,
                                other_group_summaries=other_group_summaries, 
                                other_group_indices=other_groups
                            )
                    else:
                        # Non-group-based strategies: ASMAD, S-MAD, etc.
                        
                        # Get responses from other agents
                        other_agents = [idx for idx in list(range(self.num_agents)) if idx != i]
                        visible_agents = [idx for idx in other_agents if mask_matrix[i, idx]]
                        
                        if visible_agents:
                            # Only include visible members
                            visible_contexts = [agent_contexts[idx] for idx in visible_agents]
                            
                            # Construct message with visible responses
                            messages = self.message_constructor.construct_message_with_visible_responses(
                                [ctx[2*round] for ctx in visible_contexts],
                                visible_agents,
                                weights_matrix[i, visible_agents] if self.debate_strategy == "asmad" else None
                            )
                        else:
                            # Get agent's own response
                            agent_response = agent_contexts[i][2*round]
                            # No visible members, use reflection
                            messages = self.message_constructor.construct_reflection_message(agent_response)
                    
                    # Add messages to agent context
                    for message in messages:
                        agent_context.append(message)
                
                # Generate response
                completion = self.generate_answer(agent_context[-2:], self.clients[i])
                assistant_message = completion["choices"][0]["message"]["content"]
                
                # Clean up repeated text
                assistant_message = self.clean_repeat_suffix(assistant_message)
                
                # Add response to agent context (use full message)
                agent_context.append({"role": "assistant", "content": assistant_message})
                agent_contexts[i] = agent_context
                
                # Extract answer
                text_answer = self.parse_answer(assistant_message)
                
                # Store information
                info_of_round["context"].append(assistant_message)
                info_of_round["text_answer"].append(text_answer)
                info_of_round["usage"].append(completion.get("usage", {}))
            
            # Update answer tracking for stability calculation
            text_answer_last_round = text_answer_this_round
            text_answer_this_round = info_of_round["text_answer"]
            
            # Update decision maker with new answers
            self.decision_maker.update_answer_history(text_answer_this_round)
            
            # Compute similarity matrix for next round
            context = ['search_document: ' + s for s in info_of_round['context']]
            embeddings = self.embedding_model.encode(context, normalize_embeddings=True)
            sim_matrix = compute_similarities(embeddings=embeddings, return_format='matrix')
            # ASMAD calculates similarity matrix together with answer consistency
            if self.debate_strategy == "asmad":
                # calculate equality matrix
                equality_matrix = np.equal.outer(text_answer_this_round, text_answer_this_round).astype(float)
                # apply equality_matrix to sim_matrix with weights
                sim_matrix = equality_matrix * self.asmad_equality_weight + sim_matrix * (1 - self.asmad_equality_weight)
                
            info_of_round["sim_matrix"] = sim_matrix
            
            if self.debug:
                print(f"Answers: {text_answer_this_round}")
            
            # Calculate majority answer
            majority_answer = self.decision_maker.find_majority_answer(text_answer_this_round)
            info_of_round["majority_answer"] = majority_answer
            
            # Store round information
            results['states'].append(copy.deepcopy(info_of_round))
        
        # Store final answer in results
        results['final_answer'] = results['states'][-1]['majority_answer']
        
        return results
        
    def generate_answer(self, messages, client):
        """Generate response using specified client"""
        try:
            completion = client.create_chat_completion(
                messages=messages,
                max_tokens=1024,
                temperature=0.0,
            )
        except Exception as e:
            print(f"Retrying due to error: {e}")
            time.sleep(20)
            return self.generate_answer(messages, client)
        
        return completion
    
    def parse_answer(self, input_str):
        """Extract answer from response text"""
        # Look for pattern like "(A)" or "(B)" etc. - standard format
        matches = re.findall(r'[\(\s]+([A-D])[\)\s]+', input_str)
        if matches:
            return matches[-1].upper()
            
        # Try boxed format like \boxed{A}
        matches = re.findall(r'\\boxed{([A-D])}', input_str)
        if matches:
            return matches[-1].upper()
            
        # Try simple letter match
        matches = re.findall(r'\b([A-D])\b', input_str)
        if matches:
            return matches[-1].upper()
                
        return None
    
    def clean_repeat_suffix(self, text):
        """Clean up repeated text in response"""
        n = len(text)
        for length in range(n//2, 10, -1):
            suffix = text[-length:]
            pos = text.find(suffix)
            if pos != -1 and pos + length != n:
                return text[:pos + length]
        return text