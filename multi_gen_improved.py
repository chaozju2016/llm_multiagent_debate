"""
Script for running multi-agent debate with different strategies.
"""

import copy
import json
import numpy as np
import time
import pickle
from tqdm import tqdm
import re
import argparse
import sys
import os
from glob import glob
import pandas as pd
import random

from topology import TopologyConfig
sys.path.append("..")

from sentence_transformers import SentenceTransformer
from client import LlamaClient
from debate_framework import DebateFramework

# Constants
SYSTEM_MESSAGE = """Welcome to the debate! You are a seasoned debater with expertise in succinctly and persuasively expressing your viewpoints. 
You will be assigned to debate groups, where you will engage in discussions with fellow participants. The outcomes of 
each group's deliberations will be shared among all members. It is crucial for you to leverage this information effectively 
in order to critically analyze the question at hand and ultimately arrive at the correct answer. Best of luck!"""

# Output format prompts by task type
OUTPUT_FORMATS = {
    "arithmetic": "Make sure to state your answer at the end of the response.",
    "gsm8k": "Your final answer should be a single numerical number, in the form \\boxed{answer}, at the end of your response.",
    "mmlu": "Put your final choice in parentheses at the end of your response.",
    "math": "Put your final answer in the form \\boxed{answer}, at the end of your response.",
    "gpqa": "Put your final answer in the form \\The correct answer is (insert answer here)"
}

def create_client_factory(ports, same_model=False):
    """
    Create a factory function to generate LLM clients
    
    Args:
        ports: List of port numbers for client connections
        same_model: Whether to use the same model for all agents
        
    Returns:
        Function that creates LLM client for an agent
    """
    def factory(agent_idx):
        if same_model:
            # Use first port for all clients
            port = ports[0]
        else:
            # Use port based on agent index
            port = ports[agent_idx % len(ports)]
        
        return LlamaClient(base_url=f'http://127.0.0.1:{port}')
    
    return factory

def configure_topology(args):
    """
    Configure topology based on command line arguments
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        TopologyConfig: Configuration for debate topology
    """
    # Create group structure for group-based topologies
    group_structure = None
    if args.topology in ['group', 's2mad']:
        num_groups = max(1, args.agent // args.group_size)
        group_structure = []
        for i in range(num_groups):
            start = i * args.group_size
            end = min(start + args.group_size, args.agent)
            if end > start:  # Ensure non-empty groups
                group_structure.append(list(range(start, end)))
    
    # Create topology configuration
    config = TopologyConfig(
        num_agents=args.agent,
        topology_type=args.topology
    )
    
    # Set specific configuration based on topology type
    if args.topology in ['group', 's2mad']:
        config.group_structure = group_structure
        config.intra_group_rounds = args.intra_rounds
        config.inter_group_rounds = args.inter_rounds
    
    if args.topology == 's2mad':
        config.decision_threshold = args.decision_threshold
    
    if args.topology == 'asmad':
        config.trust_radius_init = args.radius_init
        config.trust_radius_final = args.radius_final
        config.similarity_outlier_threshold = args.outlier_threshold
    
    return config

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Multi-agent debate with different strategies')
    
    # Basic configuration
    parser.add_argument('-a', '--agent', type=int, default=6, help='Number of agents (default: 6)')
    parser.add_argument('-p', '--port', type=int, default=8080, help='Base port number (default: 8080)')
    parser.add_argument('-n', '--eval_rounds', type=int, default=30, help='Evaluation rounds (default: 30)')
    parser.add_argument('-dr', '--debate_rounds', type=int, default=5, help='Debate rounds (default: 5)')
    parser.add_argument('-q', '--question_range', type=int, default=30, help='Question range (default: 30)')
    
    # Debate strategy
    parser.add_argument('-t', '--topology', type=str, default='fully_connected', 
                       choices=['fully_connected', 'ring', 'star', 'group', 's2mad', 'asmad'],
                       help='Topology strategy (default: fully_connected)')
    
    # Task type
    parser.add_argument('-tt', '--task_type', type=str, default='mmlu',
                       choices=['arithmetic', 'gsm8k', 'mmlu', 'math', 'gpqa'],
                       help='Task type for output format (default: mmlu)')
    
    # Group configuration
    parser.add_argument('-g', '--group_size', type=int, default=2, help='Size of each group (for group-based topologies)')
    parser.add_argument('-ir', '--intra_rounds', type=int, default=2, help='Intra-group debate rounds (default: 2)')
    parser.add_argument('-er', '--inter_rounds', type=int, default=1, help='Inter-group debate rounds (default: 1)')
    
    # ASMAD configuration
    parser.add_argument('-ri', '--radius_init', type=float, default=0.0, help='Initial trust radius (default: 0.0)')
    parser.add_argument('-rf', '--radius_final', type=float, default=0.8, help='Final trust radius (default: 0.8)')
    parser.add_argument('-ot', '--outlier_threshold', type=float, default=0.5, help='Outlier threshold (default: 0.5)')
    parser.add_argument('-ew', '--equality_weight', type=float, default=0.5, help='Equality Weight (default: 0.5)')
    
    # S2-MAD configuration
    parser.add_argument('-dt', '--decision_threshold', type=float, default=0.65, help='Decision threshold (default: 0.65)')
    
    # Model configuration
    parser.add_argument('-sm', '--same_model', action='store_true', help='Use same model for all agents')
    
    # Output and debugging
    parser.add_argument('-D','--debug', action='store_true', help='Enable debug output')
    parser.add_argument('-ld','--log_dir', type=str, default='multi', help='Log directory (default: multi)')
    
    return parser.parse_args()

def main():
    """Main function for running multi-agent debate"""
    # Parse command line arguments
    args = parse_args()
    
    # Setup directories
    experiment_name = args.log_dir
    os.makedirs(f'progress_data/{experiment_name}', exist_ok=True)
    os.makedirs(f'data/{experiment_name}', exist_ok=True)
    
    # Setup ports
    base_port = args.port
    # Define a fixed set of ports for different models
    ports = [base_port, base_port+1, base_port+2, base_port, base_port+1, base_port+2]
    print(f"Using ports: {ports}")
    
    # Set random seeds for reproducibility
    np.random.seed(4125)
    random.seed(3154)
    
    # Create client factory
    client_factory = create_client_factory(ports, args.same_model)
    
    # Initialize embedding model
    embedding_model = SentenceTransformer(
        model_name_or_path="../nomic-ai/nomic-embed-text-v1", 
        trust_remote_code=True,
        device='cuda'
    )
    
    # Configure topology
    topology_config = configure_topology(args)
    
    # Create debate framework
    debate_framework = DebateFramework(
        num_agents=args.agent,
        debate_strategy=args.topology,
        client_factory=client_factory,
        embedding_model=embedding_model,
        system_message=SYSTEM_MESSAGE,
        output_format=OUTPUT_FORMATS[args.task_type],
        topology_config=topology_config,
        debug=args.debug,
        asmad_equality_weight=args.equality_weight,
    )
    
    # Load datasets
    tasks = glob("cais/data/test/*.csv")
    dfs = [pd.read_csv(task) for task in tasks]
    results = {}
    
    # Run evaluation
    for eval_round in tqdm(range(args.eval_rounds), total=args.eval_rounds, 
                          desc='Eval', colour='#82b0d2', unit='traj'):
        # Select random question
        df = random.choice(dfs)
        ix = random.randint(0, len(df)-1)
        
        # Extract question and answer
        question = df.iloc[ix, 0]
        a = df.iloc[ix, 1]
        b = df.iloc[ix, 2]
        c = df.iloc[ix, 3]
        d = df.iloc[ix, 4]
        answer = df.iloc[ix, 5]
        
        # Format question
        formatted_question = f"{question}:A) {a}, B) {b}, C) {c}, D) {d}."
        
        if args.debug:
            print(f'Question: {formatted_question}')
            print(f'Ground truth: {answer}')
        
        # Run debate
        debate_results = debate_framework.run_debate(formatted_question, args.debate_rounds)
        
        # Store results
        results[eval_round] = {
            'question': formatted_question,
            'answer': answer,
            'states': debate_results['states'],
            'final_answer': debate_results['final_answer']
        }
        
        # Periodically save results
        if (eval_round+1) % max(1, args.eval_rounds // 10) == 0:
            file_path = f"progress_data/{experiment_name}/results_er{args.eval_rounds}_agents{args.agent}_dr{args.debate_rounds}_topo{args.topology}_range{args.question_range}_{eval_round}.p"
            with open(file_path, 'wb') as f:
                pickle.dump(results, f)
     
    # Calculate accuracy
    correct = 0
    total = 0
    for eval_round, result in results.items():
        if result['final_answer'] == result['answer']:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Final accuracy: {accuracy:.2f} ({correct}/{total})")
    
    # Calculate token usage
    total_tokens = 0
    for eval_round, result in results.items():
        for state in result['states']:
            for usage in state.get('usage', []):
                total_tokens += usage.get('total_tokens', 0)
    
    avg_tokens = total_tokens / total if total > 0 else 0
    print(f"Average token usage: {avg_tokens:.2f} tokens per question")
    
    # Save final results
    file_path = f"data/{experiment_name}/results_er{args.eval_rounds}_agents{args.agent}_dr{args.debate_rounds}_topo{args.topology}_ri{args.radius_init}_rf{args.radius_final}_ot{args.outlier_threshold}_ew{args.equality_weight}_acc{accuracy:.2f}_token{avg_tokens:.2f}.p"
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)
        
    print(f"Results saved to {file_path}")

if __name__ == "__main__":
    main()

