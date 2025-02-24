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
import random
from typing import List, Dict, Any
import pyarrow.parquet as pq
sys.path.append('math/modeling')
from dataset.util import last_boxed_only, last_boxed_only_string

sys.path.append("..")
from sentence_transformers import SentenceTransformer
from client import LlamaClient
from mask import MaskConfig, MaskGenerator
from similarity import compute_similarities

SYSTEM_MESSAGE = """
You are a helpful AI assistant that always responds in valid JSON format.
ONLY return a JSON object, no other text.

Required JSON structure:
{
    "summary_of_others": string,  // summary of other agents' viewpoints, leave it blank if no other agent is visible
    "independent_analysis": string,  // your own step-by-step solution
    "answer": string  // your final answer with numerical value or latex expression, wrapped in $\\boxed{}$
}
DO NOT include any text before or after the JSON.
DO NOT include ```json or ``` markers.
DO NOT include any explanations outside the JSON.
"""

class SimpleMATHDataset:
    """A simplified wrapper for MATH dataset that only loads problems without tokenization"""
    def __init__(self, data_dir: str, category: str = "algebra"):
        self.data_dir = data_dir
        self.category = category
        self.data = None
        self._load_data()
        
    def _load_data(self):
        """Load parquet files from the specified category"""
        category_path = os.path.join(self.data_dir, self.category)
        parquet_files = glob(os.path.join(category_path, "*.parquet"))
        
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {category_path}")
            
        import pyarrow.parquet as pq
        
        # Load the first parquet file (assuming all data is in one file as per the pattern test-00000-of-00001.parquet)
        self.data = pq.read_table(parquet_files[0]).to_pandas()
        print(f"Loaded {len(self.data)} problems from {parquet_files[0]}")
        
    def get_random_problem(self) -> Dict[str, str]:
        """Get a random problem from the dataset"""
        idx = random.randint(0, len(self.data) - 1)
        row = self.data.iloc[idx]
        
        # The parquet file should have 'problem' and 'solution' columns
        question = row['problem']
        solution = row['solution']
        
        # Extract final boxed answer from solution
        final_answer = last_boxed_only_string(solution)
        if not final_answer:
            final_answer = solution  # 如果无法提取boxed答案，使用完整solution
            
        return {
            "question": question,
            "solution": solution,
            "final_answer": final_answer
        }


class AdaptiveDebateController:
    def __init__(self, num_agents: int, total_rounds: int,
                 base_threshold: float = 0.0,
                 min_weight: float = 0.1,
                 max_weight: float = 0.9):
        self.num_agents = num_agents
        self.total_rounds = total_rounds
        self.base_threshold = base_threshold
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.history = []

    def compute_trust_threshold(self, current_round: int) -> float:
        """动态计算信任阈值"""
        progress = current_round / self.total_rounds
        return self.base_threshold + 0.3 * progress

    def compute_weights(self, sim_matrix: np.ndarray, current_round: int, 
                       answer_history: List[List[str]]) -> np.ndarray:
        """计算权重矩阵"""
        weights = np.zeros_like(sim_matrix)
        threshold = self.compute_trust_threshold(current_round)
        stability_scores = self.compute_stability(answer_history)
        # 计算自信权重
        progress = current_round / self.total_rounds
        base_self_weight = 0.3 + 0.5 * (progress)  # 从0.3线性增长到0.8

        
        for i in range(self.num_agents):
            self_weight = base_self_weight * (1 + 0.2 * stability_scores[i])
            weights[i,i] = np.clip(self_weight, 0.3, 0.8)
            for j in range(self.num_agents):
                if i == j:
                    continue
                    
                if sim_matrix[i,j] > threshold:
                    # 在early rounds鼓励差异性
                    if current_round < self.total_rounds // 2:
                        base_weight = self.max_weight * (1 - sim_matrix[i,j])
                    # 在later rounds才考虑共识
                    else:
                        base_weight = self.max_weight * sim_matrix[i,j]
                else:
                    base_weight = self.min_weight
                
                weights[i,j] = np.clip(
                    base_weight * (1 + 0.2 * stability_scores[j]),
                    self.min_weight,
                    self.max_weight
                )

        row_sums = weights.sum(axis=1, keepdims=True)
        normalized_weights = weights / row_sums
        
        return normalized_weights

    def compute_stability(self, answer_history: List[List[str]]) -> np.ndarray:
        """计算答案稳定性分数"""
        if len(answer_history) < 2:
            return np.zeros(self.num_agents)
            
        stability_scores = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            changes = sum(1 for t in range(1, len(answer_history))
                        if answer_history[t][i] != answer_history[t-1][i])
            stability_scores[i] = 1 - (changes / len(answer_history))
        return stability_scores

    def construct_prompt(self, weight: float, agent_response: str, agent_idx: int) -> str:
        is_visible = np.random.random() < weight
        if not is_visible:
            return "\n\n"
            
        """根据权重构建prompt"""
        if weight > 0.4:
            return f"\n\n[Critical] Please carefully analyze Agent {agent_idx}'s response: ```{agent_response}```"
        elif weight > 0.25:
            return f"\n\n[Reference] Consider Agent {agent_idx}'s perspective: ```{agent_response}```"
        elif weight > 0.1:
            return f"\n\n[Background] Agent {agent_idx}'s response was: ```{agent_response}```"
        else:
            return f"\n\n"

def generate_answer(messages: List[Dict[str, str]], client) -> Dict[str, Any]:
    """生成回答"""
    try:
        completion = client.create_chat_completion(
            messages=messages,
            max_tokens=1024,
            temperature=0.0,
        )
    except Exception as e:
        print("Retrying due to error:", e)
        time.sleep(20)
        return generate_answer(messages, client)
    return completion

def construct_debate_message(agent_contexts_other: List[List[Dict[str, str]]], 
                           idx: int,
                           weights: np.ndarray,
                           agent_indices: List[int],
                           controller: AdaptiveDebateController) -> List[Dict[str, str]]:
    """构建辩论消息"""
    if len(agent_contexts_other) == 0:
        return [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": "Can you double check that your answer is correct."}
        ]
    
    prefix_string = "These are the solutions from other num_agents: "
    for agent_idx, agent in enumerate(agent_contexts_other):
        weight = weights[agent_idx]
        agent_response = agent[idx]["content"]
        prefix_string += controller.construct_prompt(weight, agent_response, agent_indices[agent_idx])

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prefix_string + "\n\nBased on the above responses with their indicated importance, can you provide an updated answer? Examine all solutions step by step."}
    ]

def parse_answer(input_str: str) -> str:
    """解析答案"""
    pattern = r'\$\\boxed{(.*?)}\$'
    match = re.findall(pattern, input_str)
    if match:
        return match[-1]
    else:
        pattern = r'(-?\d*\.?\d+)'
        match = re.findall(pattern, input_str)
        if match:
            return match[-1]
    return None

def clean_repeat_suffix(text: str) -> str:
    """清理重复的后缀"""
    n = len(text)
    for length in range(n//2, 10, -1):
        suffix = text[-length:]
        pos = text.find(suffix)
        if pos != -1 and pos + length != n:
            return text[:pos + length]
    return text

def validate_json_response(response):
    try:
        result = json.loads(response)
        result["answer"] = "{}".format(parse_answer(str(result["answer"])) or '')
        result["independent_analysis"] += " The final answer is {}.\n".format(result["answer"])
    except json.JSONDecodeError:
        result = {
            "summary_of_others": "",
            "independent_analysis": response,
            "answer": "{}".format(parse_answer(response) or '')
        }
    return result

def find_majority(nums):
    counts = {}
    for num in nums:
        counts[num] = counts.get(num, 0) + 1
    return max(counts, key=counts.get)
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='math')
    parser.add_argument('-a', '--agent', type=int, default=3, help='Agent number (default: 3)')
    parser.add_argument('-p', '--port', type=int, default=8080, help='Port number (default: 8080)')
    parser.add_argument('-r', '--ratio', type=float, default=1.0, help='Ratio value (default: 1.0)')
    parser.add_argument('-er', '--eval_rounds', type=int, default=30, help='Evaluation rounds (default: 30)')
    parser.add_argument('-dr', '--debate_rounds', type=int, default=3, help='Debate rounds (default: 3)')
    parser.add_argument('-q', '--question_range', type=int, default=30, help='Question range (default: 30)')
    parser.add_argument('-c', '--category', type=str, default='algebra', help='MATH category (default: algebra)')
    parser.add_argument('-D','--debug', type=bool, default=False, help='Debug ouput (default: False)')
    parser.add_argument('-ld','--log_dir', type=str, default='multi', help='Log directory (default: multi)')
    args = parser.parse_args()

    debate_round = args.debate_rounds
    experiment_name = args.log_dir
    os.makedirs(f'progress_data/{experiment_name}',exist_ok=True)
    os.makedirs(f'data/{experiment_name}',exist_ok=True)
    num_agents = args.agent
    # ports = [args.port+i for i in range(1,1+num_agents)]
    port_overleaf = 0
    ports = [args.port+1,args.port+2+port_overleaf,args.port+3,args.port+1+port_overleaf,args.port+2,args.port+3+port_overleaf]
    print(ports)

    controller = AdaptiveDebateController(num_agents=num_agents, total_rounds=debate_round)
    
    np.random.seed(4125)
    random.seed(3154)

    llama_client = [LlamaClient(base_url=f'http://127.0.0.1:{port}') for port in ports]
    
    embedding_model = SentenceTransformer(
        model_name_or_path = "../nomic-ai/nomic-embed-text-v1", trust_remote_code=True,
        device = 'cuda'
        )

    evaluation_round = args.eval_rounds
    
    math_dataset = SimpleMATHDataset("hendrycks_math", category=args.category)

    results = {}

    for eval_round in tqdm(range(evaluation_round), total=evaluation_round, position=0, desc='Eval', leave=False, colour='#82b0d2', unit='traj'):
        problem = math_dataset.get_random_problem()
        question = problem["question"]
        solution = problem["solution"]
        answer = problem["final_answer"]
        if args.debug:
            print(f'question: {question}, answer: {answer}')
        
        agent_contexts = [
                [
                    {
                        "role": "system",
                        "content": SYSTEM_MESSAGE
                    },
                    {
                        "role": "user", 
                        "content": f"Can you answer the following question as accurately as possible? {question}?"
                    }
                ] for agent in range(num_agents)]

        content = agent_contexts[0][0]['content']

        results[eval_round] = {
                'question':question,
                'answer':answer,
                'states':[],
            }

        info_of_round = {}
        change_caculated = [0] * num_agents
        text_answer_this_round = [None] * num_agents
        text_answer_last_round = [None] * num_agents
        answer_history = []
    
        for round in tqdm(range(debate_round), total=debate_round, position=1, desc='Debate', leave=False, colour='#8ecfc9', unit='round'):
            info_of_round = {
                "round": round,
                "text_answer": [],
                "context": [],
                'usage': []
            }
            
    
            weights_matrix = np.ones((num_agents, num_agents)) * 0.0
            if round != 0:
                sim_matrix = results[eval_round]['states'][-1]['sim_matrix']
                weights_matrix = controller.compute_weights(
                    sim_matrix=sim_matrix,
                    current_round=round,
                    answer_history=answer_history
                )
            
            info_of_round["weights_matrix"] = weights_matrix
    
            for i, agent_context in enumerate(agent_contexts):
                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    messages = construct_debate_message(
                        # agent_contexts_other,
                        agent_contexts,
                        3*round - 1,
                        weights_matrix[i],
                        # agent_indices=list(range(i)) + list(range(i+1,num_agents)),
                        agent_indices=list(range(num_agents)),
                        controller=controller
                    )
                    
                    agent_context.extend(messages)
    
                completion = generate_answer(agent_context[-2:], llama_client[i])
                # assistant_message = completion["choices"][0]["message"]["content"]
                # assistant_message = clean_repeat_suffix(assistant_message)
                response = completion["choices"][0]["message"]["content"]
                json_response = validate_json_response(response)

                assistant_message = json_response['independent_analysis']
                
                agent_context.append({"role": "assistant", "content": assistant_message})
                agent_contexts[i] = agent_context
    
                # text_answer = parse_answer(assistant_message)
                text_answer = parse_answer(json_response['answer'])
                info_of_round["context"].append(assistant_message)
                info_of_round["text_answer"].append(text_answer)
                info_of_round['usage'].append(completion['usage'])
    
            text_answer_last_round = text_answer_this_round
            text_answer_this_round = info_of_round["text_answer"]
            answer_history.append(copy.deepcopy(text_answer_this_round))
    
            context = ['search_document: ' + s for s in info_of_round['context']]
            embeddings = embedding_model.encode(context, normalize_embeddings=True)
            sim_matrix = np.inner(embeddings, embeddings)

            answers = text_answer_this_round
            answer_sim = np.zeros((len(answers), len(answers)))
            for i in range(len(answers)):
                for j in range(len(answers)):
                    answer_sim[i,j] = 1.0 if answers[i] == answers[j] else 0.0

            alpha = 0.5  # reasoning的权重
            combined_sim = alpha * sim_matrix + (1-alpha) * answer_sim
            
            info_of_round["sim_matrix"] = combined_sim
            info_of_round["majority_answer"] = find_majority(text_answer_this_round)
            
    
            if args.debug:
                print('\n\n\n')
                # print(f'Weights matrix:\n', np.around(weights_matrix,2))
                # print(f'context:\n', info_of_round["context"])
                print(f'Round {round} answers:', text_answer_this_round, "majority: {}".format(info_of_round["majority_answer"]))
                # print(f'similarity matrix:\n', np.around(sim_matrix,2))
    
            results[eval_round]['states'].append(copy.deepcopy(info_of_round))

            if (eval_round+1) % max(1,int(evaluation_round // 10)) == 0:
                pickle.dump(results,open("progress_data/{}/multi_mmlu_results_er{}_agents{}_dr{}_ratio{}_range{}_{}.p".format(experiment_name, evaluation_round, num_agents, debate_round,0.0,args.question_range,eval_round),'wb'))
    pickle.dump(results,open("data/{}/multi_mmlu_results_er{}_agents{}_dr{}_ratio{}_range{}.p".format(experiment_name, evaluation_round, num_agents, debate_round,0.0,args.question_range),'wb'))
