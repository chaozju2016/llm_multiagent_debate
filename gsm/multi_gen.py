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
sys.path.append("..")
from sentence_transformers import SentenceTransformer
from client import LlamaClient
from mask import MaskConfig, MaskGenerator
from similarity import compute_similarities

SYSTEM_MESSAGE = """
Answer the following multiple choice question.
Your response MUST:
1. First explain your reasoning
2. End with EXACTLY ONE line starting with "$\\boxed{?}$" where the ? filled by a single numerical number
Do not include any other answer format or letter options in your explanation
"""

def generate_answer(answer_context, client):
    try:
        completion = client.create_chat_completion(
                messages=answer_context,
                # max_tokens=1024,
                temperature=0.2,
                )
    except Exception as e:
        print("retrying due to an error:",e)
        time.sleep(20)
        return generate_answer(answer_context, client)

    return completion


def construct_message_with_mask(agent_contexts_other, idx, agent_mask, agent_indices):

    all_masked = not np.any(agent_mask)
    # Use introspection in the case in which there are no other agents.
    if len(agent_contexts_other) == 0 or all_masked:
        messages = [
                {
                    "role": "system",
                    "content": SYSTEM_MESSAGE,
                },
                {
                    "role": "user", 
                    "content": "Can you double check that your answer is correct."
                },
            ]
    else:
        prefix_string = "These are the solutions to the problem from other agents: "
        for agent_idx, agent in enumerate(agent_contexts_other):
            if agent_mask[agent_idx]:  # 如果当前agent可见
                agent_response = agent[idx]["content"]
                response = "\n\n Agent {} response: ```{}```".format(agent_indices[agent_idx], agent_response)
                prefix_string = prefix_string + response
            else:
                prefix_string = prefix_string #+ "\n\n [This agent's response is masked]"

        messages = [
                {
                    "role": "system",
                    "content": SYSTEM_MESSAGE,
                },
                {
                    "role": "user",
                    "content": prefix_string + "\n\n Using the reasoning from other agents as additional advice, can you provide an updated answer? Eaxming your solution and that of other agents step by step. "
                },
            ]

    return messages 

def parse_answer(input_str):
    pattern = r'\$\\boxed{(-?\d*\.?\d+)}\$'
    match = re.findall(pattern, input_str)
    if match:
        return match[-1]
    return None

def clean_repeat_suffix(text):
    n = len(text)
    # 从最长可能的重复长度开始尝试
    for length in range(n//2, 10, -1):
        # 获取末尾的子串
        suffix = text[-length:]
        # 在去掉末尾这段后的文本中查找这个子串
        pos = text.find(suffix)
        
        # 如果找到了，并且正好是末尾（pos + length == len(remaining)）
        if pos != -1 and pos + length != n:
            # 找到了重复，返回重复之前的部分
            return text[:pos + length]
    return text
    
def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='math')
    parser.add_argument('-a', '--agent', type=int, default=3, help='Agent number (default: 3)')
    parser.add_argument('-p', '--port', type=int, default=8080, help='Port number (default: 8080)')
    parser.add_argument('-r', '--ratio', type=float, default=0.0, help='Ratio value (default: 1.0)')
    parser.add_argument('-er', '--eval_rounds', type=int, default=30, help='Evaluation rounds (default: 30)')
    parser.add_argument('-dr', '--debate_rounds', type=int, default=3, help='Debate rounds (default: 3)')
    parser.add_argument('-q', '--question_range', type=int, default=30, help='Question range (default: 30)')
    parser.add_argument('-D','--debug', type=bool, default=False, help='Debug ouput (default: False)')
    parser.add_argument('-ld','--log_dir', type=str, default='multi', help='Log directory (default: multi)')
    args = parser.parse_args()

    experiment_name = args.log_dir
    os.makedirs(f'progress_data/{experiment_name}',exist_ok=True)
    os.makedirs(f'data/{experiment_name}',exist_ok=True)
    agents = args.agent
    #ports = [args.port+i for i in range(1,1+agents)]
    ports = [args.port+1, args.port+2,args.port+3,args.port+1, args.port+2,args.port+3]
    print(ports)
    
    debate_round = args.debate_rounds
    # similarity
    lower_bound_init = 0.0
    lower_bound_final = args.ratio
    lower_bound_step = (lower_bound_final-lower_bound_init)/(debate_round-2)
    lower_bounds = np.append(
        np.arange(lower_bound_init, lower_bound_final, lower_bound_step),
        lower_bound_final) if lower_bound_init != lower_bound_final else np.ones(debate_round)*lower_bound_init
    upper_bound_init = 1.0
    upper_bound_final = 1.0
    upper_bound_step = (upper_bound_final-upper_bound_init)/(debate_round-2)
    upper_bounds = np.append(
        np.arange(upper_bound_init, upper_bound_final, upper_bound_step),
        upper_bound_final) if upper_bound_init != upper_bound_final else np.ones(debate_round)*upper_bound_init
    
    similarity_visible_range = list(zip(lower_bounds, upper_bounds))
    
    if args.debug:
        print(f'similarity_visible_range:{similarity_visible_range}')
    np.random.seed(4125)
    random.seed(3154)
    visibility_ratio=args.ratio
    mask_config = MaskConfig(
            num_agents=agents,
            visibility_ratio=visibility_ratio,  # 可以根据需要调整
            strategy='similarity',
            similarity_visible_range=similarity_visible_range
            )
    #llama_client = LlamaClient(base_url='http://127.0.0.1:{}'.format(args.port))
    llama_client = [LlamaClient(base_url=f'http://127.0.0.1:{port}') for port in ports]
    
    embedding_model = SentenceTransformer(
        model_name_or_path = "../nomic-ai/nomic-embed-text-v1", trust_remote_code=True,
        device = 'cuda'
        )

    evaluation_round = args.eval_rounds
    questions = read_jsonl(
            "grade-school-math/grade_school_math/data/test.jsonl"
        )[:evaluation_round]
    random.shuffle(questions)
    
    results = {}

    for eval_round in tqdm(range(evaluation_round), total=evaluation_round, position=0, desc='Eval', leave=False, colour='#82b0d2', unit='traj'):
        data = questions[eval_round]
        question = data['question']
        answer = data['answer']
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
                ] for agent in range(agents)]

        content = agent_contexts[0][0]['content']

        results[eval_round] = {
                'question':question,
                'answer':answer,
                'states':[],
            }

        info_of_round = {}
        change_caculated = [0] * agents
        text_answer_this_round = [None] * agents
        text_answer_last_round = [None] * agents
        for round in tqdm(range(debate_round), total=debate_round, position=1, desc='Debate', leave=False, colour='#8ecfc9', unit='round'):
            if args.debug:
                print(f'debate round{round}')
                
            info_of_round["round"] = round
            info_of_round["text_answer"] = []
            #info_of_round["confidence"] = []
            info_of_round["answer_change"] = []
            info_of_round["context"] = []
            info_of_round['usage'] = []

            if round != 0:
                # @todo: fix this
                sim_matrix= results[eval_round]['states'][-1]['sim_matrix']
                if args.debug:
                    print(f'sim_matrix:\n{sim_matrix}')
                mask_matrix = MaskGenerator.generate(
                    config=mask_config, 
                    sim_matrix=sim_matrix,
                    range_index=round-1)
                if args.debug:
                    print(f'mask_matrix:\n{mask_matrix}')
            else:
                mask_matrix = np.eye(agents, dtype=bool)
            info_of_round["mask_matrix"] = mask_matrix
                
                
            for i, agent_context in enumerate(agent_contexts):
                if args.debug:
                    print(f'agent {i}')

                #print(f'agent_context:{len(agent_context)}')
                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    messages = construct_message_with_mask(
                            agent_contexts_other,
                            # 3 stands for [sys, user, assistant]
                            3*round - 1,
                            mask_matrix[i],
                            agent_indices=list(range(i)) + list(range(i+1,agents))
                            )
                    agent_context = agent_context + messages
                    #agent_context.append(messages)
                #print(f'agent_context:{json.dumps(agent_context[-3:],indent=2)}')
                
                #[-2:] stands for latest [sys, user]
                completion = generate_answer(agent_context[-3:], llama_client[i])
                assistant_message = completion["choices"][0]["message"]["content"]
                assistant_message=clean_repeat_suffix(assistant_message)
                if args.debug:
                    print(assistant_message)
                agent_context.append({"role": "assistant", "content": assistant_message})
                agent_contexts[i] = agent_context

                text_answer = parse_answer(assistant_message)
                if args.debug:
                    print(f'answer {text_answer}, conf: {text_confidence}')
                info_of_round["context"].append(assistant_message)
                info_of_round["text_answer"].append(text_answer)
                #info_of_round["confidence"].append(text_confidence)
                
                info_of_round['usage'].append(completion['usage'])
            
            text_answer_last_round = text_answer_this_round
            text_answer_this_round = info_of_round["text_answer"]
            
            context = ['search_document: ' + s for s in info_of_round['context']]
            embeddings = embedding_model.encode(context, normalize_embeddings=True)
            sim_matrix = compute_similarities(embeddings=embeddings, return_format='matrix')
            info_of_round["sim_matrix"] = sim_matrix
            
            if args.debug:
                print(f'answer {text_answer_this_round}, conf: {info_of_round["confidence"]}')
            
            if round == 0:
                info_of_round["answer_change"] = [0] * agents
            else:
                change = [0] * agents
                for agent in range(agents):
                    if text_answer_this_round[agent] == text_answer_last_round[agent]:
                        change[agent] = 0
                    else:
                        change[agent] = 1
                change_caculated = [x + y for x, y in zip(change_caculated, change)]
                info_of_round["answer_change"] = change_caculated
            
            
            results[eval_round]['states'].append(copy.deepcopy(info_of_round))
        if args.debug:
            print(f'question: {question}, answer: {answer}')
        if (eval_round+1) % max(1,int(evaluation_round // 10)) == 0:
            pickle.dump(results,open("progress_data/{}/multi_mmlu_results_er{}_agents{}_dr{}_ratio{}_range{}_{}.p".format(experiment_name, evaluation_round, agents, debate_round,visibility_ratio,args.question_range,eval_round),'wb'))
    pickle.dump(results,open("data/{}/multi_mmlu_results_er{}_agents{}_dr{}_ratio{}_range{}.p".format(experiment_name, evaluation_round, agents, debate_round,visibility_ratio,args.question_range),'wb'))
