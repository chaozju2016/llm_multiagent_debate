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

SYSTEM_MESSAGE = "Explain your answer, putting the answer in the form (X) at the end of your response"


def generate_answer(answer_context, client):
    try:
        completion = client.create_chat_completion(
                messages=answer_context,
                max_tokens=2048,
                temperature=0,
                )
    except Exception as e:
        print("retrying due to an error:",e)
        time.sleep(20)
        return generate_answer(answer_context, client)

    return completion


def sigmoid_weights(sim_matrix):
    """非线性权重转换"""
    steepness = 8  # 控制曲线陡峭程度
    midpoint = 0.6  # 拐点位置
    return 1 / (1 + np.exp(-steepness*(sim_matrix - midpoint)))


def construct_message_with_weights(agent_contexts_other, idx, sim_weights, agent_indices):
    """基于相似度权重构建prompt
    
    Args:
        agent_contexts_other: 其他agent的上下文
        idx: 当前消息索引
        sim_weights: 相似度权重数组
        agent_indices: agent索引列表
    """
    if len(agent_contexts_other) == 0:
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
            weight = sim_weights[agent_idx]
            agent_response = agent[idx]["content"]
            
            # 根据权重调整prompt的强调程度
            if weight > 0.7:
                prompt = "\n\n[Important] Please carefully consider Agent {}'s response: ```{}```"
            elif weight > 0.3:
                prompt = "\n\n Agent {}'s response for reference: ```{}```"
            else:
                prompt = "\n\n For background, Agent {}'s response was: ```{}```"
                
            prefix_string += prompt.format(agent_indices[agent_idx], agent_response)

        messages = [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            },
            {
                "role": "user",
                "content": prefix_string + "\n\n Based on the above responses with their indicated importance, can you provide an updated answer? Examine your solution and that of other agents step by step."
            },
        ]

    return messages 

def parse_answer(input_str):
    matches = re.findall(r'[\(\s]+([A-D])[\)\s]+',input_str)
    solution=None
    for match_str in matches[::-1]:
        solution = match_str.upper()
        if solution:
            break
    return solution

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='math')
    parser.add_argument('-a', '--agent', type=int, default=3, help='Agent number (default: 3)')
    parser.add_argument('-p', '--port', type=int, default=8080, help='Port number (default: 8080)')
    parser.add_argument('-r', '--ratio', type=float, default=1.0, help='Ratio value (default: 1.0)')
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
    # ports = [args.port+i for i in range(1,1+agents)]
    ports = [args.port+1,args.port+2,args.port+3,args.port+1,args.port+2,args.port+3]
    print(ports)
    
    debate_round = args.debate_rounds
    
    if args.debug:
        print(f'similarity_visible_range:{similarity_visible_range}')
    np.random.seed(4125)
    random.seed(3154)
    visibility_ratio=args.ratio

    llama_client = [LlamaClient(base_url=f'http://127.0.0.1:{port}') for port in ports]
    
    embedding_model = SentenceTransformer(
        model_name_or_path = "../nomic-ai/nomic-embed-text-v1", trust_remote_code=True,
        device = 'cuda'
        )

    evaluation_round = args.eval_rounds

    tasks = glob("cais/data/test/*.csv")
    dfs = [pd.read_csv(task) for task in tasks]
    results = {}

    for eval_round in tqdm(range(evaluation_round), total=evaluation_round, position=0, desc='Eval', leave=False, colour='#82b0d2', unit='traj'):
        df = random.choice(dfs)
        ix = random.randint(0,len(df)-1)
        question = df.iloc[ix, 0]
        a = df.iloc[ix, 1]
        b = df.iloc[ix, 2]
        c = df.iloc[ix, 3]
        d = df.iloc[ix, 4]
        answer = df.iloc[ix, 5]
        question = "{}:A) {}, B) {}, C) {}, D) {}.".format(question, a, b, c, d)
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
                 #计算权重矩阵 (使用1-相似度作为基础,以降低高相似度的影响)
                weights_matrix = np.zeros_like(sim_matrix)
                for i in range(agents):
                    for j in range(agents):
                        if i != j:
                            # 基础权重 + 差异度的影响
                            weights_matrix[i,j] = 0.2 + 0.6 * (1 - sigmoid_weights(sim_matrix[i,j]))
    
                # 确保权重在合理范围内
                weights_matrix = np.clip(weights_matrix, 0.1, 0.9)
                if args.debug:
                    print(f'mask_matrix:\n{weights_matrix}')
            else:
                # 第一轮时使用均匀权重
                weights_matrix = np.ones((agents, agents)) * 0.5
                np.fill_diagonal(weights_matrix, 1)  # 自己的权重设为0
            info_of_round["weights_matrix"] = weights_matrix
                
                
            for i, agent_context in enumerate(agent_contexts):
                if args.debug:
                    print(f'agent {i}')

                #print(f'agent_context:{len(agent_context)}')
                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    messages = construct_message_with_weights(
                            agent_contexts_other,
                            # 3 stands for [sys, user, assistant]
                            3*round - 1,
                            weights_matrix[i],
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
