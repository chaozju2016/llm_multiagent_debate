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

sys.path.append("..")
from sentence_transformers import SentenceTransformer
from client import LlamaClient
from mask import MaskConfig, MaskGenerator
from similarity import compute_similarities

SYSTEM_MESSAGE = (
        "Make sure to state your answer and your confidence at the end of the response following format strictly."
        "You should state your answer following this format:\n"
        "My answer is *your answer*\n"
        "For example, you must say in this format:My answer is 100.\n"
        "You must follow this format to state your confidence is a float in [0,1] with at most two digits:\n"
        "My confidence is *your confidence*\n"
        "For example, you can say, my confidence is 0.85.\n"
        )



def generate_answer(answer_context, client):
    try:
        completion = client.create_chat_completion(
                messages=answer_context,
                max_tokens=200,
                temperature=0,
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
                    "content": "Can you verify that your answer is correct? Please reiterate your answer."
                },
            ]
    else:
        prefix_string = "These are the recent/updated opinions from other agents: "
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
                    "content": prefix_string + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer?"
                },
            ]

    return messages 




def parse_answer(sentence):
    #print(f'parsing\n{sentence}')
    int_matches_formatted = re.findall(r"My answer is \**\s*(-?\d+(?:\.\d+)?)\s*\**", sentence)
    float_matches_formatted = re.findall(r"My confidence is \**\s*(-?\d+\.\d+)\s*\**", sentence)
    
    int_matches = re.findall(r'(?<![\d.])-?\b\d+\b(?!\.\d|%)', sentence)
    float_matches = re.findall(r'-?\d+\.\d+', sentence)
    
    if int_matches_formatted:
        # print("answer-textformat-get!")
        answer = int( eval(int_matches_formatted[-1]))
    elif int_matches:
        # print("answer-integer-get!")
        answer = int( eval(int_matches[-1]))
    else:
        # print("Cannot find answer")
        answer = None
    
    if float_matches_formatted:
        # print("confidence-textformat-get!")
        # print()
        confidence = float(eval(float_matches_formatted[-1]))
    elif float_matches:
        # print("confidence-float-get!")
        confidence = float(float_matches[-1])
    else:
        # print("Cannot find confidence")
        confidence = None

    
    return answer,confidence

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
    parser.add_argument('-er', '--eval_rounds', type=int, default=1, help='Evaluation rounds (default: 30)')
    parser.add_argument('-dr', '--debate_rounds', type=int, default=3, help='Debate rounds (default: 3)')
    parser.add_argument('-q', '--question_range', type=int, default=30, help='Question range (default: 30)')
    parser.add_argument('-D','--debug', type=bool, default=False, help='Debug ouput (default: False)')
    parser.add_argument('-ld','--log_dir', type=str, default='multi', help='Log directory (default: multi)')
    args = parser.parse_args()

    experiment_name = args.log_dir
    os.makedirs(f'progress_data/{experiment_name}',exist_ok=True)
    os.makedirs(f'data/{experiment_name}',exist_ok=True)
    agents = args.agent
    ports = [args.port+i for i in range(1,1+agents)]
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
    # np.random.seed(4125)
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

    results = {}

    for eval_round in tqdm(range(evaluation_round), total=evaluation_round, position=0, desc='Eval', leave=False, colour='#82b0d2', unit='traj'):
        a, b, c, d, e, f = np.random.randint(0, args.question_range, size=6)
        answer = a + b * c + d - e * f
        question = '{}+{}*{}+{}-{}*{}'.format(a, b, c, d, e, f)
        
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
                        "content": f"What is the result of {question}?"
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
            info_of_round["confidence"] = []
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

                text_answer,text_confidence = parse_answer(assistant_message)
                if args.debug:
                    print(f'answer {text_answer}, conf: {text_confidence}')
                info_of_round["context"].append(assistant_message)
                info_of_round["text_answer"].append(text_answer)
                info_of_round["confidence"].append(text_confidence)
                
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
            pickle.dump(results,open("progress_data/{}/multi_math_results_er{}_agents{}_dr{}_ratio{}_range{}_{}.p".format(experiment_name, evaluation_round, agents, debate_round,visibility_ratio,args.question_range,eval_round),'wb'))
    pickle.dump(results,open("data/{}/multi_math_results_er{}_agents{}_dr{}_ratio{}_range{}.p".format(experiment_name, evaluation_round, agents, debate_round,visibility_ratio,args.question_range),'wb'))
