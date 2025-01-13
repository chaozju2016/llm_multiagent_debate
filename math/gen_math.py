import copy
import json
import numpy as np
import time
import pickle
from tqdm import tqdm
import re
import argparse
import sys

sys.path.append("..")
from client import LlamaClient
from mask import MaskConfig, MaskGenerator

def parse_bullets(sentence):
    bullets_preprocess = sentence.split("\n")
    bullets = []

    for bullet in bullets_preprocess:
        try:
            idx = bullet.find(next(filter(str.isalpha, bullet)))
        except:
            continue

        bullet = bullet[idx:]

        if len(bullet) != 0:
            bullets.append(bullet)

    return bullets


def generate_answer(answer_context, client):
    try:
        completion = client.create_chat_completion(
                messages=answer_context,
                max_tokens=400,
                temperature=0,
                )
    except:
        print("retrying due to an error......")
        time.sleep(20)
        return generate_answer(answer_context, client)

    return completion


def construct_message_with_mask(agents, question, idx, agent_mask):

    # Use introspection in the case in which there are no other agents.
    if len(agents) == 0:
        return {"role": "user", "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."}
    # 检查是否所有其他agents都被mask
    all_masked = not np.any(agent_mask)
    if all_masked:
        prefix_string = ("All other agents' responses are masked in this round. "
                "Please rethink the problem independently and carefully. "
                "You might want to: \n"
                "1. Double check your previous calculation\n"
                "2. Consider if there are alternative approaches\n"
                "3. Be extra careful about the order of operations\n\n")
    else:
        prefix_string = "These are the recent/updated opinions from other agents: "
        for agent_idx, agent in enumerate(agents):
            if agent_mask[agent_idx]:  # 如果当前agent可见
                agent_response = agent[idx]["content"]
                response = "\n\n One agent response: ```{}```".format(agent_response)
                prefix_string = prefix_string + response
            else:
                prefix_string = prefix_string + "\n\n [This agent's response is masked]"

    prefix_string = prefix_string + "\n\n Use these opinions carefully as additional advice, can you provide an updated answer? Make sure to state your answer at the end of the response.".format(question)

    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}


def parse_answer(sentence):
    int_matches_formatted = re.findall(r"My answer is (-?\d+(?:\.\d+)?)", sentence)
    float_matches_formatted = re.findall(r"My confidence is (-?\d+\.\d+)", sentence)
    int_matches = re.findall(r'(?<![\d.])-?\b\d+\b(?!\.\d|%)', sentence)
    float_matches = re.findall(r'-?\d+\.\d+', sentence)
    
    if int_matches_formatted:
        # print("answer-textformat-get!")
        answer = int(int_matches_formatted[-1])
    elif int_matches:
        # print("answer-integer-get!")
        answer = int(int_matches[-1])
    else:
        # print("Cannot find answer")
        answer = None
    
    if float_matches_formatted:
        # print("confidence-textformat-get!")
        # print()
        confidence = float(float_matches_formatted[-1])
    elif float_matches:
        # print("confidence-float-get!")
        confidence = float(float_matches[-1])
    else:
        # print("Cannot find confidence")
        confidence = None

    
    # 提取最后一个浮点数和整数
    # answer = int(int_matches[-1]) if int_matches else None
    # confidence = float(float_matches[-1]) if float_matches else None
    return answer,confidence

def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        current_frequency = List.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i

    return num


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='math')
    parser.add_argument('-p', '--port', type=int, default=8080, help='Port number (default: 8080)')
    parser.add_argument('-r', '--ratio', type=float, default=1.0, help='Ratio value (default: 1.0)')
    parser.add_argument('-q', '--question_range', type=int, default=30, help='Question range (default: 30)')
    args = parser.parse_args()

    agents = 3
    rounds = 10
    np.random.seed(0)
    visibility_ratio=args.ratio
    mask_config = MaskConfig(
            num_agents=agents,
            visibility_ratio=visibility_ratio,  # 可以根据需要调整
            strategy='fixed_ratio'
            )
    llama_client = LlamaClient(base_url='http://127.0.0.1:{}'.format(args.port))

    evaluation_round = 1

    scores = {r:[] for r in range(rounds)}
    results = []


    for eval_round in tqdm(range(evaluation_round)):
        a, b, c, d, e, f = np.random.randint(0, args.question_range, size=6)

        answer = a + b * c + d - e * f
        agent_contexts = [[{"role": "user", "content": """What is the result of {}+{}*{}+{}-{}*{}? Make sure to state your answer and your confidence at the end of the response following format strictly.You should state your answer following this format: My answer is *your answer*,For example, you must say in this format:My answer is 100.You must follow this format to state your confidence is a float between 0 and 1 with dot,for example, you can say, my confidence is 0.8  """.format(a, b, c, d, e, f)}] for agent in range(agents)]

        content = agent_contexts[0][0]['content']
        question_prompt = "We seek to find the result of {}+{}*{}+{}-{}*{}?".format(a, b, c, d, e, f)

        # text_answers = {}
        # text_confidences = {}
        
        # text_answers_acc = {}

        info_of_round = {}
        change_caculated = [0] * agents
        text_answer_this_round = [None] * agents
        text_answer_last_round = [None] * agents
        for round in range(rounds):
            # print(f'debate round{round}')
            info_of_round["round"] = round

            mask_matrix = MaskGenerator.generate(mask_config)
            #print(f'mask:{mask_matrix}')
            info_of_round["text_answer"] = []
            info_of_round["confidence"] = []
            info_of_round["answer_change"] = []
            info_of_round["context"] = []
            info_of_round["mask_matrix"] = mask_matrix

            for i, agent_context in enumerate(agent_contexts):
                # print(f'agent {i}')

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message_with_mask(
                            agent_contexts_other,
                            question_prompt,
                            2*round - 1,
                            mask_matrix[i],
                            )
                    agent_context.append(message)

#                    print("message: ", message['content'])
#                else:
#                    print('init message:',agent_context[0]['content'])

                completion = generate_answer(agent_context, llama_client)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                # print(assistant_message['content'])

            # text_answers[round] = []
            # text_confidences[round] = []

            for agent_context in agent_contexts:
                text_answer = string =  agent_context[-1]['content']
                text_answer = text_answer.replace(",", ".")
                # print("context:",text_answer)
                info_of_round["context"].append(text_answer)
                
                text_answer,text_confidence = parse_answer(text_answer)

                # if text_answer is None:
                #     continue
                # print("text_answer:",text_answer)
                

                
                info_of_round["text_answer"].append(text_answer)
                info_of_round["confidence"].append(text_confidence)
            
            text_answer_last_round = text_answer_this_round
            text_answer_this_round = info_of_round["text_answer"]
            
            
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
            
            
            # print("info_of_round:",info_of_round)
            results.append(copy.deepcopy(info_of_round))
            # print("results:",results)
                # text_answers[round].append(text_answer)
                # text_confidences[round].append(text_confidence)
            
       
    #print(results)
    pickle.dump(results,open("math_results_agents{}_rounds{}_ratio{}_range{}.p".format(agents, rounds,visibility_ratio,args.question_range),'wb'))