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
    numbers = re.findall(r'\d+', sentence)
    return numbers[-1] if numbers else None


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
    rounds = 8
    np.random.seed(0)
    visibility_ratio=args.ratio
    mask_config = MaskConfig(
            num_agents=agents,
            visibility_ratio=visibility_ratio,  # 可以根据需要调整
            strategy='fixed_ratio'
            )
    llama_client = LlamaClient(base_url='http://127.0.0.1:{}'.format(args.port))

    evaluation_round = 100

    scores = {r:[] for r in range(rounds)}
    results = []


    for eval_round in tqdm(range(evaluation_round)):
        a, b, c, d, e, f = np.random.randint(0, args.question_range, size=6)

        answer = a + b * c + d - e * f
        agent_contexts = [[{"role": "user", "content": """What is the result of {}+{}*{}+{}-{}*{}? Make sure to state your answer at the end of the response.""".format(a, b, c, d, e, f)}] for agent in range(agents)]

        content = agent_contexts[0][0]['content']
        question_prompt = "We seek to find the result of {}+{}*{}+{}-{}*{}?".format(a, b, c, d, e, f)

        text_answers = {}
        text_answers_acc = {}
        for round in range(rounds):
            #print(f'debate round{round}')

            mask_matrix = MaskGenerator.generate(mask_config)
            #print(f'mask:{mask_matrix}')

            for i, agent_context in enumerate(agent_contexts):
                #print(f'agent {i}')

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
                #print(assistant_message['content'])

            text_answers[round] = []

            for agent_context in agent_contexts:
                text_answer = string =  agent_context[-1]['content']
                text_answer = text_answer.replace(",", ".")
                text_answer = parse_answer(text_answer)

                if text_answer is None:
                    continue

                text_answers[round].append(text_answer)
            #print(f'text_answers: {text_answers}')
        #print(f'answer: {answer}')
        results.append({'eval_round':eval_round,'question':'{}+{}*{}+{}-{}*{}'.format(a, b, c, d, e, f),'answer':answer,'text_answers':text_answers})
    pickle.dump(results,open("math_results_agents{}_rounds{}_ratio{}_range{}.p".format(agents, rounds,visibility_ratio,args.question_range),'wb'))
