import json
import numpy as np
import time
import pickle
from tqdm import tqdm
import sys

sys.path.append("..")
from client import LlamaClient
import re

client = LlamaClient()
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


def generate_answer(answer_context):
    try:
        completion = client.create_chat_completion(
            messages=answer_context,
            max_tokens=140,
        )
    except:
        print("retrying due to an error......")
        time.sleep(20)
        return generate_answer(answer_context)

    return completion


def construct_message(agents, question, idx):

    # Use introspection in the case in which there are no other agents.
    if len(agents) == 0:
        return {"role": "user", "content": "Can you verify that your answer is correct. Please reiterate your answer, making sure to state your answer at the end of the response."}

    prefix_string = "These are the recent/updated opinions from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

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

    agents = 3
    rounds = 3
    np.random.seed(0)

    evaluation_round = 1
    scores = []

    generated_description = {}

    for round in tqdm(range(evaluation_round)):
        a, b, c, d, e, f = np.random.randint(0, 30, size=6)

        answer = a + b * c + d - e * f
        agent_contexts = [[{"role": "user", "content": """What is the result of {}+{}*{}+{}-{}*{}? Make sure to state your answer at the end of the response.""".format(a, b, c, d, e, f)}] for agent in range(agents)]

        content = agent_contexts[0][0]['content']
        question_prompt = "We seek to find the result of {}+{}*{}+{}-{}*{}?".format(a, b, c, d, e, f)

        text_answers = {}
        for round in range(rounds):
            print(f'debate round{round}')
            for i, agent_context in enumerate(agent_contexts):
                print(f'agent {i}')

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question_prompt, 2*round - 1)
                    agent_context.append(message)

#                    print("message: ", message['content'])
#                else:
#                    print('init message:',agent_context[0]['content'])

                completion = generate_answer(agent_context)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
#               print(assistant_message['content'])

            text_answers[round] = []

            for agent_context in agent_contexts:
                text_answer = string =  agent_context[-1]['content']
                text_answer = text_answer.replace(",", ".")
                text_answer = parse_answer(text_answer)

                if text_answer is None:
                    continue

                text_answers[round].append(text_answer)
            print(f'text_answers: {text_answers}')
        print(f'answer: {answer}')
        continue

        generated_description[(a, b, c, d, e, f)] = (
            agent_contexts,
            text_answer,
            answer,
        )

        try:
            text_answer = most_frequent(text_answers)
            print("text answer: ", text_answer, "answer: ", answer)
            if text_answer == answer:
                scores.append(1)
            else:
                scores.append(0)
        except:
            continue

        print("performance:", np.mean(scores), np.std(scores) / (len(scores) ** 0.5))

    # pickle.dump(generated_description, open("math_agents{}_rounds{}.p".format(agents, rounds), "wb"))
    #print(answer)
    
    # print(agent_context)
