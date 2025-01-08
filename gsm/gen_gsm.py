import json
import numpy as np
import random
import sys

sys.path.append("..")
from client import LlamaClient
import tqdm

client = LlamaClient()
def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct. Please reiterate your answer, with your final answer a single numerical number, in the form \\boxed{{answer}}."}

    prefix_string = "These are the solutions to the problem from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n The original math problem is {}. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response.""".format(question)
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

if __name__ == "__main__":
    agents = 5
    rounds = 5
    random.seed(0)

    generated_description = {}

    questions = read_jsonl(
        "C:/Users/cwang/Documents/llm_multiagent_debate/gsm/grade-school-math/grade_school_math/data/mytest.jsonl"
    )
    random.shuffle(questions)

    for data in questions[:1]:
        question = data['question']
        answer = data['answer']

        print(question)
        print(answer)

        agent_contexts = [[{"role": "user", "content": """Can you solve the following math problem? {} Explain your reasoning. Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response. """.format(question)}] for agent in range(agents)]

        for round in tqdm.tqdm(range(rounds)):
            for i, agent_context in enumerate(agent_contexts):
                print("agent", i)

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, question, 2*round - 1)
                    agent_context.append(message)

                completion = client.create_chat_completion(
                    messages=agent_context,
                    max_tokens=2048,
                )

                assistant_message = construct_assistant_message(completion)
                print(assistant_message)
                agent_context.append(assistant_message)

        generated_description[question] = (agent_contexts, answer)

    json.dump(generated_description, open("gsm_{}_{}.json".format(agents, rounds), "w"))

    # import pdb
    # pdb.set_trace()
    print(answer)
    print(agent_context)
