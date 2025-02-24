import pickle
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='math')
parser.add_argument('-d', '--data_dir', type=str, default='.', help='data dir')

args=parser.parse_args()

data_dir = args.data_dir

data_files = os.listdir(data_dir)
data_files = [data_file for data_file in data_files if '.p' in data_file]
print(f'find {len(data_files)} file(s)')
    



for data_file in data_files:
    
    # 读取 pickle 文件
    data = pickle.load(open(os.path.join(data_dir,data_file), "rb"))
    round_num = len(data[0]['states'])
    agent_num = len(data[0]['states'][0]['usage'])

    # round 行 agent 列
    round_agent_matrix = np.zeros((round_num, agent_num))
    
    total_tokens = 0
    for key, value in data.items():
        for round_index,round_state in enumerate(value['states']):
            for agent_index,agent_usage_unit in enumerate(round_state["usage"]):
                total_tokens += agent_usage_unit['total_tokens']
                round_agent_matrix[round_index][agent_index] += agent_usage_unit['total_tokens']
    print("total_tokens:",total_tokens)
    print("round_agent_matrix:",round_agent_matrix)