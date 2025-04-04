import pickle
import os
import argparse
import numpy as np
import re
from collections import Counter

def extract_parameters(filename):
        # 定义正则表达式模式
        pattern = r'er(\d+)_agents(\d+)_dr(\d+)_ratio([\d.]+)_range(\d+)'
        # 使用正则表达式匹配
        match = re.search(pattern, filename)
        
        if match:
            # 提取所有参数
            params = {
                    'er': int(match.group(1)),
                    'agents': int(match.group(2)),
                    'dr': int(match.group(3)),
                    'ratio': float(match.group(4)),
                    'range': int(match.group(5))
                    }
            return params
        return None

def find_most_frequent(arr):
    flat_arr = arr.flatten()
    valid_values = [x for x in flat_arr if x is not None and not np.isnan(x)]
    if not valid_values:
        return None
    counter = Counter(valid_values)
    most_frequent_value, _count = counter.most_common(1)[0]
    return most_frequent_value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='math')
    parser.add_argument('-d', '--data_dir', type=str, default='.', help='data dir')

    args=parser.parse_args()
    
    data_dir = args.data_dir

    data_files = os.listdir(data_dir)
    # only keep pickle files
    data_files = [data_file for data_file in data_files if data_file.endswith('.p')]
    print(f'find {len(data_files)} file(s)')

    accuracy = {}

    for data_file in data_files:
        params = extract_parameters(data_file)
        er = params['er']
        agents = params['agents']
        dr = params['dr']

        data = pickle.load(open(os.path.join(data_dir,data_file),'rb'))
        
        gt_answer = np.zeros(er, dtype=str)
        agent_answer = np.zeros((er, dr, agents), dtype=str)

        for traj_id, traj in data.items():
            gt_answer[traj_id] = traj['answer']
            for debate_context in traj['states']:
                agent_answer[traj_id][debate_context['round']] = debate_context['text_answer']
                # agent_answer[traj_id][debate_context['round']] = debate_context['majority_answer']

        #accuracy[data_file] = np.asarray(np.equal(gt_answer[:,None,None], agent_answer).mean(axis=-1) >=0.5).mean(axis=0)
        accuracy[data_file] = np.asarray(np.equal(gt_answer[:,None,None], agent_answer).mean(axis=-1)).mean(axis=0)
        
    
    for k in sorted(accuracy.keys()):
        print(f'{k}: {accuracy[k]}')
