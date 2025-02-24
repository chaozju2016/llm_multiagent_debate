import pandas as pd
import pickle
from typing import List, Dict

# keyword definitions
PROBLEM_KEY = 'question'
ANSWER_KEY = 'answer'
STATES_KEY = 'states'
ROUND_KEY = 'round'
TEXT_ANSWER_KEY = 'text_answer'
CONFIDENCE_KEY = 'confidence'
USAGE_KEY = 'usage'
MASK_MATRIX_KEY = 'mask_matrix'
SIM_MATRIX_KEY = 'sim_matrix'

PROBLEM_INDEX_KEY = 'problem_index'


def process_file(file_name: str) -> pd.DataFrame:
    """Process a file and return a pandas DataFrame.

    Args:
        file_name (str): The name of the file to process.

    Returns:
        pd.DataFrame: The processed data.
    
    """
    with open(file_name, 'rb') as f:
        data = pickle.load(f) # type: Dict[int, Dict]
    
    # data format:
    # data (dict[int, dict]): problem_index[problem_data]
    # 
    # problem_data:
    #   "question" (str): problem context
    #   "answer" (str): answer context
    #   "states" (list[dict]): list of states
    #
    # states:
    #   "round" (int): round index
    #   "text_answer" (list[str]): text answer of each agent, say [agent_0, agent_1, ...]
    #   "confidence" (list[float]): confidence of each agent, say [0.9, 0.8, ...]
    #   "answer_change" (list[int]): turns of answer changes of each agent, say [0, 1, ...]
    #   "context" (list[str]): output context of each agent in this round, say [context_0, context_1, ...]
    #   "usage" (list[dict]): usage of each agent in this round, say [usage_agent_0, usage_agent_1, ...]
    #   "mask_matrix" (np.ndarray): mask matrix of each agent in this round, say [mask_matrix_agent_0, mask_matrix_agent_1, ...]
    #   "sim_matrix" (np.ndarray): similarity matrix of each agent in this round, say [sim_matrix_agent_0, sim_matrix_agent_1, ...]
    #
    # usage_agent:
    #   "completion_tokens" (int): completion tokens
    #   "prompt_tokens" (int): prompt tokens
    #   "total_tokens" (int): total tokens


    # organize the data into a pandas DataFrame
    data_df = pd.DataFrame()
    records = []
    for problem_index, problem_data in data.items():
        for state in problem_data[STATES_KEY]: 
            # state type: Dict[str, Any]
            records.append({
                PROBLEM_INDEX_KEY: problem_index,
                PROBLEM_KEY: problem_data.get(PROBLEM_KEY, ''),
                ANSWER_KEY: problem_data.get(ANSWER_KEY, ''),
                ROUND_KEY: state.get(ROUND_KEY, ''),
                TEXT_ANSWER_KEY: state.get(TEXT_ANSWER_KEY, ''),
                CONFIDENCE_KEY: state.get(CONFIDENCE_KEY, ''),
                USAGE_KEY: state.get(USAGE_KEY, ''),
                MASK_MATRIX_KEY: state.get(MASK_MATRIX_KEY, ''),
                SIM_MATRIX_KEY: state.get(SIM_MATRIX_KEY, ''),
            })
    data_df = pd.concat([data_df, pd.DataFrame(records)], ignore_index=True)
    
    return data_df


def print_columns(data_df: pd.DataFrame) -> None:
    """Print the column names of the DataFrame.
    
    Args:
        data_df (pd.DataFrame): The DataFrame whose columns to print.
    """
    print("DataFrame columns:")
    for i, col in enumerate(data_df.columns, 1):
        print(f"{i}. {col}")


if __name__ == '__main__':
    file_name = "/Users/tanghuaze/llm_multiagent_debate/data/multi_mmlu_results_er100_agents6_dr5_ratio0.0_range30.p"
    data_df = process_file(file_name)
    print_columns(data_df)
