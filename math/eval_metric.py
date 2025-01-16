import os
import pickle
import numpy as np
from typing import Dict
from eval_data import extract_parameters

def match_answer_each_agent_each_round(gt_answer: np.ndarray, agent_answer: np.ndarray) -> np.ndarray:
    """Match for each agent at each round.

    Args:
        gt_answer (np.ndarray): groud truth answers, shape should be (question_num,)
        agent_answer (np.ndarray): answer from each agent from each round, shape shoule be (question_num, round_num, agent_num)
    
    Returns:
        np.ndarray: whether each answer is correct or not (1: correct, 0: wrong), shape be (question_num, round_num, agent_num)
    
    """
    accuracy = np.asarray(np.equal(gt_answer[:,None,None], agent_answer), dtype=float)
    return accuracy

def get_confusion_matrix_for_agent_each_round(gt_answer: np.ndarray, agent_answer: np.ndarray) -> Dict[str, np.ndarray]:
    """Get the confusion matrix across agents at each round.

    Args:
        gt_answer (np.ndarray): groud truth answers, shape should be (question_num,)
        agent_answer (np.ndarray): answer from each agent from each round, shape shoule be (question_num, round_num, agent_num)
    
    Returns:
        np.ndarray: confusion matrix, key be "agent{i}_vs_agent{j}", value be a 3d np.ndarray with shape (round_num, 2, 2) as (2, 2) for each confusion matrix
            confusion matrix be
                          col 0                 col 1
            row_0 (agent_i T, agent_j T), (agent_i T, agent_j F)
            row_1 (agent_i F, agent_j T), (agent_i F, agent_j F)
    """
    _, round_num, agent_num = agent_answer.shape

    accuracy = match_answer_each_agent_each_round(gt_answer, agent_answer)

    confusion_matrix_all = {}

    for i in range(agent_num):
        for j in range(i+1, agent_num):
            key_name = f"agent{i}_vs_agent{j}"
            confusion_matrix = np.zeros(shape=(round_num, 2, 2))
            for r in range(round_num):
                # Get the accuracy for agent i and agent j in round r
                agent_i_accuracy = accuracy[:, r, i]
                agent_j_accuracy = accuracy[:, r, j]

                # Calculate the confusion matrix for this round
                confusion_matrix[r, 0, 0] = np.sum((agent_i_accuracy == 1) & (agent_j_accuracy == 1))  # Both correct
                confusion_matrix[r, 0, 1] = np.sum((agent_i_accuracy == 1) & (agent_j_accuracy == 0))  # i correct, j wrong
                confusion_matrix[r, 1, 0] = np.sum((agent_i_accuracy == 0) & (agent_j_accuracy == 1))  # i wrong, j correct
                confusion_matrix[r, 1, 1] = np.sum((agent_i_accuracy == 0) & (agent_j_accuracy == 0))  # Both wrong

            confusion_matrix_all[key_name] = confusion_matrix
    
    return confusion_matrix_all
    
def match_answer_majority_voting_each_round(gt_answer: np.ndarray, agent_answer: np.ndarray) -> np.ndarray:
    """Compare the majority voting result of agents with the ground truth for each round.

    Args:
        gt_answer (np.ndarray): Ground truth answers, shape should be (question_num,).
        agent_answer (np.ndarray): Answers from each agent for each round, shape should be (question_num, round_num, agent_num).
    
    Returns:
        np.ndarray: Whether the majority voting result matches the ground truth (1: match, 0: no match or no majority), shape is (question_num, round_num).
    """
    question_num, round_num, agent_num = agent_answer.shape
    
    # Initialize the result array
    result = np.zeros(shape=(question_num, round_num))
    
    for q in range(question_num):
        for r in range(round_num):
            # Get the answers of all agents for this question and round
            answers = agent_answer[q, r, :]
            
            # Count the occurrences of each answer
            unique_answers, counts = np.unique(answers, return_counts=True)
            
            # Find the answer with the maximum count (majority voting)
            max_count_index = np.argmax(counts)
            majority_answer = unique_answers[max_count_index]
            
            # Check if the majority answer has more than half of the agents agreeing
            if counts[max_count_index] > agent_num / 2:
                # Compare the majority answer with the ground truth
                result[q, r] = float(majority_answer == gt_answer[q])
            else:
                # No majority, set to 0
                result[q, r] = 0
    
    return result

def get_confusion_matrix_agent_vs_majority(gt_answer: np.ndarray, agent_answer: np.ndarray) -> Dict[str, np.ndarray]:
    """Get the confusion matrix between each agent and the majority voting result.

    Args:
        gt_answer (np.ndarray): Ground truth answers, shape should be (question_num,).
        agent_answer (np.ndarray): Answers from each agent for each round, shape should be (question_num, round_num, agent_num).
    
    Returns:
        Dict[str, np.ndarray]: A dictionary where the key is "agent{i}" and the value is a 3D np.ndarray with shape (round_num, 2, 2),
            representing the confusion matrix for each round between the agent and the majority voting result.
            The confusion matrix is structured as:
                          col 0                 col 1
            row_0 (agent T, majority T), (agent T, majority F)
            row_1 (agent F, majority T), (agent F, majority F)
    """
    question_num, round_num, agent_num = agent_answer.shape
    
    # Initialize the result dictionary
    confusion_matrix_all = {}
    
    # Calculate the majority voting result for each question and round
    majority_voting_result = np.zeros((question_num, round_num))
    for q in range(question_num):
        for r in range(round_num):
            answers = agent_answer[q, r, :]
            unique_answers, counts = np.unique(answers, return_counts=True)
            max_count_index = np.argmax(counts)
            majority_answer = unique_answers[max_count_index]
            if counts[max_count_index] > agent_num / 2:
                majority_voting_result[q, r] = majority_answer
            else:
                majority_voting_result[q, r] = np.nan  # No majority
    
    # Calculate the confusion matrix for each agent
    for i in range(agent_num):
        key_name = f"agent{i}"
        confusion_matrix = np.zeros((round_num, 2, 2))
        
        for r in range(round_num):
            # Get the agent's answers and the majority voting result for this round
            agent_answers = agent_answer[:, r, i]
            majority_results = majority_voting_result[:, r]
            
            # Filter out questions where there is no majority
            valid_indices = ~np.isnan(majority_results)
            agent_answers_valid = agent_answers[valid_indices]
            majority_results_valid = majority_results[valid_indices]
            gt_answer_valid = gt_answer[valid_indices]
            
            # Compare agent's answers with majority results
            agent_correct = (agent_answers_valid == gt_answer_valid)
            majority_correct = (majority_results_valid == gt_answer_valid)
            
            # Calculate the confusion matrix
            confusion_matrix[r, 0, 0] = np.sum(agent_correct & majority_correct)  # Both correct
            confusion_matrix[r, 0, 1] = np.sum(agent_correct & ~majority_correct)  # Agent correct, majority wrong
            confusion_matrix[r, 1, 0] = np.sum(~agent_correct & majority_correct)  # Agent wrong, majority correct
            confusion_matrix[r, 1, 1] = np.sum(~agent_correct & ~majority_correct)  # Both wrong
        
        confusion_matrix_all[key_name] = confusion_matrix
    
    return confusion_matrix_all

if __name__ == "__main__":
    from eval_data import extract_parameters
    
    filename = "./multi_math_results_er30_agents4_dr5_ratio1.0_range30.p"

    params = extract_parameters(filename)
    er = params['er']
    agents = params['agents']
    dr = params['dr']
    
    data = pickle.load(open(filename,'rb'))

    gt_answer = np.zeros(er)
    agent_answer = np.zeros((er, dr, agents))

    for traj_id, traj in data.items():
        gt_answer[traj_id] = traj['answer']
        for debate_context in traj['states']:
            agent_answer[traj_id][debate_context['round']] = debate_context['text_answer']
    
    accuracy = match_answer_each_agent_each_round(gt_answer, agent_answer)
    print("="*8, "match result", "="*8)
    print(accuracy, accuracy.shape)

    confusion_matrix_agent_vs_agent = get_confusion_matrix_for_agent_each_round(gt_answer, agent_answer)
    print("="*8, "confusion matrix between agents", "="*8)
    for k, v in confusion_matrix_agent_vs_agent.items():
        print(k)
        print(v, v.shape)

    majority_voting_accuracy = match_answer_majority_voting_each_round(gt_answer, agent_answer)
    print("="*8, "majority voting match result", "="*8)
    print(majority_voting_accuracy, majority_voting_accuracy.shape)

    confusion_matrix_agent_vs_majority = get_confusion_matrix_agent_vs_majority(gt_answer, agent_answer)
    print("="*8, "confusion matrix agents vs majority", "="*8)
    for k, v in confusion_matrix_agent_vs_majority.items():
        print(k)
        print(v, v.shape)

