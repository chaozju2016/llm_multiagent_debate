import pickle
import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def extract_upper_triangle(matrix):
    """
    Extracts the upper triangle of a square matrix, excluding the diagonal elements.

    Args:
        matrix (list of list of int/float): A square matrix (2D list) from which the upper triangle is to be extracted.

    Returns:
        list: A list containing the elements of the upper triangle of the matrix.
    """
    upper_triangle = []
    n = len(matrix)
    
    for i in range(n):
        for j in range(i + 1, n):  # Start from i+1 to skip the diagonal elements
            upper_triangle.append(matrix[i][j])
    
    return upper_triangle

def process_file(file_name: str):
    """
    Processes a file containing problem data, extracts similarity scores from the upper triangle of similarity matrices,
    and organizes the data into a pandas DataFrame.

    Args:
        file_name (str): The name of the file to be processed. The file should contain serialized data in pickle format.

    Returns:
        pd.DataFrame: A DataFrame containing the round index, similarity scores, and problem index for each entry.
    """
    with open(file_name,'rb') as f:
        data = pickle.load(f) # Load the serialized data from the file
    
    round_idx = []  # List to store round indices
    simularity_score = []  # List to store similarity scores
    problem_idx = []  # List to store problem indices

    problem_num = len(data)  # Total number of problems in the data

    for tmp_problem_index in range(problem_num):
        problem_data = data[tmp_problem_index]['states']  # Extract the states for the current problem
        round_num = len(problem_data)  # Number of rounds in the current problem

        for i in range(round_num):
            tmp_round_idx = problem_data[i]['round']  # Extract the round index

            # Extract the upper triangle of the similarity matrix for the current round
            assert 'sim_matrix' in problem_data[i].keys(), "Keyword sim_matrix not found"
            tmp_simularity_score = extract_upper_triangle(problem_data[i]['sim_matrix'])

            # Append the extracted similarity scores, round index, and problem index to their respective lists
            for score in tmp_simularity_score:
                simularity_score.append(score)
                round_idx.append(tmp_round_idx)
                problem_idx.append(tmp_problem_index)

    # Store the collected data in a dictionary
    simu_data_dict = {
        "round": round_idx,  # List of round indices
        "simularity": simularity_score,  # List of similarity scores
        "problem": problem_idx  # List of problem indices
    }

    # Convert the dictionary to a pandas DataFrame
    data_df = pd.DataFrame(simu_data_dict)

    return data_df

def calculate_simi_mean_std(df):
    # group by round and calculate mean and std of simularity
    grouped = df.groupby('round')['simularity'].agg(['mean', 'std'])
    
    # get the max round
    max_round = grouped.index.max()
    
    # initialize the result list
    result = []
    
    # iter from 0 to max round
    for round_num in range(max_round + 1):
        if round_num in grouped.index:
            mean = grouped.loc[round_num, 'mean']
            std = grouped.loc[round_num, 'std']
        else:
            # if the round not exist, then mean and std is NaN
            mean, std = np.nan, np.nan
        
        # add the result to the list
        result.append({'round': round_num, 'mean': mean, 'std': std})
    
    return result

def plot_swarm_figure(data_df: pd.DataFrame, save_path: str = None):
    """
    Plots a swarm plot of similarity scores across rounds, with points colored by problem index.

    Args:
        data_df (pd.DataFrame): A DataFrame containing the columns 'round', 'simularity', and 'problem'.
        save_path (str, optional): The file path to save the plot. If None, the plot is not saved.
    """
    plt.clf()
    # Create a swarm plot with 'round' on the x-axis, 'simularity' on the y-axis, and colored by 'problem'
    # sns.swarmplot(x="round", y="simularity", hue="problem", data=data_df, palette="viridis",
    #     size=3,  # 减小点的大小
    #     alpha=0.6,  # 添加透明度
    #     )
    sns.stripplot(
        x="round", 
        y="simularity", 
        hue="problem", 
        data=data_df, 
        palette="viridis",
        size=3,
        alpha=0.6,
        jitter=True,  # 添加随机抖动
        dodge=True,    # 按 problem 分组错开显示
    )

    plt.title("Swarm Plot of Similarity Scores by Problem Index")
    plt.xlabel("Round")
    plt.ylabel("Similarity Score")
    
    # Add a legend to clarify the problem indices
    plt.legend(title="Problem Index", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save the plot if a save path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')  # bbox_inches='tight' ensures the legend is included in the saved image
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='math')
    parser.add_argument('-d', '--data_dir', type=str, default=None, help='data dir')
    parser.add_argument('-f', '--file_name', type=str, default=None, help='file name')
    args = parser.parse_args()
    
    data_file_list = []
    
    if args.file_name is None:
        if args.data_dir is None:
            raise ValueError('Please provide a file name')
        else:
            data_file_list = os.listdir(args.data_dir)
            # leave only pickle files
            data_file_list = [os.path.join(args.data_dir, file) for file in data_file_list if file.endswith('.p')]
            
    if args.data_dir is None:
        if args.file_name is None:
            raise ValueError('Please provide a data dir')
        else:
            data_file_list = [args.file_name]
    
    for data_file_name in data_file_list:
        data_df = process_file(file_name=data_file_name)
        plot_swarm_figure(data_df=data_df, save_path=data_file_name.replace('.p', '.png'))
        # reset sns
        sns.reset_orig()
        # reset plt
        plt.clf()