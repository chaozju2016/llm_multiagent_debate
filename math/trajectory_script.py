import json
import subprocess
import os
import pickle
import pprint

def running_N_circles(N, running_file, output_dir):
    """
    Run N circles and get N results files with different indexes to distinguish from each other.

    Args:
        N (int): Number of iterations to run. Set debugging to 2 and ultimately run 1000 iterations.
        running_file (str): Name of the file to execute.
    """
    # output_dir = "math_results_agents3_rounds8_ratio1.0_range30_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"CreateDir: {output_dir}")

    for i in range(N):
        print(f"Running {running_file}, iteration {i}...")

        # Run the target file
        # 默认已经开了服务了 9090要不要参数化
        port = '9090'
        subprocess.run(["python3.8", running_file, "-p", port])

        # Rename the generated result.p file to avoid overwriting
        # 文件名要不要参数化
        agents = 3
        rounds = 10
        ratio = 1.0
        ranges = 30
        original_file = f"math_results_agents{agents}_rounds{rounds}_ratio{ratio}_range{ranges}.p"
        if os.path.exists(original_file):
            new_file_name = f"result_{i}.p"
            new_file_path = os.path.join(output_dir, new_file_name)
            os.rename(original_file, new_file_path)
            print(f"Saved: {new_file_path}")
        else:
            print(f"Warning: {original_file} not found after iteration {i}!")

def merge_p_files(output_dir, merged_file):
    """
    Merge all .p files in the output directory into a single .p file.

    Args:
        output_dir (str): Directory containing .p files.
        merged_file (str): Name of the merged output file.
    """
    merged_data = {"trajectories": []}

    # Traverse all .p files in the directory
    index_num = 0
    for file_name in sorted(os.listdir(output_dir)):
        if file_name.endswith(".p"):

            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                merged_data["trajectories"].append({"trajectory_id":index_num,
                             "states":data})
                print(f"AppendTask:No.{index_num} is completed ")
                index_num += 1
    pprint.pprint(merged_data)
    # Save the merged data
    with open(merged_file, "wb") as f:
        pickle.dump(merged_data, f)

    print(f"All results merged into {merged_file}")

    # demo_visualized_file_name = "demo_data.json"
    # with open(demo_visualized_file_name, "w") as f:
    #     json.dump(merged_data, f, indent=4, ensure_ascii=False) 
    # print(f"Visualized data saved to {demo_visualized_file_name}")


    

if __name__ == "__main__":
    # Configuration
    N = 2  # Set debugging to 2, and ultimately run 1000 iterations
    
    output_dir = "results_agents3_rounds10_ratio1.0_range30"
    running_file = "gen_math.py"
    merged_file = "merged_results.p"

    # Run N iterations and save results
    running_N_circles(N, running_file, output_dir)

    # Merge all result files
    merge_p_files(output_dir, merged_file)
