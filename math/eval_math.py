import pickle
import numpy as np

def analyze_rounds(filename):
    # 读取pickle文件
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    # 初始化存储每轮统计数据的字典
    round_stats = {}
    for round_num in range(5):  # 0-7轮
        correct_count = 0
        total_count = 0
        errors = []

        # 遍历所有题目
        for item in data:
            true_answer = item['answer']
            round_answers = item['text_answers'][round_num]  # 这一轮三个agent的答案

            # 处理每个agent的答案
            for ans_str in round_answers:
                try:
                    ans = int(ans_str)
                    total_count += 1
                    if ans == true_answer:
                        correct_count += 1
                    errors.append(abs(ans - true_answer))
                except ValueError:
                    continue

        # 计算这一轮的统计数据
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        mean_error = np.mean(errors) if errors else 0

        round_stats[round_num] = {
                'accuracy': accuracy,
                'mean_error': mean_error
                }
    return round_stats

# 分析所有比例的文件
ratios = [0.0]
#ratios = [0.0, 0.2, 0.5, 0.8, 1.0]

for ratio in ratios:
    #filename = f'range30_round100/math_results_agents3_rounds8_ratio{ratio}_range30.p'
    filename = f'math_results_er30_agents3_dr10_ratio{ratio}_range100.p'
    print(f"\nAnalyzing results for ratio {ratio}:")

    try:
        stats = analyze_rounds(filename)
        print("\nRound\tAccuracy(%)\tMean Error")
        print("-" * 40)
        for round_num in range(8):
            stat = stats[round_num]
            print(f"{round_num}\t{stat['accuracy']:.2f}\t\t{stat['mean_error']:.2f}")
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
