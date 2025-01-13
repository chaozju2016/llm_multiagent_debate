import pickle
import numpy as np
from collections import Counter

def get_majority_answer(answers):
    """
    获取多数投票的答案
    如果答案列表只有1个元素，直接返回该答案
    如果有2个或以上元素，返回出现次数最多的答案
    """
    try:
        # 转换所有合法答案为整数
        int_answers = []
        for a in answers:
            try:
                int_answers.append(int(a))
            except (ValueError, TypeError):
                continue
        
        if not int_answers:  # 如果没有合法答案
            return None
        elif len(int_answers) == 1:  # 如果只有一个答案
            return int_answers[0]
        else:  # 如果有多个答案，返回出现最多的
            count = Counter(int_answers)
            return count.most_common(1)[0][0]
            
    except Exception as e:
        print(f"Error in get_majority_answer: {e}")
        return None

def analyze_rounds(filename):
    # 读取pickle文件
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    # 初始化存储每轮统计数据的字典
    round_stats = {}
    for round_num in range(8):  # 0-7轮
        correct_count = 0
        total_count = 0
        errors = []
        answer_counts = []  # 记录每题的答案数量
        
        # 遍历所有题目
        for item in data:
            true_answer = item['answer']
            round_answers = item['text_answers'][round_num]  # 这一轮的答案
            
            # 统计有效答案数量
            valid_answers = [ans for ans in round_answers if ans.strip() and ans.replace('-','').isdigit()]
            answer_counts.append(len(valid_answers))
            
            # 获取多数投票答案
            majority_ans = get_majority_answer(round_answers)
            
            if majority_ans is not None:
                total_count += 1
                if majority_ans == true_answer:
                    correct_count += 1
                errors.append(abs(majority_ans - true_answer))
        
        # 计算这一轮的统计数据
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        mean_error = np.mean(errors) if errors else 0
        avg_answers = np.mean(answer_counts) if answer_counts else 0
        
        round_stats[round_num] = {
            'accuracy': accuracy,
            'mean_error': mean_error,
            'total_questions': total_count,
            'correct_count': correct_count,
            'avg_answers_per_question': avg_answers
        }
    
    return round_stats

# 分析所有比例的文件
ratios = [0.0, 0.2, 0.5, 0.8, 1.0]

for ratio in ratios:
    filename = f'range30_round100/math_results_agents3_rounds8_ratio{ratio}_range30.p'
    print(f"\nAnalyzing results for ratio {ratio}:")
    
    try:
        stats = analyze_rounds(filename)
        print("\nRound\tAccuracy(%)\tMean Error\tCorrect/Total\tAvg Answers")
        print("-" * 75)
        for round_num in range(8):
            stat = stats[round_num]
            print(f"{round_num}\t{stat['accuracy']:.2f}\t\t{stat['mean_error']:.2f}\t\t"
                  f"{stat['correct_count']}/{stat['total_questions']}\t\t{stat['avg_answers_per_question']:.2f}")
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
