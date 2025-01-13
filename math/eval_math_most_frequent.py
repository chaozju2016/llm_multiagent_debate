import pickle
import numpy as np
from collections import Counter

def get_most_frequent_answer(answers_till_now):
    """
    获取到目前为止出现次数最多的答案
    args:
        answers_till_now: 到当前轮为止的所有答案列表
    """
    try:
        # 转换所有合法答案为整数
        int_answers = []
        for ans in answers_till_now:
            try:
                int_answers.append(int(ans))
            except (ValueError, TypeError):
                continue
        
        if not int_answers:  # 如果没有合法答案
            return None
            
        # 计算出现次数并返回最频繁的答案
        count = Counter(int_answers)
        return count.most_common(1)[0][0]
            
    except Exception as e:
        print(f"Error in get_most_frequent_answer: {e}")
        return None

def analyze_rounds(filename):
    # 读取pickle文件
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    # 初始化存储每轮统计数据的字典
    round_stats = {}
    
    # 为每个问题维护一个累积答案列表
    cumulative_answers = [{} for _ in data]  # 每个问题一个字典
    
    for round_num in range(5):  # 0-7轮
        correct_count = 0
        total_count = 0
        errors = []
        answer_counts = []  # 记录当前累积的答案数量
        
        # 遍历所有题目
        for idx, item in enumerate(data):
            true_answer = item['answer']
            round_answers = item['text_answers'][round_num]  # 这一轮的答案
            
            # 添加这一轮的答案到累积列表
            if round_num not in cumulative_answers[idx]:
                cumulative_answers[idx][round_num] = []
            
            # 添加有效答案
            valid_answers = [ans for ans in round_answers if ans.strip() and ans.replace('-','').isdigit()]
            cumulative_answers[idx][round_num].extend(valid_answers)
            
            # 获取到目前为止的所有答案
            all_answers_till_now = []
            for r in range(round_num + 1):
                if r in cumulative_answers[idx]:
                    all_answers_till_now.extend(cumulative_answers[idx][r])
            
            answer_counts.append(len(all_answers_till_now))
            
            # 获取最频繁答案
            most_freq_ans = get_most_frequent_answer(all_answers_till_now)
            
            if most_freq_ans is not None:
                total_count += 1
                if most_freq_ans == true_answer:
                    correct_count += 1
                errors.append(abs(most_freq_ans - true_answer))
        
        # 计算这一轮的统计数据
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        mean_error = np.mean(errors) if errors else 0
        avg_answers = np.mean(answer_counts) if answer_counts else 0
        
        round_stats[round_num] = {
            'accuracy': accuracy,
            'mean_error': mean_error,
            'total_questions': total_count,
            'correct_count': correct_count,
            'avg_cumulative_answers': avg_answers
        }
    
    return round_stats

def main():
    # 分析所有比例的文件
    ratios = [0.0, 0.2, 0.5]
    #ratios = [0.0, 0.2, 0.5, 0.8, 1.0]

    for ratio in ratios:
        filename = f'math_results_er30_agents3_dr5_ratio{ratio}_range30.p'
        print(f"\nAnalyzing results for ratio {ratio}:")
        
        try:
            stats = analyze_rounds(filename)
            print("\nRound\tAccuracy(%)\tMean Error\tCorrect/Total\tAvg Cum.Answers")
            print("-" * 80)
            for round_num in range(5):
                stat = stats[round_num]
                print(f"{round_num}\t{stat['accuracy']:.2f}\t\t{stat['mean_error']:.2f}\t\t"
                      f"{stat['correct_count']}/{stat['total_questions']}\t\t{stat['avg_cumulative_answers']:.2f}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    main()
