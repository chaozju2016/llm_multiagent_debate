import pickle
import os
import argparse
import pandas as pd
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

def plot_round_token_per_question(compare_type:str):
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['font.weight'] = 'normal'
    mpl.rcParams['font.size'] = 18

    # 数据定义
    gsm_ours = [2050.24, 4936.63, 4970.38, 4861.14, 5125.86]
    gsm_full = [3369.65, 10532.88, 17261.80, 17535.95, 15384.48]
    mmlu_ours = [2101.68, 3185.47, 3876.48, 3852.21, 4041.00]
    mmlu_full = [1753.65, 6543.49, 12034.03, 15077.61, 16741.90]

    # 分组标签和参数
    categories = ['Round 1', 'Round 2', 'Round 3', 'Round 4', 'Round 5']  # 假设5个测试场景
    bar_width = 0.3  # 柱子宽度
    x = np.arange(len(categories))  # x轴基准位置

    # 创建画布
    plt.figure(figsize=(15, 7))

    # 绘制柱子
    if compare_type =='gsm':
        plt.bar(x + 0.5*bar_width, gsm_ours, width=bar_width, label='GSM ASMAD (Ours)', color='#4e79a7')
        plt.bar(x - 0.5*bar_width, gsm_full, width=bar_width, label='GSM MAD', color='#f28e2b')
    elif compare_type == 'mmlu': 
        plt.bar(x + 0.5*bar_width, mmlu_ours, width=bar_width, label='MMLU ASMAD (Ours)', color='#4e79a7')
        plt.bar(x - 0.5*bar_width, mmlu_full, width=bar_width, label='MMLU MAD', color='#f28e2b')
    # plt.bar(x + 0.5*bar_width, mmlu_ours, width=bar_width, label='MMLU OURS', color='#59a14f')
    # plt.bar(x + 1.5*bar_width, mmlu_full, width=bar_width, label='MMLU FULL', color='#e15759')

    # 添加数值标签（自动简化显示）
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height,
                    f'{height/1e6:.1f}M' if height > 1e6 else f'{height/1e3:.0f}K',
                    ha='center', va='bottom', fontsize=14)

    add_labels(plt.gca().containers[0])  # 只显示GSM OURS标签避免重叠
    add_labels(plt.gca().containers[1])  # 只显示GSM OURS标签避免重叠
    # add_labels(plt.gca().containers[2])  # 显示MMLU OURS标签

    # 图表装饰
    plt.xticks(x, categories, rotation=0)
    plt.ylabel('Token Cost (per question)', fontsize=20,fontweight='bold')
    # plt.title('Performance Comparison: OURS vs FULL Models', fontsize=14, pad=20)
    plt.legend(loc='upper left', ncol=2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Y轴格式化为千分位
    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

    plt.tight_layout()

    save_path = './round_token_compare_fig_'+compare_type+'.png'
    plt.savefig(save_path, bbox_inches='tight')  # bbox_inches='tight' ensures the legend is included in the saved image

    plt.show()

# python round_token_compare_fig.py -t gsm
# python round_token_compare_fig.py -t mmlu
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='math')
    parser.add_argument('-t', '--type', type=str, default=None, help='gsm or mmlu')
    args = parser.parse_args()
    plot_round_token_per_question(args.type)