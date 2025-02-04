import numpy as np
from typing import Union, Tuple

def compute_similarities(embeddings: np.ndarray, 
                       return_format: str = 'matrix',
                       round_decimals: int = 4) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    计算归一化向量之间的余弦相似度。
    
    Args:
        embeddings: 归一化后的嵌入向量矩阵，shape为(n_samples, n_features)
        return_format: 返回格式，可选 'matrix' 或 'pairs'
        round_decimals: 结果保留的小数位数
    
    Returns:
        如果 return_format='matrix':
            返回相似度矩阵 shape=(n_samples, n_samples)
        如果 return_format='pairs':
            返回两个数组: (相似度数组, 对应的索引对数组)
    """
    # 验证输入
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)
    
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings 必须是2维数组，当前维度: {embeddings.ndim}")
    
    # 计算相似度矩阵
    similarity_matrix = np.dot(embeddings, embeddings.T)
    
    # 处理可能的数值误差，将对角线上的值限制在1
    np.fill_diagonal(similarity_matrix, 1.0)
    
    # 四舍五入到指定小数位
    similarity_matrix = np.round(similarity_matrix, round_decimals)
    
    if return_format == 'matrix':
        return similarity_matrix
    
    elif return_format == 'pairs':
        # 获取上三角的索引和值（不包含对角线）
        upper_indices = np.triu_indices(len(embeddings), k=1)
        similarities = similarity_matrix[upper_indices]
        
        # 创建配对索引数组
        pairs = np.array(list(zip(upper_indices[0], upper_indices[1])))
        
        return similarities, pairs
    
    else:
        raise ValueError("return_format 必须是 'matrix' 或 'pairs'")