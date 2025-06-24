import json
import numpy as np
from typing import List, Tuple

def find_nearest_points(json_path: str, target_point: Tuple[float, float], k: int) -> List[int]:
    """
    查找 representation_tsne 空间中距离目标点 (X, Y) 最近的 K 个点的索引。
    
    参数:
        json_path (str): 包含 t-SNE 结果的 JSON 文件路径。
        target_point (Tuple[float, float]): 目标点的 (X, Y) 坐标。
        k (int): 需要返回的最近点个数。
    
    返回:
        List[int]: representation_tsne 中最近 K 个点的索引列表。
    
    抛出:
        FileNotFoundError: 如果 JSON 文件未找到。
        ValueError: 如果 JSON 结构无效或 k 超过点数。
    """
    # 读取 JSON 文件
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"未找到 JSON 文件: {json_path}")
    
    # 提取 representation_tsne 数据
    representation_tsne = np.array(data['representation_tsne'])
    
    # 输入验证
    if len(representation_tsne) == 0:
        raise ValueError("representation_tsne 数据为空")
    if k > len(representation_tsne):
        raise ValueError(f"k ({k}) 不能大于点数 ({len(representation_tsne)})")
    if len(target_point) != 2:
        raise ValueError("target_point 必须是 (X, Y) 坐标的元组")
    
    # 将目标点转换为 numpy 数组
    target = np.array(target_point)
    
    # 计算欧几里得距离
    distances = np.sqrt(np.sum((representation_tsne - target) ** 2, axis=1))
    
    # 获取最近 K 个点的索引
    nearest_indices = np.argsort(distances)[:k].tolist()
    
    return nearest_indices

# 演示示例
if __name__ == "__main__":
    # 设置参数
    target_point = (-3.0, -2.9)  # 目标点 (X, Y)
    k = 10  # 查找最近的 2 个点
    
    try:
        # 调用函数
        nearest_points = find_nearest_points("T-SNE.json", target_point, k)
        print(f"目标点 ({target_point[0]}, {target_point[1]}) 最近的 {k} 个点的索引: {nearest_points}")
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: {e}")