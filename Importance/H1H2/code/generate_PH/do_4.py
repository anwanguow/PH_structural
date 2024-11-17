import numpy as np
import os
import pickle
from PointCloud import get_neighbor
from PH import PH_barcode_h1h2

# 设置全局变量
traj_index = 3

# 设置目录路径
set_dirs = ["data/distance_matrix/set_1"]
output_dirs = ["PI/barcode/set_1"]
frames = [100] + list(range(200, 1000, 100))
r_values = np.arange(1.0, 5.1, 0.1)

# 遍历set_1和set_2
for set_index, set_dir in enumerate(set_dirs):
    traj_dir = os.path.join(set_dir, f"D_{traj_index}")
    output_traj_dir = os.path.join(output_dirs[set_index], f"D_{traj_index}")
    os.makedirs(output_traj_dir, exist_ok=True)
    
    for frame in frames:
        # 加载距离矩阵
        distance_matrix_path = os.path.join(traj_dir, f"D_{frame}.npy")
        D = np.load(distance_matrix_path)
        
        # 获取粒子数目
        num_particles = D.shape[0]
        
        for r in r_values:
            # 计算每个粒子的持续同调信息
            barcodes = []
            for i in range(num_particles):
                neighbor, indices = get_neighbor(D, i, r=r)
                barcode = PH_barcode_h1h2(neighbor)
                barcodes.append(barcode)
            
            # 保存结果，使用pickle
            output_path = os.path.join(output_traj_dir, f"D_{frame}_{r:.1f}.pkl")
            with open(output_path, 'wb') as f:
                pickle.dump(barcodes, f)

