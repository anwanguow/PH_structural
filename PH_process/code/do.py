import os
import numpy as np

# 获取当前目录下的文件列表
current_files = [f for f in os.listdir() if f.startswith('PI_decay_0_') and f.endswith('.py')]

# 读取已有的i=0的文件内容
file_content = {}
for file in current_files:
    with open(file, 'r') as f:
        file_content[file] = f.readlines()

# 生成新的文件并修改第8行
for i in range(1, 9):
    for j in range(8):
        new_file_name = f'PI_decay_{i}_{j}.py'
        old_file_name = f'PI_decay_0_{j}.py'
        new_lines = file_content[old_file_name].copy()
        new_lines[7] = f'frames = np.array([100*({i}+1)])\n'  # 修改第8行内容
        with open(new_file_name, 'w') as new_file:
            new_file.writelines(new_lines)

print("新文件已生成并修改。")
