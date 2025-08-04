import os

def find_cpp_c_files(search_dir, max_depth=4):
    """
    在指定目录及其子目录（最大深度 max_depth）中搜索 .cpp 和 .c 文件。
    
    :param search_dir: 要搜索的目录
    :param max_depth: 最大搜索深度（默认值为2）
    :return: 包含 .cpp 和 .c 文件路径的列表
    """
    cpp_c_files = []
    
    for root, dirs, files in os.walk(search_dir):
        # 过滤 .autopilot 目录
        if '.autopilot' in dirs:
                # 从 dirs 列表中移除 .autopilot 目录，停止遍历其子目录
                dirs.remove('.autopilot')
        # 计算当前目录的深度
        rel_path = os.path.relpath(root, search_dir)
        depth = 0 if rel_path == "." else rel_path.count(os.sep) + 1
        
        if depth >= max_depth:
            # 达到最大深度，不再递归子目录
            dirs[:] = []

        # collect .cpp 和 .c 
        cpp_c_files.extend(
            os.path.join(root, file) for file in files if file.endswith(('.cpp', '.c'))
        )
    
    return cpp_c_files

# 测试
if __name__ == '__main__':
    path = "./workspace/HLSBatchProcessor/data/kernels/CHStone"
    cpp_c_files = find_cpp_c_files(path)
    print(cpp_c_files)
