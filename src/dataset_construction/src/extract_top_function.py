import os
from tqdm import tqdm
import concurrent.futures
import argparse
# import logging
from extract_top_function_by_clang import extract_top_function_by_clang
from extract_top_function_by_gpt4omini import extract_top_function_by_gpt4omini
from find_cpp_c_files import find_cpp_c_files

def extract_top_function(cpp_file_path):
    try:
        top_function_name = extract_top_function_by_clang(cpp_file_path)
    except Exception as e:
        print(f"[ERROR] Clang cannot find top function: {cpp_file_path}. Start to use GPT-4.")
    
    # judge if top_function_name is None
    if top_function_name is None:
        try:
            top_function_name = extract_top_function_by_gpt4omini(cpp_file_path)
        except Exception as e:
            print(f"[ERROR] GPT-4 cannot find top function: {cpp_file_path}")
            return None
    # judge if top_function_name is more than 1  uncalled_function_list = set()
    if isinstance(top_function_name, set):
        print(f"Top function name is more than 1: {top_function_name}")
        top_function_name_gpt = extract_top_function_by_gpt4omini(cpp_file_path)
        if top_function_name_gpt in top_function_name:
            top_function_name = top_function_name_gpt
        else:
            print(f"[ERROR] The top function name from GPT-4 is not in the top function name from Clang for {cpp_file_path}")
            return None

    print(f"Top function name: {top_function_name}")
    # save top_function_name in the same dir in cpp_file_path
    top_function_name_file = os.path.join(os.path.dirname(cpp_file_path), 'top_function_name.txt')
    with open(top_function_name_file, 'w') as f:
        f.write(top_function_name)
    print(f"Top function name saved to {top_function_name_file}")
    return top_function_name

def extract_top_function_in_dir(search_dir):
    """
    在指定目录及其子目录（最大深度 max_depth）中搜索 .cpp 和 .c 文件，并提取顶层函数。
    
    :param search_dir: 要搜索的目录
    """
    cpp_c_files = find_cpp_c_files(search_dir)
    with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
        # 创建进度条
        with tqdm(total=len(cpp_c_files), desc="Processing files") as pbar:
            # 提交任务
            futures = {executor.submit(extract_top_function, file): file for file in cpp_c_files}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing file {futures[future]}: {e}")
                # 更新进度条
                pbar.update(1)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='提取C/C++文件中的顶层函数')
    parser.add_argument('--dir', type=str, default='./workspace/HLSBatchProcessor/data/kernels',
                        help='要处理的目录路径')
    parser.add_argument('--file', type=str, default=None,
                        help='单个文件路径，如果指定则只处理该文件')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.file:
        # 处理单个文件
        extract_top_function(args.file)
    else:
        # 处理目录
        extract_top_function_in_dir(args.dir)