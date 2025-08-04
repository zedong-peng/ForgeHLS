import os
import argparse
# import logging

# 配置日志记录
# logging.basicConfig(filename='add_tcl.log', level=logging.INFO)

def generate_tcl_script(cpp_file_path):
    # top_function_name.txt in the dir of cpp_file_path is the top function name. read it.
    top_function_name_file = os.path.join(os.path.dirname(cpp_file_path), 'top_function_name.txt')
    try:
        with open(top_function_name_file, 'r') as f:
            top_function_name = f.read().strip()
    except FileNotFoundError:
        print(f"Can not find: {top_function_name_file}. Need to run extract_top_function.py")
        return

    file_list =[]
    # 遍历cpp_file_path下同目录的所有*.c *.cpp *.h *.hpp
    for root, dirs, files in os.walk(os.path.dirname(cpp_file_path), topdown=True):
        for file in files:
            if file.endswith(('.c', '.cpp', '.h', '.hpp')):
                file_list.append(os.path.join(root, file))
    add_files_txt = ""
    for file in file_list:
        add_files_txt += f"add_files {os.path.basename(file)}\n"

    # 生成TCL
    tcl_script = f"""
open_project -reset project
set_top {top_function_name}
{add_files_txt}
open_solution -reset "solution1"
set_part xcu280-fsvh2892-2L-e
create_clock -period 10
csynth_design
close_solution
exit
"""
    # 保存TCL脚本在cpp_file_path 同级目录
    tcl_filename = os.path.join(os.path.dirname(cpp_file_path), f'run_hls.tcl')
    with open(tcl_filename, 'w') as tcl_file:
        tcl_file.write(tcl_script)
    print(f"TCL 脚本已保存：{tcl_filename}")

def create_tcl_scripts_in_dir(repo_dir):
    """处理仓库中的所有 C++ 文件，为顶层函数生成 TCL 脚本。"""
    # 查找所有 C++ 源文件
    cpp_file_path_list = []
    # 遍历 repo_dir 目录及其子目录
    for root, dirs, files in os.walk(repo_dir, topdown=True):
        for file in files:
            if '.autopilot' in dirs:
                # 从 dirs 列表中移除 .autopilot 目录，停止遍历其子目录
                dirs.remove('.autopilot')
            if file.endswith(('.cpp', '.c')):
                cpp_file_path_list.append(os.path.join(root, file))

    if not cpp_file_path_list:
        # logging.warning(f"仓库中未找到 C++ 源文件：{repo_dir}")
        print(f"No cpp files found in the repo: {repo_dir}")
        return

    # 处理每个 C++ 文件
    for cpp_file_path in cpp_file_path_list:
        generate_tcl_script(cpp_file_path)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='为HLS项目生成TCL脚本')
    parser.add_argument('--dir', type=str, required=True, help='要处理的目录路径')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    create_tcl_scripts_in_dir(args.dir)
