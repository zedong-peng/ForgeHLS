from operator import le
import clang.cindex
import json
import os
import sys
from anytree import Node, RenderTree, PreOrderIter
from anytree.exporter import DictExporter
import subprocess
from datetime import datetime
from find_cpp_c_files import find_cpp_c_files
import argparse

def extract_for_loops(ast_node, loop_node, cpp_file=None):
    """
    递归遍历 AST, 提取所有 for 循环并构建树状结构。
    
    Args:
        ast_node: 当前 Clang AST 节点
        loop_node: 当前树结构的父节点
    
    Returns:
        None
    """
    # 如果当前节点是一个 for 循环
    if f"{ast_node.location.file}" == cpp_file:
        if ast_node.kind == clang.cindex.CursorKind.FOR_STMT:
            line = ast_node.location.line
            for child in ast_node.get_children():
                if child.kind == clang.cindex.CursorKind.COMPOUND_STMT:
                    # 如果 for 循环的子节点是复合语句，将复合语句作为当前节点
                    line = child.location.line
                    break
            # 创建树节点
            current_node = Node(
                f"for_loop Line:{line}",
                line=line,
                tripcount=0,
            )
            current_node.parent = loop_node
        else:
            current_node = loop_node  # 如果不是 for 循环，保持当前父节点不变
    else:
        current_node = loop_node  # 如果不在指定文件，保持当前父节点不变

    # 继续遍历子节点，查找嵌套的 for 循环
    for child in ast_node.get_children():
        extract_for_loops(child, loop_node=current_node, cpp_file=cpp_file)

def save_tree_to_json(root, output_file):
    """
    将树状结构导出为 JSON 文件。
    
    Args:
        root: 树的根节点
        output_file: 输出 JSON 文件的路径
    
    Returns:
        None
    """
    exporter = DictExporter()
    tree_dict = exporter.export(root)
    
    with open(output_file, 'w') as f:
        json.dump(tree_dict, f, indent=4)
    
    print(f"树状结构已保存到 {output_file}")

def print_tree(root):
    """
    打印树状结构到控制台。
    
    Args:
        root: 树的根节点
    
    Returns:
        None
    """
    for pre, fill, node in RenderTree(root):
        print(f"{pre}{node.name}")

def extract_tripcount(cpp_file, vitis_hls_dir="/tools/Xilinx/Vitis_HLS/2023.2/include"):
    """
    从文件中提取循环的迭代次数。
    
    Args:
        cpp_file: C++ 源文件路径
        vitis_hls_dir: Vitis HLS包含目录的路径
    
    Returns:
        stack: 以堆栈的形式存储循环的迭代次数
    """
    # cp cpp_file to ./tmp.cpp file
    # bug是只能处理./*.cpp, 其他目录的cpp无法用llvm pass提取, 待处理
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # 格式：年月日_时分秒_毫秒
    tmp_file = f"./tmp_{timestamp}.cpp"
    os.system(f"cp {cpp_file} {tmp_file}")

    # 分离文件名和扩展名
    base, ext = os.path.splitext(tmp_file)
    # 检查扩展名是否为 .c 或 .cpp
    if ext in [".cpp"]:
        ll_file = base + ".ll"
        ll_opt_file = base + "-opt.ll"
    elif ext in [".c"]:
        print("暂不支持 .c 文件。比如, AP data type can only be used in C++")
    else:
        print("输入文件不是 .c 或 .cpp 文件。")

    txt_file =  os.path.join(os.path.dirname(tmp_file), 'tripcount.txt')
    # 删除旧的tripcount.txt
    if os.path.exists(txt_file):
        os.system(f"rm -f {txt_file}")
        os.system(f"rm -f {ll_file}")
        os.system(f"rm -f {ll_opt_file}")

    # 创建tripcount.txt
    os.system(f"touch {txt_file}")
    os.system(f"chmod 777 {txt_file}")

    # vitis_hls_dir="/home/Xilinx/Vitis_HLS/2022.2/include"
    # clang++ -O1 -emit-llvm -S -fno-discard-value-names test.cpp -o test.ll -g -I"$vitis_hls_dir" -Wno-unknown-warning-option

    vitishls_compile_setting = "-I" + vitis_hls_dir

    process1 = subprocess.Popen(
    ["clang++", "-std=c++11", "-stdlib=libc++", "-O1", "-emit-llvm", "-S", "-fno-discard-value-names", tmp_file, "-o", ll_file, "-g", vitishls_compile_setting, "-Wno-unknown-warning-option"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
    )
    # Wait for the clang++ process to complete before proceeding
    stdout, stderr = process1.communicate()

    # Check for errors in the first process
    if process1.returncode != 0:
        print(f"Error in clang++: {stderr}")
    else:
        # excute the second command after the first command is successful
        # subprocess.Popen
        print(f"clang++ output: {stdout}")

        process2 = subprocess.Popen(
        ["opt", "-load", "./tripcount-pass/build/tripcount/libTripCountPass.so", "-enable-new-pm=0", "-tripcount", ll_file, "-o", ll_opt_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
        )
        print(f"run command: opt -load ./tripcount-pass/build/tripcount/libTripCountPass.so -enable-new-pm=0 -tripcount {ll_file} -o {ll_opt_file}")

        # 捕获标准输出和标准错误
        stdout, stderr = process2.communicate()

        # Check for errors in the second process
        if process2.returncode != 0:
            print(f"Error in opt: {stderr}")
        else:
            print(f"opt output: {stdout}")
   
    # 输出的.txt是纯数字
    # 逻辑在./tripcount-pass/tripcount/tripcount.cpp
    # 第一行是最内层循环的tripcount，最后一行是最外层循环的tripcount
    print(f"tripcount文件已保存到 {txt_file}")

    # clear tmp file
    os.system(f"rm -f {ll_file}")
    os.system(f"rm -f {ll_opt_file}")

    # 以堆栈的形式存储循环的迭代次数
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    stack = []
    for line in lines:
        if line == "\n":
            continue
        # if line is string , then append 16 to the stack
        # bug: while循环会干扰，待修复
        if line.strip().isdigit():
            stack.append(int(line))
        else:
            stack.append(16)

    print("tripcount:", stack)

    # clear tmp file
    os.system(f"rm -f {tmp_file}")
    os.system(f"rm -f {ll_file}")
    os.system(f"rm -f {ll_opt_file}")

    return stack

def extract_loop_line(cpp_file):
    # 创建 Clang 索引
    # clang.cindex.Config.set_library_file("/usr/lib/llvm-14/lib/libclang.so")
    index = clang.cindex.Index.create()
    
    # 解析源文件，生成翻译单元
    # 根据文件扩展名设置编译器参数
    if cpp_file.endswith('.c'):
        args = ['-std=c11']  # C 文件使用 C11 标准
    elif cpp_file.endswith('.cpp'):
        args = ['-std=c++11']  # C++ 文件使用 C++11 标准
    else:
        print(f"错误: 不支持的文件类型 {cpp_file}")
        sys.exit(1)
    
    translation_unit = index.parse(cpp_file, args=args)
    
    if not translation_unit:
        print(f"错误: 无法解析文件 {cpp_file}")
        sys.exit(1)
    
    print(f"解析文件 {cpp_file} 成功。")
    # 创建树的根节点
    root = Node(f"Translation Unit ({cpp_file})")
    
    # 提取 for 循环并构建树
    extract_for_loops(translation_unit.cursor, loop_node=root, cpp_file=cpp_file)
    
    return root

def join_tripcount(root, tripcount_stack):
    for node in PreOrderIter(root):
        if node.name.startswith('for_loop'):
            #判断stack是否空
            if tripcount_stack==[]:
                node.tripcount=16
            else:
                node.tripcount = tripcount_stack.pop()
                # bug. 可能漏掉部分循环
            print(node.tripcount)

def extract_loop_info(cpp_file, vitis_hls_dir="/tools/Xilinx/Vitis_HLS/2023.2/include"):
    if not os.path.isfile(cpp_file):
        print(f"错误: 文件 {cpp_file} 不存在。")
        sys.exit(1)
    print("extract_loop_info start")
    tripcount_stack = extract_tripcount(cpp_file, vitis_hls_dir)
    print("extract_tripcount done")

    print("extract_loop_line start")

    root = extract_loop_line(cpp_file)
    print("extract_loop_line done")
    # 打印树结构
    print("树状结构:")
    print_tree(root)

    join_tripcount(root, tripcount_stack)

    # 导出为 JSON
    output_dir = os.path.join(os.path.dirname(cpp_file), 'loop_info')
    os.makedirs(output_dir, exist_ok=True)
    output_json = os.path.join(output_dir, 'for_loops_tree.json')
    save_tree_to_json(root, output_json)
    print("extract_loop_info done")
    return root

def extract_loop_info_in_dir(search_dir, vitis_hls_dir="/tools/Xilinx/Vitis_HLS/2023.2/include"):
    cpp_c_files = find_cpp_c_files(search_dir)
    for cpp_file in cpp_c_files:
        extract_loop_info(cpp_file, vitis_hls_dir)

if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='提取C/C++文件中的循环信息')
    
    # 添加互斥组，用户必须指定--file或--dir之一
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', help='要处理的C/C++文件路径')
    group.add_argument('--dir', help='包含C/C++文件的目录路径')
    
    parser.add_argument('--vitis-hls-dir', default='/tools/Xilinx/Vitis_HLS/2023.2/include',
                        help='Vitis HLS包含目录的路径 (默认: /tools/Xilinx/Vitis_HLS/2023.2/include)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 根据参数处理文件或目录
    if args.file:
        if os.path.isfile(args.file):
            extract_loop_info(args.file, args.vitis_hls_dir)
        else:
            print(f"错误: 文件 {args.file} 不存在。")
            sys.exit(1)
    elif args.dir:
        if os.path.isdir(args.dir):
            extract_loop_info_in_dir(args.dir, args.vitis_hls_dir)
        else:
            print(f"错误: 目录 {args.dir} 不存在。")
            sys.exit(1)