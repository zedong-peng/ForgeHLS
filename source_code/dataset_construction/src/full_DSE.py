import json
import os
import shutil
import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional
from anytree import Node, RenderTree, PreOrderIter

from extract_loop_info import extract_loop_info
# from get_function_with_body_line_number import get_function_with_body_line_number

from pathlib import Path

from tqdm import tqdm
# from datetime import datetime

from concurrent.futures import ThreadPoolExecutor, as_completed


# import sympy

design_count = 0
stop_count = 0

# datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
# # 配置日志
# logging.basicConfig(
#     level=logging.INFO,  # 设置日志级别
#     format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
#     filename=f'./logs/pragma_design_{datetime}.log',  # 日志文件路径
#     filemode='a'  # 文件模式为追加模式（'a'）
# )

@dataclass
class Pragma:
    Type: str
    Options: str
    line: int  # 1-based 行号
    #parentFunction: str
    file_path: str  # 目标文件路径，相对于设计目录
    factor: Optional[int] = field(default=None) 
    loop_node: Optional[Node] = field(default=None)


def insert_pragma_into_cpp(file_path: str, pragma_statement: str, target_line_number: int):
    try:
        with open(file_path, 'r', encoding='utf-8') as file_handle:
            file_lines = file_handle.readlines()

        # 将1-based行号转换为0-based索引
        index = target_line_number - 1

        # 插入pragma
        if 0 <= index < len(file_lines):
            file_lines.insert(index, pragma_statement)
            # logging.info(f"Inserted pragma '{pragma_statement.strip()}' at line {target_line_number} in {file_path}")
        else:
            # 如果行号超出范围，添加到文件末尾
            file_lines.append(pragma_statement)
            logging.warning(f"Line number {target_line_number} out of range. Appended pragma '{pragma_statement.strip()}' to end of {file_path}")

        with open(file_path, 'w', encoding='utf-8') as file_handle:
            file_handle.writelines(file_lines)
    except Exception as error:
        logging.error(f"Error inserting pragma into {file_path}: {error}")
        raise error

def traverse_tree_and_collect_designs(node: Node, pragma_design_list: List[List[Pragma]]):
    """
    遍历树并收集设计
    
    """
    # node.name != "Root":  # Skip root node
    # all node on the path on root node to leaf node, add to design
    if node.is_leaf:
        design = []
        for parent in node.path:
            if parent.pragma:
                design.append(parent.pragma)
                # design exmaple:
                # [pragma1, pragma2, pragma3, pragma4]
        pragma_design_list.append(design)
        # pragma_desing_list example:
        # [[pragma1, pragma2, pragma3, pragma4], [pragma1, pragma2, pragma3, pragma4], [pragma1, pragma2, pragma3, pragma4]]
    else:
        for child in node.children:
            traverse_tree_and_collect_designs(child, pragma_design_list)

def display_pragma_tree(root_node: Node):
    for pre, _, node in RenderTree(root_node):
        pragma = node.pragma
        if pragma:
            print(f"{pre}{pragma.Type} {pragma.Options}, Line: {pragma.line}")
        else:
            print(f"{pre}Root")

def apply_pragmas_to_design_cpp(current_pragma_design: List[Pragma], design_cpp_path: str):
    """
    将pragma应用到C++文件中

    current_pragma_design example:
    [pragma1, pragma2, pragma3, pragma4]
    pragma example:
    Pragma("HLS UNROLL", "factor=2", 10, "kernel.cpp")
    """    
    # 按行号降序排序pragmas, 防止插入时行号错位
    sorted_pragmas = sorted(current_pragma_design, key=lambda p: p.line, reverse=True)

    for pragma in sorted_pragmas:
        # 插入pragma到C++文件中
        insert_pragma_into_cpp(
            design_cpp_path,
            f'#pragma {pragma.Type} {pragma.Options}\n',
            pragma.line
        )

def save_pragma_tree(root_node: Node, pragma_tree_txt_path: str):
    with open(pragma_tree_txt_path, 'a') as file:
        for pre, _, node in RenderTree(root_node):
            pragma = node.pragma
            if pragma:
                if pragma.Type == "HLS UNROLL":
                    file.write(f"{pre}{pragma.Type} {pragma.Options}, Line:{pragma.line}\n")
                else:
                    file.write(f"{pre}{pragma.Type} {pragma.Options}, Line:{pragma.line}\n")
            else:
                file.write(f"{pre}Root\n")

def generate_factors(upbar, num=5):
    if upbar <16:
        factors = [-1]  # 初始化，包含 -1 表示 complete
    else:
        factors = []

    # powers_of_two = [2**i for i in range(upbar.bit_length()) if 2**i <= (upbar/2)] # 上限设置upbar/2因为完全展开的情况和upbar是一样的
    # # 均匀采样 控制 factors 的数量。
    # if len(powers_of_two) > num - 1: 
    #     step = math.ceil(len(powers_of_two) / (num - 1))
    #     powers_of_two = powers_of_two[::step]
    #     factors += powers_of_two

    # 如果upbar过大 factors上限至多为16，不能太大
    for i in range(0, num):
        if 2**i <= (upbar):
            factors.append(2**i)
        else:
            break

    # factors.append(sympy.divisors(upbar)) # 除数

    return sorted(set(factors))  # 去重并排序

def full_dse(cpp_path: str, source_designs_dir: str):
    """
    cpp_path = "../data/3_loop_3_array/test.cpp"
    save_design_path = "../data/designs/"
    """
    global design_count
    design_count = 0
    # algo_name = 上层文件夹名
    algo_name = os.path.basename(os.path.dirname(cpp_path))

    ##########################################
    # 构建pragma树
    root_node = Node("Root")
    root_node.pragma = None
    root_node.loop_node = None
    ############################################################################################################
    #############  loop pragma ################
    print("loop design start")
    loop_root_node = extract_loop_info(cpp_path)
    count_loop=0
    print("loop_root_node success extract")

    # 遍历树，计算所有节点的数量
    def count_nodes(node):
        return len([_ for _ in node.descendants]) + 1  # 包括根节点本身
    node_count = count_nodes(loop_root_node)
    if node_count > 4:
        logging.warning(f"Loop count exceeds 4, stopping loop pragmas generation for {cpp_path}")
        return

    for loop_node in PreOrderIter(loop_root_node):
        count_loop+=1
        if count_loop>4:
            return
        if loop_node.is_root:
            continue
        leaf_nodes = [node for node in PreOrderIter(root_node) if node.is_leaf]
        for leaf_node in leaf_nodes:
            factors = generate_factors(loop_node.tripcount)
            for factor in factors:
                line = loop_node.line
                line += 1
                unroll_node = Node(name="UNROLL", parent=leaf_node)
                # try:
                #     if unroll_node.parent.pragma.Type == "HLS UNROLL" and factor <= unroll_node.parent.pragma.factor:
                #         continue
                # except AttributeError:
                #     pass  # Handle the case where any of the attributes are missing
                if factor == -1:
                    unroll_node.pragma = Pragma("HLS UNROLL", f"", line, '', -1, loop_node)
                else:
                    unroll_node.pragma = Pragma("HLS UNROLL", f"factor={factor}", line, '', factor, loop_node)
                
                np_node = Node(name="PIPELINE OFF", parent=unroll_node)
                np_node.pragma = Pragma("HLS PIPELINE OFF", "", line, '', None, loop_node)

                if factor != -1: # 完全展开与pipeline冲突
                    # if loop_node dont have a child
                    if not loop_node.children:  # Assuming `loop_node` has a `children` attribute
                        p_node = Node(name="PIPELINE", parent=unroll_node)
                        p_node.pragma = Pragma("HLS PIPELINE", "", line, '', None, loop_node)

    # 外层pipeline后内层循环不unroll
    # 前序遍历节点
    # 如果某个nodeA.pragma的type是HLS UNROLL，那么遍历其父节点
    # 如果存在同一个nestedloop的HLS PIPELINE, 那么从这棵树中删除这个nodeA
    for node in PreOrderIter(root_node):
        if node.pragma and node.pragma.Type == "HLS UNROLL":
            for parent_node in node.path:
                if parent_node.pragma and parent_node.pragma.Type == "HLS PIPELINE" and parent_node.pragma.loop_node in node.pragma.loop_node.path:
                    node.parent = None
                    break

    # # (optional) 针对大型设计，仅在最内层循环添加pipeline
    # # 如果一个nested loop有俩个以上的level，对外层level的pipeline剪枝
    # for node in PreOrderIter(root_node):
    #     if node.pragma and node.pragma.Type == "HLS PIPELINE":
    #         for parent_node in node.path:
    #             if parent_node.pragma and parent_node.pragma.Type == "HLS PIPELINE" and parent_node.pragma.loop_node in node.pragma.loop_node.path:
    #                 node.parent = None
    #                 break

    # (optional) 针对大型设计 归纳总结规则，并验证，考虑pipeline的因素，尽量靠近
    # (optional) 针对大型设计 最后保留的策略是随机采样
    
    # 避免外层unroll过大内层unroll过小的情况
    # 遍历从当前节点到根节点的所有父节点
    for node in PreOrderIter(root_node):
        try:
            if node.pragma and node.pragma.Type == "HLS UNROLL":
                current_node = node
                # 向上遍历父节点直到根节点
                while current_node.parent:
                    parent_node = current_node.parent
                    
                    # 检查父节点的条件
                    if parent_node.pragma and parent_node.pragma.Type == "HLS UNROLL":
                        if node.pragma.factor < parent_node.pragma.factor:
                            # print(f"Unroll factor of {node.pragma.loop_node.name} is greater than or equal to that of {parent_node.pragma.loop_node.name}")
                            node.parent = None
                            break  # 停止进一步检查父节点，跳出循环
                    # 继续遍历更高一级的父节点
                    current_node = parent_node
        except AttributeError:
            # 如果某个节点没有父节点或其他属性为空，则跳过
            continue

 
    # # 检查node叶子节点数量，如果超过100个，直接返回    
    # leaf_nodes = [node for node in PreOrderIter(root_node) if node.is_leaf]
    # if len(leaf_nodes) > 100:
    #     logging.warning(f"More than 100 leaf nodes for {cpp_path}")
    #     return

    print("loop design success")
    ############################################################################################################
    #############  array pragma ################    
    ##########################################
    # array info
    array_info_path = os.path.join(os.path.dirname(cpp_path), "array_info", "array_info.json")
    try:
        with open(array_info_path, 'r') as file:
            array_information = json.load(file)
    except FileNotFoundError:
        logging.warning(f"No array information found in {array_info_path}")
        array_information = []

    array_count = 0
    for array in array_information:
        for dim in range(array['dimensions']):
            array_count = array_count + 1

    # top_function_line = get_function_with_body_line_number(cpp_path)
    # if top_function_line == None:
    #     return None
    # print(f"top_function_line: {top_function_line}")
    # return
    if array_count > 3:
        logging.warning(f"array count exceeds 4, stopping loop pragmas generation for {cpp_path}")
        return
        # # set the same factor for all array partition pragma
        # tmp_node = None
        # leaf_nodes = [node for node in PreOrderIter(root_node) if node.is_leaf]
        # for leaf_node in leaf_nodes:
        #     try:
        #         factors = generate_factors(array_information[0]['size_per_dimension'][0])
        #     except (IndexError, KeyError, TypeError):
        #         factors = generate_factors(16)
        #     for factor in factors:
        #         for array in array_information:
        #             for dim in range(array['dimensions']):
        #                 # line = array['line_per_dimension'][0] + 2 # 防止换行大括号
        #                 line = top_function_line + 1
        #                 if array == array_information[0] and dim == 0:
        #                     node = Node(name="ARRAY_PARTITION", parent=leaf_node)
        #                 else:
        #                     node = Node(name="ARRAY_PARTITION", parent=tmp_node)
        #                     factor = node.parent.pragma.factor

        #                 current_dim = dim + 1 # 1-based 维度

        #                 if factor == -1:
        #                     node.pragma = Pragma("HLS ARRAY_PARTITION", f"variable={array['array_name']} type=complete dim={current_dim}", line, '', -1, None)
        #                 else:
        #                     node.pragma = Pragma("HLS ARRAY_PARTITION", f"variable={array['array_name']} type=cyclic dim={current_dim} factor={factor}", line, '', factor, None)

        #                 tmp_node = node

    # 如果数组维度小于等于3，直接在叶子节点添加array partition pragma
    if array_count <= 3:
    # if array_count > 0:
        for array in array_information:
            for dim in range(array['dimensions']):
                line = array['size_per_dimension'][dim] + 1 #  +2可以防止换行大括号
                # line = top_function_line + 1
                current_dim = dim + 1 # 1-based 维度
                leaf_nodes = [node for node in PreOrderIter(root_node) if node.is_leaf]
                for leaf_node in leaf_nodes:
                    try:
                        factors = generate_factors(array['size_per_dimension'][dim])
                    except (IndexError, KeyError, TypeError):
                        factors = generate_factors(16)
                    for factor in factors:
                        node = Node(name="ARRAY_PARTITION", parent=leaf_node)
                        if factor == -1:
                            node.pragma = Pragma("HLS ARRAY_PARTITION", f"variable={array['array_name']} type=complete dim={current_dim}", line, '', -1, None)
                        else:
                            node.pragma = Pragma("HLS ARRAY_PARTITION", f"variable={array['array_name']} type=cyclic dim={current_dim} factor={factor}", line, '', factor, None)

    # for i in arange(0,3,1):
    #     leaf_nodes = [node for node in PreOrderIter(root_node) if node.is_leaf]
    #     for leaf_node in leaf_nodes:
    #         node = Node(name="INLINE", parent=leaf_node)
    #         node.pragma = Pragma("HLS INLINE", "", top_function_line, '')
    #         node = Node(name="INLINE", parent=leaf_node)
    #         node.pragma = Pragma("HLS INLINE", "recursive", top_function_line, '')
    #         node = Node(name="INLINE", parent=leaf_node)
    #         node.pragma = Pragma("HLS INLINE", "off", top_function_line, '')

    # for i in arange(0,2,1):
    #     leaf_nodes = [node for node in PreOrderIter(root_node) if node.is_leaf]
    #     for leaf_node in leaf_nodes:
    #         node = Node(name="FUNCTION PIPELINE", parent=leaf_node)
    #         node.pragma = Pragma("HLS PIPELINE", "", top_function_line, '')
    #         node = Node(name="FUNCTION PIPELINE", parent=leaf_node)
    #         node.pragma = Pragma("HLS PIPELINE", "off", top_function_line, '')

    # 叶子节点到根节点的路径上的pragmas构成一个设计
    pragma_design_list = []
    traverse_tree_and_collect_designs(root_node, pragma_design_list)

    # 打印pragma树结构
    pragma_tree_txt_path = os.path.join(os.path.dirname(cpp_path), "pragma_tree.txt")
    with open(pragma_tree_txt_path, 'w') as file:
        file.write(f"Design count: {design_count}\nPragma tree:\n")
    save_pragma_tree(root_node, pragma_tree_txt_path)
    print("Design count and pragma tree saved at:", pragma_tree_txt_path)

    # # for test, only run the code here
    # return

    # 清理目标目录，重新生成设计
    if os.path.exists(f"{source_designs_dir}/{algo_name}"):
        shutil.rmtree(f"{source_designs_dir}/{algo_name}")

    def copy_kernel_to_design(cpp_path, current_pragma_design, destination_design_path):
        # os.makedirs(destination_design_path, exist_ok=True)

        shutil.copytree(Path(cpp_path).parent, destination_design_path, dirs_exist_ok=True)
        # shutil.rmtree(os.path.join(destination_design_path, "project"), ignore_errors=True) # delete project folder in destination_design_path 防止hls生成的.c干扰后续tcl文件的生成
        # shutil.rmtree(os.path.join(destination_design_path, "pragma_tree.txt"), ignore_errors=True)
        # shutil.rmtree(os.path.join(destination_design_path, "tripcount.txt"), ignore_errors=True)
        # shutil.rmtree(os.path.join(destination_design_path, "array_info"), ignore_errors=True)
        # shutil.rmtree(os.path.join(destination_design_path, "loop_info"), ignore_errors=True)

        design_cpp_path = os.path.join(destination_design_path, os.path.basename(cpp_path))
        apply_pragmas_to_design_cpp(current_pragma_design, design_cpp_path)

    # 删除cpp_path同目录下的多余文件 避免复制的时候多次删除
    # 删除 project 目录（如果存在）
    project_dir = os.path.join(os.path.dirname(cpp_path), "project")
    shutil.rmtree(project_dir, ignore_errors=True)  # 删除 project 文件夹

    # shutil.rmtree(os.path.join(os.path.dirname(cpp_path), "array_info"), ignore_errors=True)
    # shutil.rmtree(os.path.join(os.path.dirname(cpp_path), "loop_info"), ignore_errors=True)

    # design_0: origin design
    destination_design_path = f"{source_designs_dir}/{algo_name}/design_{design_count}"
    # copy_kernel_to_design(cpp_path, [], destination_design_path)
    # design_count += 1


    # # 提前保存可能的pareto designs
    # pareto_design_list = []
    # for current_pragma_design in pragma_design_list:
    #     # 所有HLS ARRAY_PARTITION 和 HLS UNROLL 的factor相同的设计可能是pareto designs
    #     # 保存到 pareto_design_list
    #     pareto_design = []
    #     is_pareto_design = True
    #     tmp_factor = -2
    #     for pragma in current_pragma_design:
    #         if pragma.Type == "HLS ARRAY_PARTITION" or pragma.Type == "HLS UNROLL":
    #             if tmp_factor == -2:
    #                 tmp_factor = pragma.factor
    #             elif tmp_factor != pragma.factor:
    #                 is_pareto_design = False
    #                 break
    #     if is_pareto_design:
    #         pareto_design_list.append(current_pragma_design)
    # # pareto designs
    # for pareto_design in pareto_design_list:
    #     destination_design_path = f"{source_designs_dir}/{algo_name}/design_{design_count}"
    #     copy_kernel_to_design(cpp_path, pareto_design, destination_design_path)
    #     design_count += 1
    # # if a design in pareto_design_list, remove it from pragma_design_list
    # for pareto_design in pragma_design_list:
    #     if pareto_design in pareto_design_list:
    #         pragma_design_list.remove(pareto_design)   
    
    total_design_count = 0
    for current_pragma_design in tqdm(pragma_design_list, position=1, desc=f"Pragma Design for {cpp_path}", leave=False):
        total_design_count += 1

    for current_pragma_design in tqdm(pragma_design_list, position=1, desc=f"Pragma Design for {cpp_path}", leave=False):

        # random
        import random
        random_float = random.random()
        if random_float < 3000/total_design_count:
            destination_design_path = f"{source_designs_dir}/{algo_name}/design_{design_count}"
            copy_kernel_to_design(cpp_path, current_pragma_design, destination_design_path)
            design_count += 1
        
        # # full
        # destination_design_path = f"{source_designs_dir}/{algo_name}/design_{design_count}"
        # copy_kernel_to_design(cpp_path, current_pragma_design, destination_design_path)
        # design_count += 1

        # # top100
        # if design_count > 100: # too many designs, return
        #     logging.warning(f"More than 500 designs for {cpp_path}")
        #     return

    with open(pragma_tree_txt_path, 'a') as file:
        file.write(f"\nDesign count: {design_count}\n")

def pragma_design(cpp_path_str, source_designs_dir):
    print(f"Running pragma_design for {cpp_path_str}...")
    cpp_path = Path(cpp_path_str)
    # 检查 cpp_path 是否存在且为文件
    if not cpp_path.is_file():
        print(f"警告: {cpp_path} 不是一个有效的文件路径。")
        return
    
    # 获取 cpp_path 所在的目录
    parent_dir = cpp_path.parent
    # 定义 array_info 文件夹路径
    array_info_dir = parent_dir / 'array_info'
    
    # 检查 array_info 文件夹是否存在
    if not array_info_dir.exists():
        print(f"{array_info_dir} 不存在")
        return
        # extract_array_info(cpp_path)
    else:
        # overwrite array info 
        # extract_array_info(cpp_path)
        # print(f"{array_info_dir} 已存在，覆盖提取数组信息。")

        # not overwrite array info
        print(f"{array_info_dir} 已存在，跳过提取。")
    
    # full dse
    full_dse(cpp_path_str, source_designs_dir)

    # # bayesian dse
    # from pragma_design_bayesian import pragma_design_bayesian
    # print(f"Running pragma_design_bayesian for {cpp_path_str}")
    # print(f"cpp_path_str: {cpp_path_str}")
    # print(f"source_designs_dir: {source_designs_dir}")
    # pragma_design_bayesian(cpp_path_str, source_designs_dir)

def pragma_design_in_dir(search_dir: str, source_designs_dir: str, max_workers=32):
    # 设置最大深度为1，表示只搜索search_dir及其直接子目录
    max_depth = 2

    cpp_path_list = []

    for root, dirs, files in os.walk(search_dir):
        # 计算当前目录的深度
        rel_path = os.path.relpath(root, search_dir)
        if rel_path == ".":
            depth = 0
        else:
            depth = rel_path.count(os.sep) + 1  # +1 因为相对路径不包含根目录的分隔符

        if depth >= max_depth:
            # 达到最大深度，不再进一步递归
            dirs[:] = []

        for file in files:
            if file.endswith(".cpp"):
                cpp_path_list.append(os.path.join(root, file))
            if file.endswith(".c"):
                cpp_path_list.append(os.path.join(root, file))

    print(f"cpp_path_list: {cpp_path_list}")
    # 调用并行处理函数
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务到线程池
        futures = {
            executor.submit(pragma_design, cpp_path_str, source_designs_dir): cpp_path_str
            for cpp_path_str in cpp_path_list
        }
        
        # 使用 tqdm 显示进度
        for future in tqdm(as_completed(futures), total=len(cpp_path_list), position=0, desc="Processing C++ files"):
            try:
                future.result()  # 获取任务结果（如果有异常会抛出）
            except Exception as e:
                print(f"处理文件时发生错误: {e}")

if __name__ == "__main__":
    # pragma_design_in_dir("/data/HLSBatchProcessor/data/kernels/ai_fpga_hls_algorithms/agglomerative_clustering", "/data/HLSBatchProcessor/data/raw_designs/test")
    pragma_design_in_dir("../../benchmark/CHStone-flatten/", "../data/designs/CHStone/")
    # from pragma_design_bayesian import pragma_design_bayesian
    # pragma_design_bayesian("../../benchmark/PolyBench-flatten/jacobi-1d/jacobi-1d.c", "../data/designs/PolyBench/")


