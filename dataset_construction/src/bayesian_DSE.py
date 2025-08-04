import concurrent.futures
import os
import subprocess
import json
from skopt import gp_minimize
from skopt.space import Integer, Categorical
import numpy as np
import shutil
import logging
import traceback
from tqdm import tqdm  # 导入 tqdm
from datetime import datetime
import threading
import signal
import sys
import time
import argparse  # 添加这一行导入

import xml.etree.ElementTree as ET
from pathlib import Path

from full_DSE import apply_pragmas_to_design_cpp, Pragma
# from get_function_with_body_line_number import get_function_with_body_line_number
from find_cpp_c_files import find_cpp_c_files

# 创建一个全局线程锁
log_lock = threading.Lock()

# 添加全局变量跟踪所有进程
all_processes = set()
executor_shutdown_event = threading.Event()

def log_safe(message, level=logging.INFO):
    with log_lock:
        if level == logging.INFO:
            logging.info(message)
        elif level == logging.ERROR:
            logging.error(message)
        # 将信息同时打印到控制台（方便调试）
        print(f"[{logging.getLevelName(level)}] {message}")

# 1. 提取性能指标
def gather_csynth_data(root_dir):
    """提取HLS综合报告中的性能和资源数据"""
    # 将路径转换为 Path 对象，便于处理
    root_dir = Path(root_dir)
    
    try:
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file == 'csynth.xml':
                    file_path = os.path.join(root, file)
                    rpt_file_path = os.path.join(root, 'csynth.rpt')
                    
                    # 检查是否存在 csynth.rpt 文件
                    if not os.path.exists(rpt_file_path):
                        log_safe(f"[INFO] 缺少 csynth.rpt 文件，跳过: {file_path}")
                        continue
            
                    try:
                        # 解析 XML 文件
                        tree = ET.parse(file_path)
                        root = tree.getroot()

                        # 提取所需信息，增加错误检查
                        data = {}
                        xml_paths = {
                            'part': './/UserAssignments/Part',
                            'target_clock_period': './/UserAssignments/TargetClockPeriod',
                            'best_case_latency': './/PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency',
                            'worst_case_latency': './/PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency',
                            'bram_18k': './/AreaEstimates/Resources/BRAM_18K',
                            'lut': './/AreaEstimates/Resources/LUT',
                            'dsp': './/AreaEstimates/Resources/DSP',
                            'ff': './/AreaEstimates/Resources/FF'
                        }
                        
                        for key, path in xml_paths.items():
                            element = root.find(path)
                            if element is None:
                                log_safe(f"XML路径未找到: {path}", level=logging.ERROR)
                                return None
                            data[key] = element.text
                            
                        log_safe(f"    data: {data}")
                        return data
                    except ET.ParseError as e: 
                        log_safe(f"    XML 解析错误: {file_path}, 错误信息: {str(e)}", level=logging.ERROR)
                    except AttributeError as e:
                        log_safe(f"    缺少必要的 XML 元素: {file_path}, 错误信息: {str(e)}", level=logging.ERROR)
        
        log_safe(f"未找到csynth.xml文件: {root_dir}", level=logging.ERROR)
        return None
    except Exception as e:
        log_safe(f"gather_csynth_data发生未预期异常: {str(e)}", level=logging.ERROR)
        log_safe(traceback.format_exc(), level=logging.ERROR)
        return None

# 2. 生成HLS的pragma指令并插入到C++代码
def generate_pragma(cpp_path, config, array_information, loop_information):
    """
    根据优化配置生成pragma指令并插入到C++代码
    """
    current_pragma_design = []
    i = 0

    # 添加数组分区pragma
    for array in array_information:
        for dim in range(array['dimensions']):
            try:
                line = array['line_per_dimension'][0] + 1  # +1可以防止换行大括号
                current_dim = dim + 1  # 1-based 维度
                current_pragma_design.append(
                    Pragma("HLS ARRAY_PARTITION", 
                           f"variable={array['array_name']} type=cyclic dim={current_dim} factor={config[i]}", 
                           line, '', config[i], None)
                )
                i += 1
            except (KeyError, IndexError) as e:
                log_safe(f"生成数组pragma时出错: {e}, 数组信息: {array}", level=logging.ERROR)
                continue

    # 递归添加循环优化pragma
    def traverse_loop_tree(loop_info, current_pragmas, index):
        if "children" not in loop_info:
            return index
        
        for child in loop_info["children"]:
            try:
                line = child["line"] + 1
                # 添加展开指令
                current_pragmas.append(
                    Pragma("HLS UNROLL", f"factor={config[index]}", line, '', config[index], None)
                )
                index += 1
                
                # 添加流水线指令
                current_pragmas.append(
                    Pragma("HLS PIPELINE", f"{config[index]}", line, '', config[index], None)
                )
                index += 1
                
                # 递归处理子循环
                index = traverse_loop_tree(child, current_pragmas, index)
            except (KeyError, IndexError) as e:
                log_safe(f"生成循环pragma时出错: {e}, 循环信息: {child}", level=logging.ERROR)
                # 跳过这个循环但继续处理其他循环
                continue
                
        return index

    # 开始循环树遍历
    traverse_loop_tree(loop_information, current_pragma_design, i)
    
    # 将pragma应用到代码中
    try:
        apply_pragmas_to_design_cpp(current_pragma_design, cpp_path)
        log_safe(f"成功应用{len(current_pragma_design)}个pragma到{cpp_path}")
    except Exception as e:
        log_safe(f"应用pragma到{cpp_path}时出错: {e}", level=logging.ERROR)
        log_safe(traceback.format_exc(), level=logging.ERROR)

# 3. 目标函数：评估HLS设计的性能
def objective_function(config, cpp_file_path, path_to_data, array_information, loop_information):
    """
    根据配置生成HLS代码，调用Vitis HLS综合，并提取性能指标
    """
    try:
        # 创建保存设计的目录
        algo_name = os.path.basename(os.path.dirname(cpp_file_path))
        source_name = os.path.basename(os.path.dirname(os.path.dirname(cpp_file_path)))
        design_base_path = os.path.join(path_to_data, "designs", source_name, algo_name)
        
        # 确保目录存在
        os.makedirs(design_base_path, exist_ok=True)
        
        # 找到可用的设计编号
        count = 0
        while True:
            save_design_path = os.path.join(design_base_path, f"design_{count}")
            if not os.path.exists(save_design_path):
                break
            count += 1
        
        os.makedirs(save_design_path)
        log_safe(f"    Running objective_function for {save_design_path}")
        log_safe(f"    config: {config}")

        # 复制源代码文件到设计目录
        log_safe(f"    Copying Vitis HLS source code...")
        base_dir = os.path.dirname(cpp_file_path)
        for file in os.listdir(base_dir):
            file_path = os.path.join(base_dir, file)
            if os.path.isfile(file_path) and file.endswith(('.c', '.cpp', '.h', '.hpp', '.tcl', '.txt')):
                shutil.copy(file_path, save_design_path)

        # 生成pragma并应用到代码
        cpp_dest_path = os.path.join(save_design_path, os.path.basename(cpp_file_path))
        log_safe(f"    Running generate_pragma...")
        generate_pragma(cpp_dest_path, config, array_information, loop_information)

        # 调用Vitis HLS进行综合，添加超时机制
        log_safe(f"    Running vitis_hls...")
        with open(os.path.join(save_design_path, "vitis_hls_output.log"), 'w') as log_file:
            # 创建进程并设置超时
            process = subprocess.Popen(
                ["vitis_hls", "-f", "run_hls.tcl"], 
                cwd=save_design_path, 
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # 记录进程以便清理
            all_processes.add(process)
            
            try:
                # 设置30分钟超时
                result = process.wait(timeout=1800)
                if result != 0:
                    log_safe(f"    Error when running vitis_hls: {save_design_path}", level=logging.ERROR)
                    # 提取错误日志
                    try:
                        with open(os.path.join(save_design_path, "vitis_hls.log"), 'r') as file:
                            error_lines = [line for line in file if "ERROR" in line]
                            if error_lines:
                                log_safe(f"    Vitis HLS 错误信息:", level=logging.ERROR)
                                for line in error_lines[:10]:  # 限制只显示前10行错误
                                    log_safe(f"      {line.strip()}", level=logging.ERROR)
                    except Exception as e:
                        log_safe(f"    无法读取错误日志: {e}", level=logging.ERROR)
                    
                    return float('inf')  # 返回一个高代价表示失败
            except subprocess.TimeoutExpired:
                log_safe(f"    Vitis HLS process timed out after 30 minutes, killing process", level=logging.ERROR)
                process.kill()
                return float('inf')
            finally:
                # 从跟踪集合中移除进程
                if process in all_processes:
                    all_processes.remove(process)
        
        log_safe(f"    vitis_hls completed")

        # 提取时延和资源占用
        data = gather_csynth_data(save_design_path)
        if data is None:
            log_safe(f"    Error when running gather_csynth_data({save_design_path})", level=logging.ERROR)
            return float('inf')  # 返回一个高代价表示失败
        
        try: 
            latency = int(data['best_case_latency'])  # 时延
        except ValueError:
            log_safe(f"    Latency of {save_design_path} is undefined.", level=logging.ERROR)
            return float('inf')

        # 计算资源占用比例
        available_resources = {
            'BRAM_18K': 4032,
            'LUT': 1303680,
            'DSP': 9024,
            'FF': 2607360
        }
        resource = (int(data['bram_18k'])/available_resources['BRAM_18K'] + int(data['lut'])/available_resources['LUT'] + int(data['dsp'])/available_resources['DSP'] + int(data['ff'])/available_resources['FF'])/4

        log_safe(f"    latency: {latency}, resource: {resource:.6f}")

        # 计算代价函数：优化时延和资源使用的平衡
        cost = np.sqrt((np.log10(latency) - 0)**2 + (np.log10(max(1e-6, resource)) - (-6))**2)
        log_safe(f"    cost: {cost}")
        
        # 将结果保存为JSON文件以便后续分析
        with open(os.path.join(save_design_path, "optimization_result.json"), 'w') as f:
            json.dump({
                "config": [int(c) if isinstance(c, (int, np.integer)) else c for c in config],
                "latency": latency,
                "resource": resource,
                "cost": cost,
                "resource_details": {
                    "bram_18k": data['bram_18k'],
                    "lut": data['lut'],
                    "dsp": data['dsp'],
                    "ff": data['ff']
                }
            }, f, indent=2)
            
        return cost  # 最小化目标
        
    except Exception as e:
        log_safe(f"objective_function发生未预期异常: {str(e)}", level=logging.ERROR)
        log_safe(traceback.format_exc(), level=logging.ERROR)
        return float('inf')  # 出错时返回无穷大代价

# 4. 贝叶斯优化：运行优化
def run_bayesian_optimization(bounds, cpp_file_path, path_to_data, array_information, loop_information):
    """
    运行贝叶斯优化，找到最优配置。
    """
    # 使用 skopt 库的 gp_minimize 进行贝叶斯优化
    log_safe(f"  Running Bayesian optimization for {cpp_file_path}")
    result = gp_minimize(
        lambda config: objective_function(config, cpp_file_path, path_to_data, array_information, loop_information),  # 目标函数
        dimensions=bounds,          # 参数空间的边界
        acq_func="EI",              # 采集函数：Expected Improvement
        n_jobs=-1,                  # 并行数
        n_calls=40,                 # 迭代次数
        n_random_starts=20          # 初始采样点数量
    )
    log_safe(f"  Optimization completed. Best configuration: {result.x}")
    log_safe(f"  Best cost: {result.fun}")
    return result

# 日志配置函数
def setup_logging(cpp_path):
    time = datetime.now().strftime('%Y%m%d-%H%M%S')
    file_name = os.path.basename(cpp_path)
    # algo name is the folder name of cpp_path
    algo_name = os.path.basename(os.path.dirname(cpp_path))
    # source name is the folder name of algo name
    source_name = os.path.basename(os.path.dirname(os.path.dirname(cpp_path)))
    log_dir = './logs'
    log_file_path = os.path.join(log_dir, f'pragma_design_bayesian_{source_name}-{algo_name}-{file_name}_{time}.log')

    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"create log folder: {log_dir}")

    try:
        logging.basicConfig(
            level=logging.INFO,
            filename=log_file_path,  # 日志文件
            filemode='a',        # 追加写入
            format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
        )
        print(f"Logging to {log_file_path}")  # 确保打印正确的日志文件路径

        # 测试日志是否可以写入
        log_safe(f"Logging for pragma_design_bayesian({cpp_path})")

    except Exception as e:
        print(f"无法配置日志: {e}")
        return False
    return True

# 添加信号处理函数
def signal_handler(sig, frame):
    log_safe("接收到终止信号，正在清理资源...", level=logging.WARNING)
    executor_shutdown_event.set()
    
    # 尝试终止所有子进程
    for process in list(all_processes):
        try:
            if process.is_alive():
                log_safe(f"正在终止进程 {process.pid}", level=logging.WARNING)
                process.terminate()
        except Exception as e:
            log_safe(f"终止进程时出错: {e}", level=logging.ERROR)
    
    # 等待一段时间让进程正常终止
    time.sleep(2)
    
    # 强制终止仍在运行的进程
    for process in list(all_processes):
        try:
            if process.is_alive():
                log_safe(f"强制终止进程 {process.pid}", level=logging.WARNING)
                process.kill()
        except Exception as e:
            log_safe(f"强制终止进程时出错: {e}", level=logging.ERROR)
    
    log_safe("资源清理完成，正在退出...", level=logging.WARNING)
    sys.exit(0)

# 5. 主函数
def pragma_design_bayesian(cpp_path, path_to_data, bayesian_opt_number=25):
    """
    cpp_path = "./dse/3_loop_3_array/test.cpp"
    path_to_data = "./dse/designs/"
    """
    bounds = []

    # 设置日志记录
    if not setup_logging(cpp_path):
        print("日志配置失败，退出")
        return None

    # array info
    array_info_path = os.path.join(os.path.dirname(cpp_path), "array_info", "array_info.json")
    try:
        with open(array_info_path, 'r') as file:
            array_information = json.load(file)
            
        for array in array_information:
            for dim in range(array['dimensions']):
                bounds.append(Categorical([1,1,1,1,2,2,2,4,4,8,16]))

    except FileNotFoundError:
        log_safe(f"No array information found in {array_info_path}", level=logging.WARNING)
        array_information = []
    except Exception as e:
        log_safe(f"Error when loading array information from {array_info_path}: {e}", level=logging.ERROR)
        log_safe(traceback.format_exc(), level=logging.ERROR)
        array_information = []

    # log_safe(f"Array bounds: {bounds}")

    # loop info 
    loop_info_path = os.path.join(os.path.dirname(cpp_path), "loop_info", "for_loops_tree.json")
    with open(loop_info_path, 'r') as file:
        loop_information = json.load(file)

    if not loop_information:
        log_safe(f"No loop information found in {loop_info_path}", level=logging.WARNING)
        loop_information = []

    def traverse_loop_tree1(loop_information, current_pragma_design):
        if "children" not in loop_information:
            return
        for child in loop_information["children"]:
            bounds.append(Categorical([1,1,1,1,2,2,2,4,4,8,16]))
            bounds.append(Categorical(["", "OFF", "OFF"]))
            # bounds.append(Categorical(["OFF"]))
            traverse_loop_tree1(child, current_pragma_design)
    
    i = 0
    current_pragma_design = []
    traverse_loop_tree1(loop_information, current_pragma_design)

    log_safe(f"Bounds: {bounds}")

    # 运行贝叶斯优化
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(run_bayesian_optimization, bounds, cpp_path, path_to_data, array_information, loop_information) for _ in range(bayesian_opt_number)]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                log_safe(f"No. {i+1} bayesian optimization for {cpp_path}, save to {path_to_data}")
                log_safe(f"  最优配置: {result.x}")
                log_safe(f"  最优代价: {result.fun}")
            except Exception as e:
                log_safe(f"  优化失败: {e}", level=logging.ERROR)
                log_safe(traceback.format_exc(), level=logging.ERROR)  # 记录完整的 Traceback

    return

def pragma_design_bayesian_in_dir(search_dir, path_to_data, max_workers=20, bayesian_opt_number=25):
    """
    遍历目录下的所有C++文件，为每个文件运行贝叶斯优化，并行执行。
    """
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    cpp_files = find_cpp_c_files(search_dir)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=round(max_workers/5)) as executor:
        # 提交每个文件的优化任务到进程池
        futures = {}
        for cpp_file in cpp_files:
            if executor_shutdown_event.is_set():
                log_safe("检测到终止信号，停止提交新任务", level=logging.WARNING)
                break
            future = executor.submit(pragma_design_bayesian, cpp_file, path_to_data, bayesian_opt_number)
            futures[future] = cpp_file
        
        # 使用 tqdm 显示进度条
        with tqdm(total=len(futures), desc="贝叶斯优化进度") as pbar:
            # 等待所有任务完成并处理结果
            for future in concurrent.futures.as_completed(futures):
                if executor_shutdown_event.is_set():
                    log_safe("检测到终止信号，停止处理结果", level=logging.WARNING)
                    break
                
                cpp_file = futures[future]
                try:
                    # 添加超时机制
                    result = future.result(timeout=3600)  # 1小时超时
                    log_safe(f"Optimization completed successfully for {cpp_file}.")
                except concurrent.futures.TimeoutError:
                    log_safe(f"Optimization timed out for {cpp_file}", level=logging.ERROR)
                except Exception as e:
                    log_safe(f"Optimization failed for {cpp_file}: {e}", level=logging.ERROR)
                    log_safe(traceback.format_exc(), level=logging.ERROR)  # 记录完整的 Tracebacks
                finally:
                    pbar.update(1)  # 更新进度条

if __name__ == '__main__':
    try:
        # 创建命令行参数解析器
        parser = argparse.ArgumentParser(description='贝叶斯优化HLS设计的pragma配置')
        parser.add_argument('--search_dir', type=str, required=True,
                            help='要搜索C/C++文件的目录路径')
        parser.add_argument('--data_path', type=str, required=True,
                            help='保存设计和结果的数据路径')
        parser.add_argument('--max_workers', type=int, default=32,
                            help='最大并行工作进程数 (默认: 32)')
        parser.add_argument('--bayesian_opt_number', type=int, default=25,
                            help='每个文件运行的贝叶斯优化次数 (默认: 25)')
        parser.add_argument('--single_file', type=str, default=None,
                            help='仅优化单个文件 (可选)')
        
        # 解析命令行参数
        args = parser.parse_args()
        
        # 根据参数执行相应的操作
        if args.single_file:
            log_safe(f"对单个文件进行优化: {args.single_file}")
            pragma_design_bayesian(args.single_file, args.data_path, args.bayesian_opt_number)
        else:
            log_safe(f"对目录中的所有文件进行优化: {args.search_dir}")
            pragma_design_bayesian_in_dir(args.search_dir, args.data_path, 
                                         max_workers=args.max_workers, 
                                         bayesian_opt_number=args.bayesian_opt_number)
    except KeyboardInterrupt:
        log_safe("程序被用户中断", level=logging.WARNING)
        # 触发信号处理器进行清理
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        log_safe(f"主程序异常: {e}", level=logging.ERROR)
        log_safe(traceback.format_exc(), level=logging.ERROR)
        # 触发信号处理器进行清理
        signal_handler(signal.SIGTERM, None)
