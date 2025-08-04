#!/usr/bin/env python3
import os
import json
import random
from datetime import datetime

def extract_code(source_code_array):
    """从source_code数组中提取代码内容"""
    if not source_code_array or not isinstance(source_code_array, list):
        return ""
    
    # 合并多个文件的代码
    code_content = ""
    for file_info in source_code_array:
        if isinstance(file_info, dict) and 'file_content' in file_info:
            if code_content:
                code_content += "\n\n// NEXT FILE\n\n"
            code_content += file_info.get('file_content', '')
    
    return code_content

def main():
    # 输入JSON文件路径
    json_file = ""
    
    # 输出训练和测试数据的路径
    output_dir = ""
    os.makedirs(output_dir, exist_ok=True)
    
    print("开始生成训练和测试数据...")
    
    # 读取JSON文件，按source_name和algo_name分组
    designs_by_source_algo = {}
    
    # 修改这部分：直接读取整个JSON数组
    with open(json_file, 'r') as f:
        designs = json.load(f)
        print(f"读取到 {len(designs)} 个设计")
        
        for design in designs:
            source_name = design.get('source_name', '')
            algo_name = design.get('algo_name', '')
            
            if not source_name or not algo_name:
                continue
            
            # 创建分组键
            key = f"{source_name}:{algo_name}"
            
            if key not in designs_by_source_algo:
                designs_by_source_algo[key] = []
                
            designs_by_source_algo[key].append(design)
    
    print(f"找到 {len(designs_by_source_algo)} 个source_name:algo_name组合")
    
    # 准备训练和测试数据
    training_data = []
    testing_data = []
    
    # 跟踪处理的组合和成功生成的数据
    total_combinations = 0
    successful_combinations = 0
    skipped_combinations = 0
    
    # 处理每个组合
    for key, designs in designs_by_source_algo.items():
        total_combinations += 1
        print(f"\n处理组合: {key}")
        
        # 只选择Pareto最优设计
        pareto_designs = [d for d in designs if d.get('is_pareto') == True]
        print(f"Pareto设计数量: {len(pareto_designs)}")
        if len(pareto_designs) < 3:
            print(f"跳过：Pareto设计数量少于3个")
            skipped_combinations += 1
            continue
        
        # 检查是否有不同的策略
        strategies = set(d.get('latency-resource-strategy', '') for d in pareto_designs)
        print(f"找到的策略种类: {strategies}")
        if len(strategies) < 3:
            print(f"跳过：策略种类少于3种")
            skipped_combinations += 1
            continue
        
        # 找到每种策略的一个示例，只从Pareto设计中选择
        strategy_examples = {}
        for strategy in ["low-latency-high-resource", "medium-latency-medium-resource", "high-latency-low-resource"]:
            strategy_pareto_designs = [d for d in pareto_designs if d.get('latency-resource-strategy') == strategy]
            print(f"策略 {strategy} 的Pareto设计数量: {len(strategy_pareto_designs)}")
            if strategy_pareto_designs:
                strategy_pareto_designs.sort(key=lambda x: x.get('Best-caseLatency', float('inf')))
                strategy_examples[strategy] = strategy_pareto_designs[0]
        
        # 如果没有找到全部三种策略的示例，跳过
        if len(strategy_examples) < 3:
            print(f"跳过：没有找到全部三种策略的示例")
            skipped_combinations += 1
            continue
        
        # 由于所有设计使用相同的算法，我们可以假设它们共享相同的输入代码
        # 从第一个设计中提取输入代码
        first_design = designs[0]
        input_code_array = first_design.get('source_code', [])
        input_code = extract_code(input_code_array)
        
        if not input_code:
            skipped_combinations += 1
            continue
        
        # 处理每种策略
        valid_examples = True
        strategy_data = []
        
        for strategy, design in strategy_examples.items():
            # 获取输出代码 (包含pragma的代码)
            output_code_array = design.get('source_code', [])
            output_code = extract_code(output_code_array)
            
            if not output_code:
                valid_examples = False
                break
            
            # 创建指令
            base_instruction = "Input is high level synthesis code based on C++, output code is the high level synthesis code with appropriate pragma, which is optimized for the target FPGA device."
            
            if strategy == "low-latency-high-resource":
                instruction = base_instruction + " The output code is optimized for low latency and high resource usage."
            elif strategy == "medium-latency-medium-resource":
                instruction = base_instruction + " The output code is optimized for medium latency and medium resource usage."
            else:  # high-latency-low-resource
                instruction = base_instruction + " The output code is optimized for high latency and low resource usage."
            
            # 创建数据项
            data_item = {
                "instruction": instruction,
                "input": input_code,
                "output": output_code,
                "metrics": {
                    # 资源使用
                    "BRAM_18K": design.get("BRAM_18K", "N/A"),
                    "LUT": design.get("LUT", "N/A"),
                    "DSP": design.get("DSP", "N/A"),
                    "FF": design.get("FF", "N/A"),
                    
                    # 可用资源
                    "Avialable_BRAM_18K": design.get("Avialable_BRAM_18K", "N/A"),
                    "Avialable_DSP": design.get("Avialable_DSP", "N/A"),
                    "Avialable_FF": design.get("Avialable_FF", "N/A"),
                    "Avialable_LUT": design.get("Avialable_LUT", "N/A"),
                    
                    # 延迟信息
                    "Best-caseLatency": design.get("Best-caseLatency", "N/A"),
                    "Worst-caseLatency": design.get("Worst-caseLatency", "N/A"),
                    
                    # 策略
                    "strategy": strategy
                }
            }
            
            strategy_data.append((strategy, data_item))
        
        if valid_examples:
            # 添加到训练或测试集
            for strategy, data_item in strategy_data:
                if is_training:
                    training_data.append(data_item)
                else:
                    testing_data.append(data_item)
            
            successful_combinations += 1
    
    # 保存训练和测试数据
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
    
    training_data_path = os.path.join(output_dir, f"training_data_{timestamp}.jsonl")
    with open(training_data_path, 'w') as f:
        for item in training_data:
            f.write(json.dumps(item) + '\n')
    
    testing_data_path = os.path.join(output_dir, f"testing_data_{timestamp}.jsonl")
    with open(testing_data_path, 'w') as f:
        for item in testing_data:
            f.write(json.dumps(item) + '\n')
    
    print("\n生成完成！")
    print(f"总处理组合数: {total_combinations}")
    print(f"成功生成数据的组合数: {successful_combinations}")
    print(f"跳过的组合数: {skipped_combinations}")
    print(f"训练数据: {len(training_data)} 个样本，保存在 {training_data_path}")
    print(f"测试数据: {len(testing_data)} 个样本，保存在 {testing_data_path}")

if __name__ == "__main__":
    main() 