import os
import re
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from modelscope import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import asyncio
from typing import Any
import numpy as np
from openai import AsyncOpenAI  
import aioconsole  


class LLMClient:
    def __init__(self, model_name, api_key, base_url):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.client = AsyncOpenAI(  
            base_url=self.base_url,
            api_key=self.api_key
        )

    async def get_response(self, prompt: str) -> Any:
        # messages = [{"role": "user", "content": prompt}]
        messages = [{"role": "user", "content": prompt}]
        
        completion=  await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            n=1,
            temperature=0.1,
            max_tokens=64,
        )
        
        return completion.choices[0].message.content

async def process_prompts(client, prompts):
    tasks = []
    for prompt in prompts:
        tasks.append(client.get_response(prompt))
    # 并行执行所有请求
    return await asyncio.gather(*tasks)

def inference_with_lora_model(prompts, batch_size=1):
    client = LLMClient(
        model_name="HLS_model",
        api_key="token-abc123",
        base_url="http://localhost:8015/v1"
    )
    
    # 只创建一次事件循环，处理所有提示
    return asyncio.run(process_prompts(client, prompts))

# 格式化函数
def formatting_func(example):
    text = f'''
{example["instruction"]}

```start of code
{example["input"]}
```end of code

```start of prediction
Prediction:
'''
    return text

# 提取指标值的函数
def extract_metrics(text):
    """从文本中提取latency, lut, dsp, ff的值"""
    # 尝试从JSON格式中提取，支持多种键名格式
    json_patterns = [
        # 标准格式：latency, lut, dsp, ff
        r'{\s*"(?i:latency)"\s*:\s*(\d+)\s*,\s*"(?i:lut)"\s*:\s*(\d+)\s*,\s*"(?i:dsp)"\s*:\s*(\d+)\s*,\s*"(?i:ff)"\s*:\s*(\d+)\s*}',
        # 大写格式：LUT, DSP, FF
        r'{\s*"(?i:latency)"\s*:\s*(\d+)\s*,\s*"LUT"\s*:\s*(\d+)\s*,\s*"DSP"\s*:\s*(\d+)\s*,\s*"FF"\s*:\s*(\d+)\s*}',
        # Worst-caseLatency格式
        r'{\s*"(?i:worst[-\s]*case[-\s]*latency)"\s*:\s*(\d+)\s*,\s*"LUT"\s*:\s*(\d+)\s*,\s*"DSP"\s*:\s*(\d+)\s*,\s*"FF"\s*:\s*(\d+)\s*}',
        # 支持任意顺序的JSON格式
        r'{\s*"[^"]*"(?i:latency)"[^"]*"\s*:\s*(\d+)\s*,\s*"[^"]*"(?i:lut)"[^"]*"\s*:\s*(\d+)\s*,\s*"[^"]*"(?i:dsp)"[^"]*"\s*:\s*(\d+)\s*,\s*"[^"]*"(?i:ff)"[^"]*"\s*:\s*(\d+)\s*}',
        # 不带引号的格式：{latency: 0, lut:0, dsp:0, ff:0}
        r'{\s*(?i:latency)\s*:\s*(\d+)\s*,\s*(?i:lut)\s*:\s*(\d+)\s*,\s*(?i:dsp)\s*:\s*(\d+)\s*,\s*(?i:ff)\s*:\s*(\d+)\s*}',
        # 不带引号的大写格式：{LATENCY: 0, LUT:0, DSP:0, FF:0}
        r'{\s*(?i:latency)\s*:\s*(\d+)\s*,\s*LUT\s*:\s*(\d+)\s*,\s*DSP\s*:\s*(\d+)\s*,\s*FF\s*:\s*(\d+)\s*}',
        # 不带引号的Worst-caseLatency格式：{Worst-caseLatency: 0, LUT:0, DSP:0, FF:0}
        r'{\s*(?i:worst[-\s]*case[-\s]*latency)\s*:\s*(\d+)\s*,\s*LUT\s*:\s*(\d+)\s*,\s*DSP\s*:\s*(\d+)\s*,\s*FF\s*:\s*(\d+)\s*}',
    ]
    
    for json_pattern in json_patterns:
        json_match = re.search(json_pattern, text, re.DOTALL | re.IGNORECASE)
        if json_match:
            return {
                'latency': int(json_match.group(1)),
                'lut': int(json_match.group(2)),
                'dsp': int(json_match.group(3)),
                'ff': int(json_match.group(4))
            }, True
    
    # 如果标准JSON格式不匹配，尝试更灵活的JSON解析
    # 查找包含所有四个指标的JSON对象（带引号和不带引号）
    json_object_patterns = [
        # 带引号的格式
        r'\{[^{}]*"(?i:latency|worst[-\s]*case[-\s]*latency)"[^{}]*"(?i:lut)"[^{}]*"(?i:dsp)"[^{}]*"(?i:ff)"[^{}]*\}',
        # 不带引号的格式
        r'\{[^{}]*\b(?i:latency|worst[-\s]*case[-\s]*latency)\b[^{}]*\b(?i:lut)\b[^{}]*\b(?i:dsp)\b[^{}]*\b(?i:ff)\b[^{}]*\}',
    ]
    
    for json_object_pattern in json_object_patterns:
        json_object_match = re.search(json_object_pattern, text, re.DOTALL | re.IGNORECASE)
        if json_object_match:
            json_str = json_object_match.group(0)
            
            # 尝试提取各个值（带引号和不带引号）
            latency_patterns = [
                r'"(?i:latency|worst[-\s]*case[-\s]*latency)"\s*:\s*(\d+)',  # 带引号
                r'\b(?i:latency|worst[-\s]*case[-\s]*latency)\b\s*:\s*(\d+)',  # 不带引号
            ]
            lut_patterns = [
                r'"(?i:lut)"\s*:\s*(\d+)',  # 带引号
                r'\b(?i:lut)\b\s*:\s*(\d+)',  # 不带引号
            ]
            dsp_patterns = [
                r'"(?i:dsp)"\s*:\s*(\d+)',  # 带引号
                r'\b(?i:dsp)\b\s*:\s*(\d+)',  # 不带引号
            ]
            ff_patterns = [
                r'"(?i:ff)"\s*:\s*(\d+)',  # 带引号
                r'\b(?i:ff)\b\s*:\s*(\d+)',  # 不带引号
            ]
            
            # 尝试匹配latency
            latency_match = None
            for pattern in latency_patterns:
                latency_match = re.search(pattern, json_str, re.IGNORECASE)
                if latency_match:
                    break
            
            # 尝试匹配LUT
            lut_match = None
            for pattern in lut_patterns:
                lut_match = re.search(pattern, json_str, re.IGNORECASE)
                if lut_match:
                    break
            
            # 尝试匹配DSP
            dsp_match = None
            for pattern in dsp_patterns:
                dsp_match = re.search(pattern, json_str, re.IGNORECASE)
                if dsp_match:
                    break
            
            # 尝试匹配FF
            ff_match = None
            for pattern in ff_patterns:
                ff_match = re.search(pattern, json_str, re.IGNORECASE)
                if ff_match:
                    break
            
            if latency_match and lut_match and dsp_match and ff_match:
                return {
                    'latency': int(latency_match.group(1)),
                    'lut': int(lut_match.group(1)),
                    'dsp': int(dsp_match.group(1)),
                    'ff': int(ff_match.group(1))
                }, True
    
    # 尝试从文本格式中提取，支持多种格式和大小写不敏感
    # 定义多种匹配模式，按优先级排序
    latency_patterns = [
        r'(?i)worst[-\s]*case[-\s]*latency[:\s]*=?\s*(\d+)',  # Worst-case latency: 123 或 worst case latency = 123
        r'(?i)latency[:\s]*=?\s*(\d+)',  # latency: 123 或 latency = 123
        r'(?i)latency[:\s]*(\d+)',  # latency: 123
    ]
    
    lut_patterns = [
        r'(?i)lut[:\s]*=?\s*(\d+)',  # LUT: 123 或 LUT = 123
        r'(?i)look[-\s]*up[-\s]*table[:\s]*=?\s*(\d+)',  # look-up table: 123
    ]
    
    dsp_patterns = [
        r'(?i)dsp[:\s]*=?\s*(\d+)',  # DSP: 123 或 DSP = 123
        r'(?i)digital[-\s]*signal[-\s]*processor[:\s]*=?\s*(\d+)',  # digital signal processor: 123
    ]
    
    ff_patterns = [
        r'(?i)ff[:\s]*=?\s*(\d+)',  # FF: 123 或 FF = 123
        r'(?i)flip[-\s]*flop[:\s]*=?\s*(\d+)',  # flip-flop: 123
    ]
    
    # 尝试匹配latency
    latency_match = None
    for pattern in latency_patterns:
        latency_match = re.search(pattern, text, re.IGNORECASE)
        if latency_match:
            break
    
    # 尝试匹配LUT
    lut_match = None
    for pattern in lut_patterns:
        lut_match = re.search(pattern, text, re.IGNORECASE)
        if lut_match:
            break
    
    # 尝试匹配DSP
    dsp_match = None
    for pattern in dsp_patterns:
        dsp_match = re.search(pattern, text, re.IGNORECASE)
        if dsp_match:
            break
    
    # 尝试匹配FF
    ff_match = None
    for pattern in ff_patterns:
        ff_match = re.search(pattern, text, re.IGNORECASE)
        if ff_match:
            break
    
    metrics = {}
    if latency_match:
        metrics['latency'] = int(latency_match.group(1))
    if lut_match:
        metrics['lut'] = int(lut_match.group(1))
    if dsp_match:
        metrics['dsp'] = int(dsp_match.group(1))
    if ff_match:
        metrics['ff'] = int(ff_match.group(1))
    
    # 检查是否所有值都找到了
    all_found = all(k in metrics for k in ['latency', 'lut', 'dsp', 'ff'])
    
    return metrics, all_found

# 计算评估指标
def calculate_metrics(predictions, ground_truths):
    """计算MAPE和RMSE"""
    metrics = {}
    
    for key in ['latency', 'lut', 'dsp', 'ff']:
        pred_values = np.array([p[key] for p in predictions if key in p])
        true_values = np.array([g[key] for g in ground_truths if key in g])
        
        # 确保有足够的数据进行计算
        if len(pred_values) > 0 and len(true_values) > 0 and len(pred_values) == len(true_values):
            # RMSE (Root Mean Square Error)
            rmse = np.sqrt(np.mean((true_values - pred_values) ** 2))
            metrics[f'{key}_rmse'] = float(rmse)
            
            # MAPE (Mean Absolute Percentage Error)
            # 避免除以零的情况
            non_zero_indices = true_values != 0
            if np.any(non_zero_indices):
                # 只对非零值计算MAPE
                mape = np.mean(np.abs((true_values[non_zero_indices] - pred_values[non_zero_indices]) / 
                                      true_values[non_zero_indices])) * 100
                # 将MAPE值保存为带百分号的字符串
                metrics[f'{key}_mape(%)'] = f"{float(mape):.4f}%"
            else:
                # 如果所有真实值都是零，则无法计算MAPE
                metrics[f'{key}_mape(%)'] = "Nan. All ground truth values are zero."
            
            # 添加一个替代指标：平均绝对误差(MAE)
            mae = np.mean(np.abs(true_values - pred_values))
            metrics[f'{key}_mae'] = float(mae)
    
    return metrics

def main(base_model_path, test_dataset_path,
         test_output_dir = "./testing_output",
         lora_weights_path=None, device=None,
         batch_size=1):  # 默认批处理大小为1，保持结果一致性

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        # Handle case where device is an integer (GPU index)
        if isinstance(device, int) or (isinstance(device, str) and device.isdigit()):
            device = torch.device(f"cuda:{device}")
        else:
            device = torch.device(device)

    # model = AutoModelForCausalLM.from_pretrained(
    #     base_model_path, 
    #     local_files_only=True,
    #     load_in_8bit=False,
    #     trust_remote_code=True
    # ).to(device)


    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
        local_files_only=True,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # if lora_weights_path is not None: 
    #     peft_model = PeftModel.from_pretrained(model, lora_weights_path).to(device)
    # else:
    #     peft_model = model

    test_dataset = load_dataset('json', data_files=test_dataset_path)
    print(f"Loaded dataset: {test_dataset_path}")
    print(f"Testing output dir: {test_output_dir}")

    os.makedirs(test_output_dir, exist_ok=True)

    # 统计变量
    total_examples = 0
    successful_extractions = 0
    all_predictions = []
    all_ground_truths = []

    with open(test_output_dir + "/predictions.jsonl", "w") as file:
        for example in tqdm(test_dataset['train'], desc="Processing examples"):
            total_examples += 1
            
            # 格式化 prompt
            prompt = formatting_func(example)

            # 推理
            predictions = inference_with_lora_model([prompt])
            
            # 从预测中提取指标
            pred_metrics, pred_success = extract_metrics(predictions[0])
            
            # 从真实值中提取指标
            true_metrics = {}
            if 'output' in example:
                true_metrics, _ = extract_metrics(example['output'])
            
            # 记录结果
            result = {
                "prompt": prompt,
                "prediction": predictions[0],
                **pred_metrics
            }
            
            # 添加真实值到结果中
            if true_metrics:
                result["ground_truth"] = true_metrics
            
            if pred_success:
                successful_extractions += 1
                all_predictions.append(pred_metrics)
                if true_metrics:
                    all_ground_truths.append(true_metrics)
                
                file.write(json.dumps(result) + "\n")
            else:
                error_result = {
                    "prompt": prompt,
                    "prediction": predictions[0],
                    "error": "Failed to extract values"
                }
                # 如果有真实值，也添加到错误结果中
                if true_metrics:
                    error_result["ground_truth"] = true_metrics
                    
                file.write(json.dumps(error_result) + "\n")
    
    # 计算成功率
    success_rate = (successful_extractions / total_examples) * 100 if total_examples > 0 else 0
    print(f"Successfully extracted metrics from {successful_extractions} out of {total_examples} examples ({success_rate:.2f}%)")
    
    # 计算评估指标
    if all_predictions and all_ground_truths and len(all_predictions) == len(all_ground_truths):
        evaluation_metrics = calculate_metrics(all_predictions, all_ground_truths)
        
        # 保存评估结果
        with open(os.path.join(test_output_dir, "evaluation_metrics.json"), "w") as f:
            json.dump({
                "success_rate": success_rate,
                **evaluation_metrics
            }, f, indent=2)
        
        # 打印评估结果
        print("\nEvaluation Metrics:")
        for metric, value in evaluation_metrics.items():
            if isinstance(value, str):
                print(f"{metric}: {value}")
            else:
                print(f"{metric}: {value:.4f}")
    else:
        print("Not enough data to calculate evaluation metrics")
        with open(os.path.join(test_output_dir, "evaluation_metrics.json"), "w") as f:
            json.dump({
                "success_rate": success_rate,
                "error": "Not enough data to calculate evaluation metrics"
            }, f, indent=2)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a LoRA model")
    parser.add_argument("--base_model_path", type=str, required=True, 
                        help="Path to the base model")
    parser.add_argument("--test_dataset_path", type=str, required=True, 
                        help="Path to the test dataset JSON file")
    parser.add_argument("--test_output_dir", type=str, default="./testing_output", 
                        help="Directory to save test outputs")
    parser.add_argument("--lora_weights_path", type=str, default=None, 
                        help="Path to LoRA weights (optional)")
    parser.add_argument("--device", type=str, default=None, 
                        help="Device to run inference on (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Batch size for inference")
    
    args = parser.parse_args()
    
    main(
        base_model_path=args.base_model_path,
        test_dataset_path=args.test_dataset_path,
        test_output_dir=args.test_output_dir,
        lora_weights_path=args.lora_weights_path,
        device=args.device,
        batch_size=args.batch_size
    )
    