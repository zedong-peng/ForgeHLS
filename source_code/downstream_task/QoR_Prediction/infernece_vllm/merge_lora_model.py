import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

def merge_lora_model(base_model_path, lora_model_path, output_path):
    """
    合并 LoRA 模型与基础模型
    
    Args:
        base_model_path (str): 基础模型路径
        lora_model_path (str): LoRA 微调模型路径
        output_path (str): 合并后模型的保存路径
    """
    print(f"开始合并模型...")
    print(f"基础模型路径: {base_model_path}")
    print(f"LoRA 模型路径: {lora_model_path}")
    print(f"输出路径: {output_path}")
    
    # 1. 加载基础模型和分词器
    print("正在加载基础模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

    # 2. 加载 LoRA 微调模型
    print("正在加载 LoRA 微调模型...")
    lora_model = PeftModel.from_pretrained(base_model, lora_model_path)

    # 3. 合并 LoRA 参数与基础模型参数
    print("正在合并 LoRA 参数与基础模型参数...")
    merged_model = lora_model.merge_and_unload()  # 合并参数并卸载 LoRA 适配器

    # 4. 保存合并后的模型
    print("正在保存合并后的模型...")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"合并完成！模型已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="合并 LoRA 模型与基础模型")
    parser.add_argument(
        "--base_model_path", 
        type=str, 
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="基础模型路径 (默认: Qwen/Qwen2.5-Coder-1.5B-Instruct)"
    )
    parser.add_argument(
        "--lora_model_path", 
        type=str, 
        required=True,
        help="LoRA 微调模型路径"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True,
        help="合并后模型的保存路径"
    )
    
    args = parser.parse_args()
    
    merge_lora_model(args.base_model_path, args.lora_model_path, args.output_path)

if __name__ == "__main__":
    main()