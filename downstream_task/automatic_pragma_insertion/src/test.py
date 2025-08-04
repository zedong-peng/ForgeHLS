import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import re
import json
import torch
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from modelscope import AutoModelForCausalLM, AutoTokenizer
       
def inference_with_lora_model(lora_model, tokenizer, prompts):
    batch = tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to('cuda')
    prompt_lens = [len(input_ids) for input_ids in batch['input_ids']]
    with torch.amp.autocast('cuda'):
        output_tokens = lora_model.generate(
            **batch, 
            max_new_tokens=max(prompt_lens) + 100,
            pad_token_id=tokenizer.eos_token_id  # 设置pad_token_id
        )
    return [tokenizer.decode(output_tokens[i][prompt_lens[i]:], skip_special_tokens=True) for i in range(len(prompts))]

# 格式化函数
def formatting_func(example):
    text = f'''
    ### Instruction ###
    {example["instruction"]}

    ### Input ###
    {example["input"]}

    ### Output ###
    '''
    return text

def main(test_jsonl_path, output_jsonl_path, model_path, lora_weights_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        local_files_only=True,
        load_in_8bit=False,
        trust_remote_code=True
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
        local_files_only=True,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 测试设置
    if lora_weights_path:
        peft_model = PeftModel.from_pretrained(model, lora_weights_path).to(device)
    else:
        peft_model = model
    
    # 读取../data/test_set.jsonl
    with open(test_jsonl_path, 'r') as infile, open(output_jsonl_path, 'w') as outfile:
        batch_size = 8
        lines = infile.readlines()
        for i in tqdm(range(0, len(lines), batch_size), desc="Processing lines"):
            batch_lines = lines[i:i + batch_size]
            examples = [json.loads(line) for line in batch_lines]
            prompts = [formatting_func(example) for example in examples]
            
            predictions = inference_with_lora_model(peft_model, tokenizer, prompts)
            predictions = [
                prediction.split('##', 1)[0] if '##' in prediction else prediction
                for prediction in predictions
            ]
            for example, prediction in zip(examples, predictions):
                # Add prediction directly to the example dictionary
                example["predict"] = prediction
                outfile.write(json.dumps(example) + '\n')
    
if __name__ == "__main__":
    input_jsonl_path = '../data/testing_data_2024_11_16_23_27.jsonl'
    output_jsonl_path = '../result/llama3/llama3-origin_test_set_eval.jsonl'
    model_path = '../output/llama3/llama3-origin'
    lora_weights_path = '../output/llama3/llama3-origin/checkpoint-4131'
    main(input_jsonl_path, output_jsonl_path, model_path, lora_weights_path)
    