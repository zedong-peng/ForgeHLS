import os
import json
import subprocess
from tqdm import tqdm
from openai import OpenAI

from concurrent.futures import ThreadPoolExecutor
from find_top_function import find_top_function_by_clang

count = 0
def extract_top_function(client, text):
    prompt = '''
举一个例子
#include <stdint.h>

void lcd_counter_3(uint32_t count, uint8_t *lcd_output) {
    uint8_t digit;
    for (int i = 0; i < 4; ++i) {
#pragma HLS PIPELINE OFF 
        digit = count % 10; // Extract the next digit
    }
}
这里对应的top_function是lcd_counter_3。
对于这下面的代码，输出对应的top_function。
''' + text + '除了top_function之外，不要输出任何内容!!'
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    llm_response = client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
        temperature=0.0,
        stream=False
    )
    llm_outputs = llm_response.choices[0].message.content.strip()
    return llm_outputs

def read_jsonl(file_path):
    """读取jsonl文件并返回解析后的数据"""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]
   
def write_files(best_dir, predict_dir, example, top_function):
    """将数据写入相应的文件"""
    with open(os.path.join(best_dir, 'tmp.cpp'), 'w') as f:
        f.write(example["output"])
    
    with open(os.path.join(predict_dir, 'tmp.cpp'), 'w') as f:
        f.write(example["predict"])
    
    tcl_content = f'''
set XPART xcu280-fsvh2892-2L-e

set PROJ "design_0"
set SOLN "solution1"
set CLKP 10

open_project -reset $PROJ

add_files tmp.cpp
set_top {top_function}

open_solution -reset $SOLN
set_part $XPART
create_clock -period $CLKP

csynth_design

exit
    '''
    
    for dir_path in [best_dir, predict_dir]:
        with open(os.path.join(dir_path, 'run_hls.tcl'), 'w') as f:
            f.write(tcl_content)   

def run_hls_scripts(directories):
    """在每个文件夹下运行run_hls.tcl"""
    for dir_path in directories:
        try:
            subprocess.run(
                ['vitis_hls', '-f', 'run_hls.tcl'],
                cwd=dir_path,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running HLS script in {dir_path}: {e}")


def create_directories(base_dir, top_function):
    """创建所需的文件夹"""
    global count
    best_dir = os.path.join(base_dir, f"{top_function}_best_{count}")
    predict_dir = os.path.join(base_dir, f"{top_function}_predict_{count}")
    count += 1 # 防止文件夹重名
    
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(predict_dir, exist_ok=True)
    return best_dir, predict_dir

def process_example(example, base_dir):    
    # gpt提取 top_function
    top_function = example["top_function_name"]

    best_dir, predict_dir = create_directories(base_dir, top_function)
    write_files(best_dir, predict_dir, example, top_function)
    run_hls_scripts([best_dir, predict_dir])

def main(file_path, base_dir):
    examples = read_jsonl(file_path)
    os.makedirs(base_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = [executor.submit(process_example, example, base_dir) for example in tqdm(examples)]
        for future in futures:
            future.result()  # 等待所有线程完成

if __name__ == "__main__":
    input_jsonl_path = '../result/llama3/llama3-finetune-2024-11-16-23-33_test_set_eval.jsonl'
    base_dir = '../result/llama3/llama3-finetune-2024-11-16-23-33'
    main(input_jsonl_path, base_dir)
