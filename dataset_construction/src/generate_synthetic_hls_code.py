from openai import OpenAI
import os
import re
import json
import shutil

def query_gpt_n_times(prompt, n=1, model="chatgpt-4o-latest", temperature=0.2):
    # https://platform.openai.com/docs/models#current-model-aliases
    # model="gpt-4o",
    # model="gpt-4o-mini",
    # model="o1-mini",
    # model="o1-preview",

    # openai_api_key = os.getenv('OPENAI_API_KEY')
    openai_api_key = "sk-QpHUrsblHgB7kAzcpwLmrFz3yKKTiFVlFOW2vgVc7ARfqsXR"
    if openai_api_key:
        print("OpenAI API Key successfully retrieved.")
    else:
        print("Failed to retrieve OpenAI API Key.")
    openai_api_base = "https://a.fe8.cn/v1"

    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
    messages = [
        {"role": "system", "content": "You are an expert Vitis HLS programming assistant."},
        {"role": "user", "content": prompt}
    ]
    llm_response = client.chat.completions.create(
        messages=messages,
        model=model,
        # max_tokens=256,
        temperature=temperature,
        stream=False,
        n=n
    )
    # check if the response is successful
    if llm_response.choices[0].message.content:
        print("Response received successfully.")
    else:
        print("Failed to receive response.")
    return llm_response

def create_hls_code_by_gpt4omini(algorithm_name, output_path):
    # write functin comment about args and return
    """
    Generate HLS C++ implementations for a given algorithm using GPT-4o-mini.

    Args:
        algorithm_name (str): algorithm_name
        output_path (str): output_path/algorithm_name/implementation_i/implementation_i.cpp
    """

    prompt = f"""
Please generate a Vitis HLS code for the following algorithm:

Algorithm Name: {algorithm_name}

Requirements:
- Input and Output Parameters: The code should include Large-scale, fixed, and generated to be used as a benchmark input and output parameter types, reflecting real world usage.
- Large scale: the trip count or the size of array should be a large scale like 2^10 or less or more. Given by macro definition.
- Synthesis Readiness: The code should be a complete, standalone function or module that can be directly high level synthesized in Vitis HLS without the need for additional files or configurations. 
- Please generate a C++ program with all necessary header files correctly included, using only synthesizable constructs supported by Vitis HLS, ensuring the code can be successfully synthesized.
- Do not provide any test code. Do not provide any explanation. Only output the cppcode and the function name.
- The code should not directly include HLS pragmas
- For algorithms that may be more complex, please expand them reasonably.
- After the code is generated, please provide the function name in the comment, as shown in the example below. Format: // Top function name: function_name
- Try to avoid using while loops and use for loops instead.

Vitis HLS coding styles:
- The function and its calls must contain the entire functionality of the design.
- None of the functionality can be performed by system calls to the operating system.
- The C/C++ constructs must be of a fixed or bounded size.
- The implementation of those constructs must be unambiguous.
- Ensure the code does not use C++ features unsupported by vitis hls, such as:
    - Dynamic memory allocation (e.g., new and delete)
    - Virtual functions and polymorphism
    - Recursion
    - Complex data structures from the standard library (e.g., std::vector, std::map, std::list)
    - File I/O
    - Multithreading and concurrency (e.g., std::thread, std::mutex)
    - Pointer
    - Random number generation (e.g., std::rand, std::random)

Example:

Algorithm Name: gemm

Code Implementation:

```cpp
void gemm(int ni, int nj, int nk,
   double alpha,
   double beta,
   double C[ 1000 ][1100],
   double A[ 1000 ][1200],
   double B[ 1200 ][1100])
{{
  int i, j, k;
  for (i = 0; i < 1000; i++) {{
    for (j = 0; j < 1100; j++)
      C[i][j] *= beta;
    for (k = 0; k < 1200; k++) {{
      for (j = 0; j < 1100; j++)
        C[i][j] += alpha * A[i][k] * B[k][j];
    }}
  }}
}}
// Top function name: gemm
```


"""

    llm_response = query_gpt_n_times(prompt,1)

    code_block_list = []
    function_name_list = []

    print(llm_response)
    for i, choice in enumerate(llm_response.choices):
        print(f"Response {i + 1}: {choice.message.content}")
        for code_block in re.findall(r'```cpp(?:[a-zA-Z]+)?\n(.*?)```', choice.message.content, re.DOTALL):
            code_block_list.append(code_block)
        
        # 匹配 "Top function name: " 后面的函数名
        for function_name in re.findall(r"Top function name:\s*(\w+)", choice.message.content):
            function_name_list.append(function_name)
            print(f"Top function name: {function_name}")


    # 遍历每个代码块并保存
    for i, code_block in enumerate(code_block_list):
        if i >= len(function_name_list):
            print(f"Warning: Index {i} is out of range for function_name_list")
            continue
        # 准备目录路径
        # algorithm_name = algorithm_name.replace(" ", "_").replace("-", "_").replace("/", "_").replace(":", "_").replace("*", "_").replace("?", "_").replace("\"", "_").replace("<", "_").replace(">", "_").replace("|", "_").replace(",", "_").replace(".", "_").replace(";", "_").replace("!", "_").replace("@", "_").replace("#", "_").replace("$", "_").replace("%", "_").replace("^", "_").replace("&", "_").replace("(", "_").replace(")", "_").replace("[", "_").replace("]", "_").replace("{", "_").replace("}", "_").replace("~", "_").replace("`", "_").replace("+", "_").replace("=", "_").replace(" ", "_")
        directory_path = os.path.join(output_path, function_name_list[i], f"implementation_{i + 1}")
        file_path = os.path.join(directory_path, f"{function_name_list[i]}.cpp")

        # 如果目录存在，则删除目录
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
        os.makedirs(directory_path)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(code_block)
        
        print(f"Implementation {i + 1} saved to: {file_path}")

    print("All implementations saved successfully.")

import json
from concurrent.futures import ThreadPoolExecutor

def process_algorithms(file_path, output_dir):
    with open(file_path, 'r') as f:
        algorithms = json.load(f)
    for algorithm_name in algorithms:
        create_hls_code_by_gpt4omini(algorithm_name, output_dir)



if __name__ == "__main__":

    # algorithm_name = "Huffman encoding"
    # output_path = "../data/kernels/test"
    # create_hls_code_by_gpt4omini(algorithm_name, output_path)

    with ThreadPoolExecutor() as executor:
        executor.submit(process_algorithms, '../data/algo_list/ai_fpga_hls_algorithms.json', '../data/kernels/ai_fpga_hls_algorithms')
        executor.submit(process_algorithms, '../data/algo_list/leetcode_hls_algorithms.json', '../data/kernels/leetcode_hls_algorithms')
        executor.submit(process_algorithms, '../data/algo_list/operators.json', '../data/kernels/operators')
        executor.submit(process_algorithms, '../data/algo_list/rtl_chip.json', '../data/kernels/rtl_chip')
        executor.submit(process_algorithms, '../data/algo_list/rtl_ip.json', '../data/kernels/rtl_ip')
        executor.submit(process_algorithms, '../data/algo_list/rtl_module.json', '../data/kernels/rtl_module')
        # executor.submit(process_algorithms, '../data/algo_list/hls_algorithms.json', '../data/kernels/hls_algorithms')


    