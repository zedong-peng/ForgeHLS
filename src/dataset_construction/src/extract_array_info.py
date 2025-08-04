import os
import json
import re
from tqdm import tqdm
import concurrent.futures
from gpt4omini_example import query_gpt4omini
from find_cpp_c_files import find_cpp_c_files
import argparse

def add_line_numbers(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    code_with_linenumber = "\n".join(f"line: {idx:<4} {line.rstrip()}" for idx, line in enumerate(lines, start=1))
    return code_with_linenumber

def extract_array_info_json_from_response(response):
    """
    使用正则表达式从模型响应中提取 JSON 内容。
    支持 JSON 对象和数组。
    """
    # # 打印响应内容用于调试
    # print("模型响应内容:")
    # print(response)
    
    # 尝试从 ```json ``` 中提取
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response, re.IGNORECASE)
    if json_match:
        json_str = json_match.group(1)
    else:
        # 尝试从第一个 { 或 [ 开始
        start = response.find('{')
        if start == -1:
            start = response.find('[')
        if start == -1:
            print("响应中不包含 JSON 对象或数组。")
            return None
        json_str = response[start:]
    
    try:
        array_info = json.loads(json_str)
        return array_info
    except json.JSONDecodeError as e:
        print(f"JSON 格式错误，无法解析: {e}")
        print("提取的 JSON 字符串:")
        print(json_str)
        return None

def extract_array_info(file_path):
    """
    extract /path/to/cpp array info to /path/to/cpp/array_info/array_info.json
    
    :param file_path: C++ file path
    """
    if not os.path.isfile(file_path):
        print(f"文件不存在：{file_path}")
        return

    # with open(file_path, 'r', encoding='utf-8') as file:
    #     c_code = file.read()

    c_code = add_line_numbers(file_path)

    prompt = f"""
Please carefully read the following C++ code and extract the dimensional information of all arrays. For each array, extract the following details:

#### **Extraction Requirements**
1. **array_name**: The variable name of the array.
2. **dimensions**: The number of dimensions in the array.
3. **size_per_dimension**: The actual size of each dimension (e.g., 10, 20). Use specific numeric values, not symbolic expressions (e.g., N, M). If the size is defined by a macro (e.g., `M` or `N`), replace the macro with its actual numeric value from the macro definition. If can not find the actual number, set 16 as default.
4. **line_per_dimension**: The line number where the array is fisrtly defined or used. If the array is a function parameter, use the starting line number of the function body (e.g., the line with the opening brace `{{`), because when adding HLS array pragma for the array as function parameter, it should be added to the first line of the function body. 

#### **Attention**
这些数组信息将会被用于vitis HLSd的pragma设计, 并不需要太多, 所以最好只在json中记录顶层函数（Top Function）的传入参数的数组.
If no array find, return blank json like this:
```json
[]
```


#### **Output Format**
Provide the results in JSON array format, as shown in the example below:
```json
[
    {{
        "array_name": "A",
        "dimensions": 2,
        "size_per_dimension": [10, 10]
        "line_per_dimension": [8, 15]
    }}
]
```

#### **Example**
Consider the following C++ code snippet:
```cpp
#define N 1024

int accumulate(int A[N], int B[N])
{{

    int out_accum = 0;
    int x;
    int y;
    for (x = 0; x < N; x++){{
        for (y = 0; y < N; y++){{
            out_accum += A[x]+B[y];
        }}
    }}
    return out_accum;
}}
```
The array information extracted from the above code snippet would be:
```json
[
    {{
        "array_name": "A",
        "dimensions": 1,
        "size_per_dimension": [
            1024
        ],
        "line_per_dimension": [
            3
        ]
    }},
    {{
        "array_name": "B",
        "dimensions": 1,
        "size_per_dimension": [
            1024
        ],
        "line_per_dimension": [
            3
        ]
    }}
]

#### **C++ Code to Analyze**
```cpp
{c_code}
```

"""
    # print(prompt)
    response = query_gpt4omini(prompt)
    if response:
        # 提取数组信息，假设 GPT-4 返回的是 JSON 数组
        array_info = extract_array_info_json_from_response(response.strip())
        if array_info is not None:
            # 定义输出目录和文件路径
            output_dir = os.path.join(os.path.dirname(file_path), "array_info")
            os.makedirs(output_dir, exist_ok=True)  # 创建目录（如果不存在）
            output_path = os.path.join(output_dir, "array_info.json")
            try:
                # 将数组信息写入 JSON 文件，使用缩进提高可读性
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(array_info, f, ensure_ascii=False, indent=4)
                print(f"Array information saved to {output_path}")
            except IOError as e:
                print(f"写入文件时出错：{e}")
        else:
            # save blank json file
            output_dir = os.path.join(os.path.dirname(file_path), "array_info")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "array_info.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=4)
            print(f"Array information saved to {output_path}")
    else:
        print("未能获取 GPT-4 响应。")

def extract_array_info_in_dir(search_dir):
    """
    在指定目录及其子目录（最大深度 max_depth）中搜索 .cpp 和 .c 文件，并提取数组信息。
    
    :param search_dir: 要搜索的目录
    """
    cpp_c_files = find_cpp_c_files(search_dir)
    print(cpp_c_files)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # 创建进度条
        with tqdm(total=len(cpp_c_files), desc="Processing files") as pbar:
            # 提交任务
            futures = {executor.submit(extract_array_info, file): file for file in cpp_c_files}
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='提取C/C++文件中的数组信息')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dir', type=str, help='要处理的目录路径')
    group.add_argument('--file', type=str, help='要处理的单个文件路径')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    if args.file:
        extract_array_info(args.file)
    elif args.dir:
        extract_array_info_in_dir(args.dir)