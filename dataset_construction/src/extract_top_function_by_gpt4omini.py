import os

from gpt4omini_example import query_gpt4omini

def extract_top_function_by_gpt4omini(file_path):
    if not os.path.isfile(file_path):
        print(f"文件不存在：{file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as file:
        c_code = file.read()

    # print(f"extract_top_function_by_gpt4omini{file_path}")

    prompt = f"""
        请阅读以下 C++ 代码，帮助我找出最适合作为顶层函数（Top Function）进行 HLS 综合的函数名称。
        请仅返回函数名称，不要包含其他内容。\n\n
        代码如下：\n{c_code}
    """
    response = query_gpt4omini(prompt)
    if response:
        # 提取函数名称，假设 GPT-4 仅返回函数名称
        top_function_name = response.strip()
        print(f"检测到的顶层函数名称：{top_function_name}")
        # 可能需要进一步处理，例如去除代码片段，只保留函数名称
    else:
        top_function_name = None
        print("未能检测到顶层函数名称。")

    return top_function_name

if __name__ == "__main__":
    extract_top_function_by_gpt4omini('./workspace/benchmark/MachSuite-flatten/aes/aes.c')
