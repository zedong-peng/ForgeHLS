import clang.cindex

# 递归打印 AST
def print_ast(node, file_path, indent=0):
    if f"{node.location.file}" == file_path:
        spelling = node.spelling if node.spelling else '-'
        displayname = node.displayname if node.displayname else '-'
        
        # 打印当前节点的种类、名称和行号
        print('  ' * indent + f"{spelling} ({displayname}): {node.kind} at {node.location.file}:{node.location.line}")
    
    # 递归遍历子节点
    for child in node.get_children():
        print_ast(child, file_path, indent + 1)


if __name__ == '__main__':
    file_path = './test.cpp'
    print(f"AST of {file_path}:")
    # 创建一个索引对象
    index = clang.cindex.Index.create()
    
    # 解析翻译单元，指定编译参数以确保正确解析
    translation_unit = index.parse(
        file_path,
        args=[
            '-x', 'c++',          # 指定语言为 C++
            '-std=c++14',         # 指定 C++ 标准（根据需要调整）
            # '-I/usr/include',     # 添加必要的包含路径（根据您的环境调整）
            # '-I/home/Program/Xilinx/Vitis_HLS/2023.2/include', # 不要添加HLS的头文件, 会污染AST
            # 可以根据需要添加更多编译参数
        ],
        # options=clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD # 详细处理记录 会污染AST
    )
    
    # 获取翻译单元的根游标
    root_cursor = translation_unit.cursor

    print_ast(root_cursor, file_path)
