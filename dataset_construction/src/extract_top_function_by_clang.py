import clang.cindex
import os

# 设置 Clang 库的路径（根据您的安装位置调整）
# clang.cindex.Config.set_library_file('/usr/lib/llvm-10/lib/libclang.so.1')  # 示例路径

# 递归查找函数调用的地方，仅限于主文件
def collect_called_functions(node, main_file):
    called_functions = set()

    def _collect(node):
        # 只处理主文件中的节点
        if node.location and node.location.file and os.path.abspath(node.location.file.name) != os.path.abspath(main_file):
            return

        # 处理函数调用表达式
        if node.kind == clang.cindex.CursorKind.CALL_EXPR:
            referenced = node.referenced
            if referenced and referenced.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                called_functions.add(referenced.spelling)

        # 处理声明引用表达式
        elif node.kind == clang.cindex.CursorKind.DECL_REF_EXPR:
            # 检查是否是重载的声明引用
            if node.kind == clang.cindex.CursorKind.OVERLOADED_DECL_REF:
                for referenced in node.get_overloaded_decls():
                    if referenced.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                        called_functions.add(referenced.spelling)
            else:
                referenced = node.referenced
                if referenced and referenced.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                    called_functions.add(referenced.spelling)

        # 递归遍历子节点
        for child in node.get_children():
            _collect(child)

    _collect(node)
    return called_functions

# 查找未被调用的顶层函数（top functions），仅限于主文件
def find_uncalled_function(node, called_functions, main_file):
    uncalled_function_list = set()

    def _find_top(node):
        # 只处理主文件中的节点
        if node.location and node.location.file and os.path.abspath(node.location.file.name) != os.path.abspath(main_file):
            return

        if node.kind == clang.cindex.CursorKind.FUNCTION_DECL and node.is_definition():
            function_name = node.spelling
            if function_name not in called_functions:
                uncalled_function_list.add(function_name)

        if node.kind == clang.cindex.CursorKind.FUNCTION_TEMPLATE and node.is_definition():
            function_name = node.spelling
            if function_name not in called_functions:
                uncalled_function_list.add(function_name)

        # 递归遍历子节点
        for child in node.get_children():
            _find_top(child)

    _find_top(node)
    return uncalled_function_list

def print_ast(node, indent=0):
    spelling = node.spelling if node.spelling else '-'
    
    # 打印当前节点的种类、名称和行号
    print('  ' * indent + f"{spelling}: {node.kind} at line {node.location.line}")
    
    # 递归遍历子节点
    for child in node.get_children():
        print_ast(child, indent + 1)

# 主函数
def extract_top_function_by_clang(file_path):
    print(f"extract_top_function_by_clang: {file_path}")
    index = clang.cindex.Index.create()
    # 根据需要添加编译参数，如包含路径、宏定义等
    # if file is cpp, add '-std=c++11'
    if file_path.endswith('.cpp'):
        translation_unit = index.parse(file_path, args=['-std=c++11']) 
    if file_path.endswith('.c'):
        translation_unit = index.parse(file_path, args=['-std=c11'])
    ast_root_node = translation_unit.cursor

    # print_ast(ast_root_node)

    # 收集所有被调用的函数，仅限于主文件
    called_functions = collect_called_functions(ast_root_node, file_path)
    # print("\nCalled Functions:")
    # for function in called_functions:
    #     print(f"  {function}")


    # 查找 top functions，仅限于主文件
    uncalled_functions = find_uncalled_function(ast_root_node, called_functions, file_path)

    if len(uncalled_functions) == 1:
        top_function = uncalled_functions.pop()
        if top_function:
            print("Set the onlyone uncall function as Top Function:", f"{top_function}" if top_function else "None")
        else:
            print("No Top Function Found.")
        return top_function
    elif len(uncalled_functions) > 1:
        # 报错：找到多个未被调用的函数
        print("Multiple Uncalled Functions Found:", ", ".join(str(function) for function in uncalled_functions))
        return uncalled_functions
    else:
        # 报错：没有找到未被调用的函数
        print("No Uncalled Functions Found.")
        return None

if __name__ == '__main__':
    #file_path = '/data/HLS_data/kernels/gpt4omini/NBCD_ADDER/NBCD_ADDER.cpp'
    
    file_path = './workspace/HLSBatchProcessor/data/kernels/operators/encoder_32_to_5/encoder_32_to_5.cpp'
    top_function = extract_top_function_by_clang(file_path)

    