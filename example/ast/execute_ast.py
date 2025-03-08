import ast

def execute_function_body(func_ast, env, args=None, kwargs=None):
    """
    执行函数体的 AST 语句
    :param func_ast: 函数的 AST 节点
    :param env: 全局变量环境
    :param args: 函数的位置参数（列表）
    :param kwargs: 函数的关键字参数（字典）
    """
    if not isinstance(func_ast, ast.FunctionDef):
        raise ValueError("Provided AST is not a function definition")

    # 将函数名添加到环境中
    env[func_ast.name] = lambda *a, **kw: execute_function_body(func_ast, env, a, kw)

    # 如果提供了参数，将它们添加到环境中
    if args is not None:
        for arg_name, arg_value in zip(func_ast.args.args, args):
            env[arg_name.arg] = arg_value

    if kwargs is not None:
        for arg_name, arg_value in kwargs.items():
            env[arg_name] = arg_value

    # 遍历函数体的语句并逐行执行
    for stmt in func_ast.body:
        code = compile(ast.Module(body=[stmt], type_ignores=[]), filename="<ast>", mode="exec")
        exec(code, env)

# 示例用法
if __name__ == "__main__":
    # 示例函数代码
    func_code = """
def example_func(x, y):
    z = x + y
    print(z)
    return z
"""

    # 解析代码为 AST
    func_ast = ast.parse(func_code).body[0]

    # 全局变量环境
    env = {}

    # 执行函数体，并传入参数
    execute_function_body(func_ast, env, args=[10, 20])

    # 输出全局变量环境中的结果
    print(env)  # 应该包含 x, y, z 的值