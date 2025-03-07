import ast
import astunparse

# 原始的 Python 代码
source_code = """
def add(a, b):
    return a + b
"""

# 解析源代码为 AST 对象
tree = ast.parse(source_code)

# 遍历 AST 对象，找到需要修改的节点
class ModifyReturnVisitor(ast.NodeTransformer):
    def visit_Return(self, node):
        # 修改 return 语句中的表达式
        new_expr = ast.BinOp(
            left=node.value.left,
            op=ast.Add(),
            right=ast.BinOp(
                left=node.value.right,
                op=ast.Mult(),
                right=ast.Constant(n=2)
            )
        )
        node.value = new_expr
        # return self.generic_visit(node)
        return node

# 应用修改
visitor = ModifyReturnVisitor()
modified_tree = visitor.visit(tree)

# 将修改后的 AST 对象转换回 Python 代码
modified_code = astunparse.unparse(modified_tree)
print("修改后的代码：")
print(modified_code)

# 创建一个命名空间，用于执行修改后的代码
namespace = {}
exec(modified_code, namespace)

# 获取新的函数对象
new_add = namespace['add']

# 测试新的函数
result = new_add(1, 2)
print("调用新函数的结果：", result)  # 输出: 5