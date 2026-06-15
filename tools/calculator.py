import ast


ALLOWED_NODES = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd,
}


def calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if type(node) not in ALLOWED_NODES:
                return "Error: unsafe expression."
        result = eval(compile(tree, "", "eval"))
        return str(result)
    except Exception:
        return "Error: could not evaluate expression."
