from tools.registry import execute_tool


def execute_plan(plan: list[dict]) -> list[dict]:
    if not plan:
        return []

    results = []

    for step in plan:
        name = step.get("tool", "")
        args = step.get("args", {})

        print(f"  🔧 Calling {name}({args})")
        result = execute_tool(name, args)
        short = str(result)[:150]
        print(f"     Result: {short}")

        results.append({
            "tool": name,
            "args": args,
            "result": str(result)
        })

    return results
