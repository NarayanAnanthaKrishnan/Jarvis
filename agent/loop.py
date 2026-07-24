import json
import time
from collections.abc import Generator
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from config import MAX_STEPS, PARALLEL_WORKERS, REFLECTION_ENABLED
from agent.prompts import SYSTEM_PROMPT, USER_PROMPT_FORMAT, TOOL_DESCRIPTIONS, REFLECTION_PROMPT
from tools.registry import execute_tool
from tools.profile_loader import load_profile
from memory.retrieval_gate import needs_memory
from ops.tracer import trace


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    brace = text.find("{")
    if brace >= 0:
        depth = 0
        for i in range(brace, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[brace:i+1]
    return text


def _format_steps(steps: list[dict]) -> str:
    if not steps:
        return "(none)"
    lines = []
    for i, s in enumerate(steps, 1):
        result_preview = str(s["result"])[:200]
        lines.append(f"{i}. {s['tool']}({s['args']}) → {result_preview}")
    return "\n".join(lines)


def think(user_input: str, steps: list[dict], conversation_history: list[str], llm, memories: str = "(none)") -> dict:
    profile = load_profile()
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    conv_str = "\n".join(conversation_history[-8:]) if conversation_history else "(none)"
    steps_str = _format_steps(steps)

    system_content = (SYSTEM_PROMPT.replace("{TOOL_DESCRIPTIONS}", TOOL_DESCRIPTIONS)
                      .replace("{PROFILE}", profile or "(none)")
                      .replace("{MEMORIES}", memories)
                      .replace("{CURRENT_DATE}", current_date)
                      .replace("{CONVERSATION_HISTORY}", conv_str))

    user_content = (USER_PROMPT_FORMAT.replace("{GOAL}", user_input)
                    .replace("{STEPS}", steps_str))

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    response = llm.call_raw(messages, temp=0.1)

    if response is None:
        return {"done": True, "answer": "I could not process that request right now.", "delivery": "speak"}

    raw = _extract_json(response["message"]["content"])
    try:
        decision = json.loads(raw)
        if not isinstance(decision, dict):
            raise ValueError("not a dict")
        return decision
    except (json.JSONDecodeError, ValueError):
        raw_text = response["message"]["content"].strip()
        print(f"  ⚠ JSON parse failed, trying raw text as answer")
        if len(raw_text) > 5 and not any(c in raw_text for c in "{["):
            return {"done": True, "answer": raw_text, "delivery": "speak"}
        print(f"  Raw: {raw_text[:200]}")
        return {"done": True, "answer": "I could not process that request right now.", "delivery": "speak"}


def _reflect(user_input: str, steps: list[dict], answer: str, llm) -> dict:
    steps_str = _format_steps(steps)
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    prompt = (REFLECTION_PROMPT.replace("{CURRENT_DATE}", current_date)
              .replace("{GOAL}", user_input)
              .replace("{STEPS}", steps_str)
              .replace("{ANSWER}", answer))
    messages = [
        {"role": "system", "content": "You are a verification assistant. Return only valid JSON."},
        {"role": "user", "content": prompt}
    ]
    response = llm.call_raw(messages, temp=0.1)
    if response is None:
        return {"correct": True}
    raw = _extract_json(response["message"]["content"])
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {"correct": True}


def _stream_summarize(user_input: str, steps: list[dict], llm) -> Generator[str, None, None]:
    formatted = "\n".join(
        f"[{s['tool']}] {str(s['result'])[:300]}" for s in steps
    )
    prompt = (
        f"The user's question was: {user_input}\n\n"
        f"Tool results gathered:\n{formatted}\n\n"
        f"Answer the user's original question using the data above. "
        f"Produce a concise spoken answer in 1-2 sentences. Use exact numbers. "
        f"If the data doesn't fully answer the question, say what you know and what's missing. "
        f"No meta-commentary."
    )
    messages = [
        {"role": "system", "content": "You produce short spoken responses from tool data. Answer the user's question directly."},
        {"role": "user", "content": prompt}
    ]
    yield from llm.stream_chat(messages, temp=0.1)


def run_agent(user_input: str, conversation_history: list[str], llm) -> dict:
    steps: list[dict] = []
    start_ts = time.time()
    trace("turn_start", user_input=user_input)

    gate_needed = needs_memory(user_input, llm)
    trace("gate", needs_memory=gate_needed)
    if gate_needed:
        from memory.store import memory_store
        semantic = memory_store.query("semantic", user_input, n=5)
        episodic = memory_store.query("episodic", user_input, n=3)
        memories = "\n".join(f"- {m}" for m in (semantic + episodic)) if semantic or episodic else "(none)"
    else:
        memories = "(none)"

    for i in range(MAX_STEPS):
        decision = think(user_input, steps, conversation_history, llm, memories)

        if decision.get("done"):
            answer = decision.get("answer", "")
            delivery = decision.get("delivery", "speak")
            trace("think", step=i + 1, tool=None, done=True, reason="")

            was_llm_fallback = "could not process" in answer.lower()
            if REFLECTION_ENABLED and steps and not was_llm_fallback:
                print(f"  🔍 Reflecting on answer...")
                verdict = _reflect(user_input, steps, answer, llm)
                if not verdict.get("correct", True):
                    issue = verdict.get("issue", "answer may be incomplete")
                    hint = verdict.get("hint", "")
                    print(f"  ⚠ Reflection: {issue}")
                    steps.append({
                        "tool": "_reflection",
                        "args": {},
                        "result": f"Reflection feedback: {issue}. {hint}"
                    })
                    if i + 1 < MAX_STEPS:
                        continue

            trace("final", delivery=delivery, output_preview=answer[:200])
            elapsed = time.time() - start_ts
            trace("turn_end", steps=len(steps), elapsed_s=round(elapsed, 2))
            return {"output": answer, "delivery": delivery, "stream": None, "gate_needed": gate_needed}

        if "parallel" in decision:
            tools_list = decision["parallel"]
            reason = decision.get("reason", "")
            trace("think", step=i + 1, tool="parallel", done=False, reason=reason)
            print(f"  🔧 Step {i+1}: Parallel ({len(tools_list)} tools) — {reason}")
            results = []
            with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
                futures = {}
                for t in tools_list:
                    tn = t.get("tool", "")
                    ta = t.get("args", {})
                    futures[pool.submit(execute_tool, tn, ta, llm)] = (tn, ta)
                for f in futures:
                    tn, ta = futures[f]
                    r = f.result()
                    results.append(f"{tn} → {r[:200]}")
                    print(f"     {tn} → {str(r)[:150]}")
                    trace("tool", name=tn, args=ta, result_preview=str(r)[:200])
                    if str(r).startswith("Error"):
                        print(f"  ⚠ Tool error in {tn}. Agent will see this in next think step.")
            combined = " | ".join(results)
            steps.append({"tool": "parallel", "args": tools_list, "result": combined})
            continue

        tool = decision.get("tool", "")
        args = decision.get("args", {})
        reason = decision.get("reason", "")
        trace("think", step=i + 1, tool=tool, done=False, reason=reason)

        is_dup = any(s["tool"] == tool and s["args"] == args for s in steps)
        if is_dup:
            print(f"  ⚠ Skipping duplicate: {tool}({args}) — already executed")
            steps.append({"tool": tool, "args": args, "result": "(duplicate — already executed above)"})
            continue

        print(f"  🔧 Step {i+1}: {tool}({args}){' — ' + reason if reason else ''}")
        result = execute_tool(tool, args, llm)
        print(f"     Result: {str(result)[:150]}")
        trace("tool", name=tool, args=args, result_preview=str(result)[:200])

        is_error = str(result).startswith("Error")
        if is_error:
            print(f"  ⚠ Tool error detected. Agent will see this in next think step.")

        steps.append({"tool": tool, "args": args, "result": str(result)})

    print(f"  ⚠ MAX_STEPS ({MAX_STEPS}) reached. Generating final answer from partial data...")
    trace("final", delivery="speak", output_preview="[stream fallback]")
    elapsed = time.time() - start_ts
    trace("turn_end", steps=len(steps), elapsed_s=round(elapsed, 2))
    return {"output": "", "delivery": "speak", "stream": _stream_summarize(user_input, steps, llm), "gate_needed": gate_needed}
