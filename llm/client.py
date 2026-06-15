import json
import ollama
from config import LLM_PRIMARY
from llm.prompts import SYSTEM_PROMPT


class LLMClient:
    def __init__(self):
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.client = ollama.Client(host="http://localhost:11434")

    def _chat(self, model: str, tools: list | None = None) -> dict | None:
        try:
            kwargs = {
                "model": model,
                "messages": self.history,
                "stream": False,
                "options": {"num_predict": 100, "num_ctx": 2048, "temperature": 0.3}
            }
            if tools:
                kwargs["tools"] = tools
            return self.client.chat(**kwargs)
        except Exception:
            return None

    def call_raw(self, messages: list[dict], tools: list | None = None, temp: float = 0.3) -> dict | None:
        try:
            kwargs = {
                "model": LLM_PRIMARY,
                "messages": messages,
                "stream": False,
                "options": {"num_predict": 150, "num_ctx": 1024, "temperature": temp}
            }
            if tools:
                kwargs["tools"] = tools
            return self.client.chat(**kwargs)
        except Exception:
            return None

    def chat(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        response = self._chat(LLM_PRIMARY)
        if response is None:
            reply = "[Error: Model unavailable]"
            print(f"❌ {reply}")
            self.history.pop()
            return reply

        reply = response["message"]["content"]
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def chat_with_tools(self, user_input: str, tools: list[dict]) -> str:
        from tools.registry import execute_tool

        self.history.append({"role": "user", "content": user_input})

        for _ in range(5):
            response = self._chat(LLM_PRIMARY, tools)
            if response is None:
                reply = "[Error: Model unavailable]"
                print(f"❌ {reply}")
                self.history.pop()
                return reply

            message = response["message"]

            if "tool_calls" not in message:
                reply = message["content"]
                self.history.append({"role": "assistant", "content": reply})
                return reply

            for tool_call in message["tool_calls"]:
                name = tool_call["function"]["name"]
                raw_args = tool_call["function"]["arguments"]
                if isinstance(raw_args, str):
                    args = json.loads(raw_args)
                else:
                    args = raw_args

                result = execute_tool(name, args)
                self.history.append({
                    "role": "tool",
                    "content": str(result),
                    "name": name
                })

        reply = "[Error: Too many tool call rounds]"
        self.history.pop()
        return reply
