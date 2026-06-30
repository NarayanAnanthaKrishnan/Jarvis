import json
import threading
from collections.abc import Generator
from config import PROVIDER, OPENAI_API_KEY, OPENAI_MODEL, GEMINI_API_KEY, GEMINI_MODEL, ANTHROPIC_API_KEY, ANTHROPIC_MODEL
from llm.prompts import SYSTEM_PROMPT
from tools.profile_loader import load_profile


_MODEL_MAP = {
    "openai": OPENAI_MODEL,
    "gemini": GEMINI_MODEL,
    "anthropic": ANTHROPIC_MODEL,
}


class LLMClient:
    def __init__(self):
        profile = load_profile()
        system_content = SYSTEM_PROMPT.replace("{PROFILE}", f"User profile:\n{profile}" if profile else "")
        self.history = [{"role": "system", "content": system_content}]
        self._clients = {}
        self._client_lock = threading.Lock()

    def _get_provider_order(self) -> list[str]:
        primary = PROVIDER
        all_providers = ["openai", "gemini", "anthropic"]
        ordered = [primary] + [p for p in all_providers if p != primary]
        key_map = {
            "openai": OPENAI_API_KEY,
            "gemini": GEMINI_API_KEY,
            "anthropic": ANTHROPIC_API_KEY,
        }
        return [p for p in ordered if key_map.get(p)]

    def _get_client(self, provider: str):
        cached = self._clients.get(provider)
        if cached is not None:
            return cached

        with self._client_lock:
            if provider in self._clients:
                return self._clients[provider]
            if provider == "openai":
                from openai import OpenAI
                self._clients["openai"] = OpenAI(api_key=OPENAI_API_KEY)
            elif provider == "gemini":
                from google import genai
                self._clients["gemini"] = genai.Client(api_key=GEMINI_API_KEY)
            elif provider == "anthropic":
                from anthropic import Anthropic
                self._clients["anthropic"] = Anthropic(api_key=ANTHROPIC_API_KEY)

        return self._clients[provider]

    @staticmethod
    def _get_model(provider: str) -> str:
        return _MODEL_MAP.get(provider, "gpt-4o-mini")

    @staticmethod
    def _convert_messages(messages: list, provider: str) -> tuple[list, str | None]:
        system_instruction = None
        result = []

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                if provider in ("gemini", "anthropic"):
                    system_instruction = content
                    continue
                result.append(msg)
                continue

            if provider == "openai":
                entry: dict = {"role": role, "content": content}
                if role == "assistant" and "tool_calls" in msg:
                    entry["tool_calls"] = msg["tool_calls"]
                if role == "tool":
                    entry["tool_call_id"] = msg.get("tool_call_id", "")
                result.append(entry)

            elif provider == "gemini":
                from google.genai import types
                if role == "user":
                    result.append(types.Content(parts=[types.Part(text=content)], role="user"))
                elif role == "assistant":
                    parts: list = []
                    if content:
                        parts.append(types.Part(text=content))
                    for tc in msg.get("tool_calls", []):
                        raw = tc["function"]["arguments"]
                        args = json.loads(raw) if isinstance(raw, str) else raw
                        parts.append(types.Part.from_function_call(
                            name=tc["function"]["name"], args=args
                        ))
                    result.append(types.Content(parts=parts, role="model"))
                elif role == "tool":
                    result.append(types.Content(
                        parts=[types.Part.from_function_response(
                            name=msg.get("name", ""),
                            response={"result": content}
                        )],
                        role="function"
                    ))

            elif provider == "anthropic":
                if role == "user":
                    result.append({
                        "role": "user",
                        "content": [{"type": "text", "text": content}]
                    })
                elif role == "assistant":
                    blocks: list = []
                    if content:
                        blocks.append({"type": "text", "text": content})
                    for tc in msg.get("tool_calls", []):
                        raw = tc["function"]["arguments"]
                        args = json.loads(raw) if isinstance(raw, str) else raw
                        blocks.append({
                            "type": "tool_use",
                            "id": tc.get("id", ""),
                            "name": tc["function"]["name"],
                            "input": args
                        })
                    result.append({"role": "assistant", "content": blocks})
                elif role == "tool":
                    result.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": content
                        }]
                    })

        return result, system_instruction

    @staticmethod
    def _convert_tools(tools: list | None, provider: str) -> list | None:
        if not tools:
            return None

        if provider == "openai":
            return tools

        converted = []
        for t in tools:
            if t.get("type") != "function":
                continue
            fn = t.get("function", {})
            if provider == "gemini":
                from google.genai import types
                converted.append(types.FunctionDeclaration(
                    name=fn["name"],
                    description=fn.get("description", ""),
                    parameters=fn.get("parameters", {}),
                ))
            elif provider == "anthropic":
                converted.append({
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {})
                })

        if provider == "gemini":
            from google.genai import types
            return [types.Tool(function_declarations=converted)]
        return converted

    @staticmethod
    def _normalize_response(response, provider: str) -> dict:
        if provider == "openai":
            choice = response.choices[0]
            content = choice.message.content or ""
            tool_calls_raw = choice.message.tool_calls or []
            tool_calls = []
            for tc in tool_calls_raw:
                tool_calls.append({
                    "id": tc.id,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                })

        elif provider == "gemini":
            content = ""
            tool_calls = []
            try:
                content = response.text or ""
            except (ValueError, AttributeError):
                pass
            try:
                candidate = response.candidates[0]
                if candidate and candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.function_call:
                            fc = part.function_call
                            args = dict(fc.args) if fc.args else {}
                            tool_calls.append({
                                "id": fc.id or fc.name,
                                "function": {
                                    "name": fc.name,
                                    "arguments": json.dumps(args)
                                }
                            })
            except (AttributeError, IndexError, TypeError):
                pass

        elif provider == "anthropic":
            content = ""
            tool_calls = []
            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    input_dict = dict(block.input) if hasattr(block.input, "items") else block.input
                    tool_calls.append({
                        "id": block.id,
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(input_dict)
                        }
                    })

        else:
            content = ""
            tool_calls = []

        result = {"message": {"content": content}}
        if tool_calls:
            result["message"]["tool_calls"] = tool_calls
        return result

    def _chat(self, model: str, tools: list | None = None) -> dict | None:
        providers = self._get_provider_order()
        for provider in providers:
            try:
                client = self._get_client(provider)
                messages, system_instruction = self._convert_messages(self.history, provider)
                converted_tools = self._convert_tools(tools, provider)

                if provider == "openai":
                    kwargs = {
                        "model": self._get_model(provider),
                        "messages": messages,
                        "max_tokens": 300,
                    }
                    if converted_tools:
                        kwargs["tools"] = converted_tools
                    raw = client.chat.completions.create(**kwargs)

                elif provider == "gemini":
                    from google.genai import types
                    raw = client.models.generate_content(
                        model=self._get_model(provider),
                        contents=messages,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            tools=converted_tools,
                            temperature=0.3,
                            max_output_tokens=300,
                        ),
                    )

                elif provider == "anthropic":
                    kwargs = {
                        "model": self._get_model(provider),
                        "messages": messages,
                        "max_tokens": 300,
                    }
                    if system_instruction:
                        kwargs["system"] = system_instruction
                    if converted_tools:
                        kwargs["tools"] = converted_tools
                    raw = client.messages.create(**kwargs)

                return self._normalize_response(raw, provider)

            except Exception as e:
                print(f"  [X] LLM call failed ({provider}): {e}")
                continue

        return None

    def call_raw(self, messages: list[dict], tools: list | None = None, temp: float = 0.3, max_tokens: int = 1000) -> dict | None:
        providers = self._get_provider_order()
        for provider in providers:
            try:
                client = self._get_client(provider)
                converted_messages, system_instruction = self._convert_messages(messages, provider)
                converted_tools = self._convert_tools(tools, provider)

                if provider == "openai":
                    kwargs = {
                        "model": self._get_model(provider),
                        "messages": converted_messages,
                        "max_tokens": max_tokens,
                    }
                    if converted_tools:
                        kwargs["tools"] = converted_tools
                    raw = client.chat.completions.create(**kwargs)

                elif provider == "gemini":
                    from google.genai import types
                    raw = client.models.generate_content(
                        model=self._get_model(provider),
                        contents=converted_messages,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            tools=converted_tools,
                            temperature=temp,
                            max_output_tokens=max_tokens,
                        ),
                    )

                elif provider == "anthropic":
                    kwargs = {
                        "model": self._get_model(provider),
                        "messages": converted_messages,
                        "max_tokens": max_tokens,
                    }
                    if system_instruction:
                        kwargs["system"] = system_instruction
                    if converted_tools:
                        kwargs["tools"] = converted_tools
                    raw = client.messages.create(**kwargs)

                return self._normalize_response(raw, provider)

            except Exception as e:
                print(f"  [X] LLM call failed ({provider}): {e}")
                continue

        return None

    MAX_HISTORY_EXCHANGES = 20

    def refresh_memories(self, query: str):
        from memory.store import memory_store
        semantic = memory_store.query("semantic", query, n=5)
        episodic = memory_store.query("episodic", query, n=3)
        memories = ""
        if semantic or episodic:
            items = [f"- {m}" for m in semantic + episodic]
            memories = "Relevant memories:\n" + "\n".join(items)
        profile = load_profile()
        system_content = SYSTEM_PROMPT.replace("{PROFILE}", f"User profile:\n{profile}" if profile else "")
        if memories:
            system_content += f"\n\n{memories}"
        self.history[0] = {"role": "system", "content": system_content}

    def rotate_session(self) -> str:
        from memory.session import summarize
        from memory.store import memory_store
        if len(self.history) <= 1:
            return ""
        summary = summarize(self, self.history)
        if summary:
            memory_store.add("episodic", summary, metadata={"type": "session"})
        profile = load_profile()
        system_content = SYSTEM_PROMPT.replace("{PROFILE}", f"User profile:\n{profile}" if profile else "")
        self.history = [{"role": "system", "content": system_content}]
        return summary

    def chat(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})
        response = self._chat(PROVIDER)
        if response is None:
            reply = "[Error: Model unavailable]"
            print(f"[X] {reply}")
            self.history.pop()
            return reply

        reply = response["message"]["content"]
        self.history.append({"role": "assistant", "content": reply})

        if len(self.history) > self.MAX_HISTORY_EXCHANGES * 2:
            self.history = [self.history[0]] + self.history[-(self.MAX_HISTORY_EXCHANGES * 2 - 1):]
        return reply

    def chat_with_tools(self, user_input: str, tools: list[dict]) -> str:
        from tools.registry import execute_tool

        self.history.append({"role": "user", "content": user_input})

        for _ in range(5):
            response = self._chat(PROVIDER, tools)
            if response is None:
                reply = "[Error: Model unavailable]"
                print(f"[X] {reply}")
                self.history.pop()
                return reply

            message = response["message"]

            if "tool_calls" not in message:
                reply = message["content"]
                self.history.append({"role": "assistant", "content": reply})
                return reply

            assistant_msg: dict = {"role": "assistant", "content": message.get("content", "")}
            normalized_tc = []
            for tc in message["tool_calls"]:
                normalized_tc.append({
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"]
                    }
                })
            assistant_msg["tool_calls"] = normalized_tc
            self.history.append(assistant_msg)

            for tc in message["tool_calls"]:
                name = tc["function"]["name"]
                raw_args = tc["function"]["arguments"]
                tool_call_id = tc.get("id", "")
                if isinstance(raw_args, str):
                    args = json.loads(raw_args)
                else:
                    args = raw_args

                result = execute_tool(name, args)
                self.history.append({
                    "role": "tool",
                    "content": str(result),
                    "name": name,
                    "tool_call_id": tool_call_id,
                })

        reply = "[Error: Too many tool call rounds]"
        self.history.pop()
        return reply

    def stream_chat(self, messages: list[dict], temp: float = 0.3, max_tokens: int = 1000) -> Generator[str, None, None]:
        providers = self._get_provider_order()
        for provider in providers:
            try:
                client = self._get_client(provider)
                converted_messages, system_instruction = self._convert_messages(messages, provider)

                if provider == "openai":
                    stream = client.chat.completions.create(
                        model=self._get_model(provider),
                        messages=converted_messages,
                        max_tokens=max_tokens,
                        temperature=temp,
                        stream=True,
                    )
                    for chunk in stream:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            yield delta.content
                    return

                elif provider == "gemini":
                    from google.genai import types
                    stream = client.models.generate_content_stream(
                        model=self._get_model(provider),
                        contents=converted_messages,
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction,
                            temperature=temp,
                            max_output_tokens=max_tokens,
                        ),
                    )
                    for chunk in stream:
                        if chunk.text:
                            yield chunk.text
                    return

                elif provider == "anthropic":
                    kwargs = {
                        "model": self._get_model(provider),
                        "messages": converted_messages,
                        "max_tokens": max_tokens,
                    }
                    if system_instruction:
                        kwargs["system"] = system_instruction
                    with client.messages.create(stream=True, **kwargs) as stream:
                        for event in stream:
                            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                                yield event.delta.text
                    return

            except Exception as e:
                print(f"  [X] Stream LLM call failed ({provider}): {e}")
                continue
