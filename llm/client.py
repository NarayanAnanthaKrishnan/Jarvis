import json
import threading
import time
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import config
from config import PROVIDER, OPENAI_API_KEY, OPENAI_MODEL, GEMINI_API_KEY, GEMINI_MODEL, ANTHROPIC_API_KEY, ANTHROPIC_MODEL
from llm.prompts import SYSTEM_PROMPT
from tools.profile_loader import load_profile
from ops.tracer import trace


_MODEL_MAP = {
    "openai": OPENAI_MODEL,
    "gemini": GEMINI_MODEL,
    "anthropic": ANTHROPIC_MODEL,
}


@dataclass(frozen=True)
class EmailModelProfile:
    model: str
    thinking_level: str
    timeout_seconds: int
    max_tokens: int


class ModelRequestError(RuntimeError):
    pass


class LLMClient:
    def __init__(self) -> None:
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
        if provider == "gemini":
            candidates = getattr(response, "candidates", None)
            reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        elif provider == "openai":
            reason = getattr(response.choices[0], "finish_reason", None)
        elif provider == "anthropic":
            reason = getattr(response, "stop_reason", None)
        else:
            reason = None
        if isinstance(reason, str):
            result["finish_reason"] = reason
        if tool_calls:
            result["message"]["tool_calls"] = tool_calls
        return result

    def _chat(self, model: str, tools: list | None = None) -> dict | None:
        providers = self._get_provider_order()
        for provider in providers:
            for attempt in range(3):
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
                    err = str(e)
                    if attempt < 2 and any(x in err for x in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "rate limit", "too many requests")):
                        wait = 2 ** attempt
                        print(f"  [~] {provider} busy (attempt {attempt+1}/3), retrying in {wait}s...")
                        time.sleep(wait)
                        continue
                    print(f"  [X] LLM call failed ({provider}): {type(e).__name__}")
                    break

        return None

    def _call_email(self, messages: list[dict], response_schema: dict | None, agent_id: str = "email") -> dict:
        from google.genai import types

        profile = EmailModelProfile(config.EMAIL_MODEL, config.EMAIL_THINKING_LEVEL,
                                    config.EMAIL_MODEL_TIMEOUT_SECONDS, config.EMAIL_MODEL_MAX_TOKENS)
        if not GEMINI_API_KEY:
            raise ModelRequestError("The email model needs GEMINI_API_KEY configured.")
        if profile.thinking_level not in {"low", "medium", "high"}:
            raise ModelRequestError("Set EMAIL_THINKING_LEVEL to low, medium, or high.")
        started = time.monotonic()
        status = "error"
        try:
            client = self._get_client("gemini")
            contents, system = self._convert_messages(messages, "gemini")
            raw = client.models.generate_content(
                model=profile.model, contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system, temperature=1.0, max_output_tokens=profile.max_tokens,
                    thinking_config=types.ThinkingConfig(thinking_level=profile.thinking_level.upper()),
                    response_mime_type="application/json" if response_schema else None,
                    response_json_schema=response_schema,
                    http_options=types.HttpOptions(timeout=profile.timeout_seconds * 1000,
                                                   retry_options=types.HttpRetryOptions(attempts=1)),
                ),
            )
            result = self._normalize_response(raw, "gemini")
            if not result.get("message", {}).get("content"):
                raise ModelRequestError("The email model returned no usable response. Please retry the request.")
            status = "complete"
            return result
        except ModelRequestError:
            raise
        except Exception as exc:
            code = getattr(exc, "code", None)
            if code == 429:
                message = "The email model reached its quota. Check Gemini billing or quota before retrying."
            elif code in (401, 403, 404):
                message = "The email model is unavailable for this account. Check EMAIL_MODEL and Gemini access."
            elif "timeout" in type(exc).__name__.lower() or code == 504:
                message = "The email model timed out. Please retry the request."
            else:
                message = "The email model could not complete the request. Check the connection and model configuration."
            trace("error", where=f"{agent_id}_model", model=profile.model, error_type=type(exc).__name__)
            raise ModelRequestError(message) from exc
        finally:
            trace("model_call", agent_id=agent_id, model=profile.model, status=status, elapsed_s=round(time.monotonic() - started, 3))

    def call_raw(self, messages: list[dict], tools: list | None = None, temp: float = 0.3, max_tokens: int = 1000,
                 *, profile: str = "general", response_schema: dict | None = None) -> dict | None:
        if profile in {"email", "calendar", "mixed"}:
            if tools:
                raise ValueError("Workspace actions use structured output rather than SDK tool execution")
            return self._call_email(messages, response_schema, profile)
        if profile != "general" or response_schema is not None:
            raise ValueError("Unsupported model profile or response schema")
        started = time.monotonic()
        providers = self._get_provider_order()
        for provider in providers:
            for attempt in range(3):
                try:
                    client = self._get_client(provider)
                    converted_messages, system_instruction = self._convert_messages(messages, provider)
                    converted_tools = self._convert_tools(tools, provider)

                    if provider == "openai":
                        kwargs = {
                            "model": self._get_model(provider),
                            "messages": converted_messages,
                            "max_tokens": max_tokens,
                            "temperature": temp,
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
                            "temperature": temp,
                        }
                        if system_instruction:
                            kwargs["system"] = system_instruction
                        if converted_tools:
                            kwargs["tools"] = converted_tools
                        raw = client.messages.create(**kwargs)

                    trace("model_call", agent_id="general", model=self._get_model(provider), status="complete", elapsed_s=round(time.monotonic() - started, 3))
                    return self._normalize_response(raw, provider)

                except Exception as e:
                    err = str(e)
                    if attempt < 2 and any(x in err for x in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "rate limit", "too many requests")):
                        wait = 2 ** attempt
                        print(f"  [~] {provider} busy (attempt {attempt+1}/3), retrying in {wait}s...")
                        time.sleep(wait)
                        continue
                    print(f"  [X] LLM call failed ({provider}): {type(e).__name__}")
                    break

        return None

    MAX_HISTORY_EXCHANGES = 20

    def reset_history(self) -> None:
        profile = load_profile()
        content = SYSTEM_PROMPT.replace("{PROFILE}", f"User profile:\n{profile}" if profile else "")
        self.history = [{"role": "system", "content": content}]

    def refresh_memories(self, query: str) -> None:
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

        if len(self.history) > self.MAX_HISTORY_EXCHANGES * 2 + 1:
            self.history = [self.history[0]] + self.history[-(self.MAX_HISTORY_EXCHANGES * 2):]
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
                        "temperature": temp,
                    }
                    if system_instruction:
                        kwargs["system"] = system_instruction
                    with client.messages.create(stream=True, **kwargs) as stream:
                        for event in stream:
                            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                                yield event.delta.text
                    return

            except Exception as e:
                print(f"  [X] Stream LLM call failed ({provider}): {type(e).__name__}")
                continue
