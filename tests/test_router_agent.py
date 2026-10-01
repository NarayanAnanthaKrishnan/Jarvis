import json
from unittest.mock import Mock

import pytest

import config
from agent.context import TurnContext
from agent.json_utils import extract_object
from agent.router import RouteDecision, route_turn
from agent import loop
from email_agent.models import EmailResult
from tools.registry import execute_tool


def test_json_with_braces_and_validation() -> None:
    assert extract_object('```json\n{"answer":"Use } and { symbols"}\n```')["answer"] == "Use } and { symbols"
    for decision in [{"tool": "calculate", "args": None}, {"done": "false"}, {"parallel": []}, {"done": True, "answer": "x", "delivery": "execute"}]:
        with pytest.raises(ValueError):
            loop.validate_decision(decision)


@pytest.mark.parametrize("choice, confidence, agent", [("email", .95, "email"), ("email", .4, "general"), ("mixed", .95, "general"), ("unclear", .95, "general")])
def test_routing_and_context(monkeypatch: pytest.MonkeyPatch, choice: str, confidence: float, agent: str) -> None:
    monkeypatch.setattr(config, "JEV_ROUTING_ENABLED", True)
    client = Mock()
    probabilities = {key: float(key == choice) for key in ("email", "general", "mixed", "unclear")}
    client.system_one.return_value = {"answers": {"route": {"choice": choice, "confidence": confidence, "probabilities": probabilities}, "memory": {"noul": .1}}}
    result = route_turn("make it shorter", ["User: draft an email"], {"active_draft_id": 7}, client)
    assert result.agent_id == agent
    assert result.memory_needed is False
    assert client.system_one.call_args.kwargs["state"]["workflow"]["active_draft_id"] == 7
    assert client.system_one.call_count == 1


def test_router_disabled_failure_and_invalid_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    client = Mock()
    monkeypatch.setattr(config, "JEV_ROUTING_ENABLED", False)
    assert route_turn("email", [], {}, client).agent_id == "general"
    client.system_one.assert_not_called()
    monkeypatch.setattr(config, "JEV_ROUTING_ENABLED", True)
    client.system_one.side_effect = TimeoutError()
    assert route_turn("email", [], {}, client).memory_needed is True
    client.system_one.side_effect = None
    client.system_one.return_value = {"answers": {"route": {"choice": "send_without_confirmation", "confidence": 1}}}
    assert route_turn("email", [], {}, client).agent_id == "general"


def test_pending_email_followup_takes_precedence_over_jev(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "JEV_ROUTING_ENABLED", True)
    client = Mock()
    result = route_turn("later today 6pm", [], {"active_draft_id": 7, "action": "schedule", "awaiting": "time"}, client)
    assert result.agent_id == "email"
    assert result.fallback_reason == ""
    client.system_one.assert_not_called()


def test_tool_permissions_are_enforced() -> None:
    assert "not available" in execute_tool("paste_at_cursor", {"text": "bad"}, context=TurnContext(agent_id="email"))
    assert "read-only" in execute_tool("take_note", {"note": "bad"}, parallel=True)
    assert "invalid" in execute_tool("calculate", None)


def test_parallel_results_preserved_and_mutations_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loop, "route_turn", lambda *a: RouteDecision(memory_needed=False))
    seen = []
    decisions = iter([{"parallel": [{"tool": "take_note", "args": {"note": "bad"}}]}, {"parallel": [{"tool": "search_web", "args": {"query": "x"}}]}, {"done": True, "answer": "done"}])
    def think(*args: object) -> dict:
        seen.append(list(args[1]))
        return next(decisions)
    tool = Mock(return_value="x" * 1000 + "important ending")
    monkeypatch.setattr(loop, "think", think)
    monkeypatch.setattr(loop, "execute_tool", tool)
    loop.run_agent("query", [], Mock())
    assert tool.call_count == 1
    assert "important ending" in seen[-1][-1]["result"]


def test_handoff_and_preview_share_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loop, "route_turn", lambda *a: RouteDecision(memory_needed=False))
    decisions = iter([{"tool": "handoff_email", "args": {"instruction": "send draft"}}, {"tool": "email_prepare", "args": {"draft_id": 1, "action": "send"}}])
    monkeypatch.setattr(loop, "think", lambda *a: next(decisions))
    monkeypatch.setattr(loop, "execute_tool", lambda *a: EmailResult("awaiting_confirmation", "Review first", 1, 9, "Full preview", 9))
    context = TurnContext()
    result = loop.run_agent("send draft", [], Mock(), context)
    assert context.remaining_steps == 4
    assert context.handed_off and context.agent_id == "email"
    assert result["confirmation_id"] == 9


def test_cancellation_prevents_tool_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loop, "route_turn", lambda *a: RouteDecision(memory_needed=False))
    context = TurnContext()
    def think(*args: object) -> dict:
        context.cancelled.set()
        return {"tool": "take_note", "args": {"note": "bad"}}
    tool = Mock()
    monkeypatch.setattr(loop, "think", think)
    monkeypatch.setattr(loop, "execute_tool", tool)
    with pytest.raises(InterruptedError):
        loop.run_agent("take a note", [], Mock(), context)
    tool.assert_not_called()
