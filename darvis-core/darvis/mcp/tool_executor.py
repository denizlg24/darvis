import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional

from darvis.mcp.client import MCPClient


class ToolExecutionStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    raw_text: str = ""


@dataclass
class ToolExecution:
    tool_name: str
    arguments: dict[str, Any]
    status: ToolExecutionStatus = ToolExecutionStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    has_side_effects: bool = False


@dataclass
class ToolExecutorConfig:
    max_iterations: int = 5
    tool_timeout: float = 30.0


TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>\s*({.*?})\s*</tool_call>',
    re.DOTALL
)

VOICE_FEEDBACK_TEMPLATES = {
    "web_search": [
        "Let me search for that...",
        "Give me a moment to look that up...",
        "Searching the web for you...",
    ],
    "system_info": [
        "Let me check that for you...",
        "One moment...",
        "Checking the system...",
    ],
    "default": [
        "Give me a second...",
        "Let me handle that...",
        "Working on it...",
    ]
}


class ToolExecutor:
    def __init__(
        self,
        mcp_client: MCPClient,
        config: Optional[ToolExecutorConfig] = None
    ):
        self._client = mcp_client
        self._config = config or ToolExecutorConfig()
        self._executions: list[ToolExecution] = []
        self._iteration_count = 0

    @property
    def executions(self) -> list[ToolExecution]:
        return self._executions

    @property
    def iteration_count(self) -> int:
        return self._iteration_count

    def reset(self) -> None:
        self._executions = []
        self._iteration_count = 0

    def detect_tool_call(self, response: str) -> Optional[ToolCall]:
        match = TOOL_CALL_PATTERN.search(response)
        if not match:
            return None

        try:
            json_str = match.group(1)
            data = json.loads(json_str)

            name = data.get("name") or data.get("tool")
            arguments = data.get("arguments") or data.get("args") or data.get("parameters") or {}

            if not name:
                return None

            available_tools = [t.name for t in self._client.tools]
            if name not in available_tools:
                print(f"[TOOL] Unknown tool: {name}")
                return None

            return ToolCall(
                name=name,
                arguments=arguments,
                raw_text=match.group(0)
            )

        except json.JSONDecodeError as e:
            print(f"[TOOL] Failed to parse tool call JSON: {e}")
            return None

    def get_voice_feedback(self, tool_name: str) -> str:
        import random
        templates = VOICE_FEEDBACK_TEMPLATES.get(
            tool_name,
            VOICE_FEEDBACK_TEMPLATES["default"]
        )
        return random.choice(templates)

    async def execute(self, tool_call: ToolCall) -> ToolExecution:
        execution = ToolExecution(
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
            status=ToolExecutionStatus.RUNNING,
            started_at=datetime.now(),
            has_side_effects=self._has_side_effects(tool_call.name)
        )
        self._executions.append(execution)
        self._iteration_count += 1

        print(f"[TOOL] Executing: {tool_call.name}({tool_call.arguments})")

        try:
            result = await self._client.call_tool(
                tool_call.name,
                tool_call.arguments
            )

            if result is None:
                execution.status = ToolExecutionStatus.FAILED
                execution.error = "Tool execution returned no result"
            else:
                execution.status = ToolExecutionStatus.COMPLETED
                execution.result = result
                print(f"[TOOL] Result: {result[:200]}..." if len(result) > 200 else f"[TOOL] Result: {result}")

        except Exception as e:
            execution.status = ToolExecutionStatus.FAILED
            execution.error = str(e)
            print(f"[TOOL] Error: {e}")

        execution.completed_at = datetime.now()
        return execution

    def _has_side_effects(self, tool_name: str) -> bool:
        read_only_tools = {"web_search", "system_info"}
        return tool_name not in read_only_tools

    def can_continue(self) -> bool:
        return self._iteration_count < self._config.max_iterations

    def format_tool_result_for_context(self, execution: ToolExecution) -> str:
        if execution.status == ToolExecutionStatus.COMPLETED:
            return f"<tool_result>\n{execution.result}\n</tool_result>"
        else:
            error_msg = execution.error or "Unknown error"
            return f"<tool_result>\nError: {error_msg}\n</tool_result>"

    def cancel_pending(self) -> None:
        for execution in self._executions:
            if execution.status == ToolExecutionStatus.RUNNING:
                execution.status = ToolExecutionStatus.CANCELLED
                execution.completed_at = datetime.now()
