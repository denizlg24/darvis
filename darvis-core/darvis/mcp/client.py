import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MCPConfig:
    server_path: str = "../darvis-mcp-server"
    python_executable: str = "uv"
    startup_timeout: float = 30.0
    request_timeout: float = 60.0

    @classmethod
    def from_env(cls) -> "MCPConfig":
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        default_path = os.path.join(base_dir, "darvis-mcp-server")

        return cls(
            server_path=os.environ.get("MCP_SERVER_PATH", default_path),
            python_executable=os.environ.get("MCP_PYTHON", "uv"),
            startup_timeout=float(os.environ.get("MCP_STARTUP_TIMEOUT", "30.0")),
            request_timeout=float(os.environ.get("MCP_REQUEST_TIMEOUT", "60.0")),
        )


@dataclass
class ToolSchema:
    name: str
    description: str
    input_schema: dict[str, Any]


class MCPClient:
    def __init__(self, config: Optional[MCPConfig] = None):
        self._config = config or MCPConfig.from_env()
        self._process: Optional[asyncio.subprocess.Process] = None
        self._request_id = 0
        self._pending_requests: dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._tools: list[ToolSchema] = []
        self._running = False

    @property
    def tools(self) -> list[ToolSchema]:
        return self._tools

    @property
    def is_running(self) -> bool:
        return self._running and self._process is not None

    async def start(self) -> bool:
        if self._running:
            return True

        server_path = self._config.server_path

        if not os.path.isdir(server_path):
            print(f"[MCP] Server path not found: {server_path}")
            return False

        main_py = os.path.join(server_path, "main.py")
        if not os.path.exists(main_py):
            print(f"[MCP] main.py not found in {server_path}")
            return False

        try:
            print(f"[MCP] Starting server from {server_path}...")

            if sys.platform == "win32":
                self._process = await asyncio.create_subprocess_exec(
                    self._config.python_executable, "run", "python", "main.py",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=server_path,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                self._process = await asyncio.create_subprocess_exec(
                    self._config.python_executable, "run", "python", "main.py",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=server_path
                )

            self._running = True
            self._reader_task = asyncio.create_task(self._read_responses())

            await asyncio.sleep(0.5)

            await self._initialize()

            tools_result = await self._list_tools()
            if tools_result:
                self._tools = tools_result
                print(f"[MCP] Loaded {len(self._tools)} tools: {[t.name for t in self._tools]}")
            else:
                print("[MCP] Warning: No tools loaded")

            return True

        except Exception as e:
            print(f"[MCP] Failed to start server: {e}")
            await self.stop()
            return False

    async def stop(self) -> None:
        self._running = False

        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
            self._reader_task = None

        if self._process:
            try:
                if self._process.stdin:
                    self._process.stdin.close()
                    try:
                        await self._process.stdin.wait_closed()
                    except Exception:
                        pass

                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self._process.kill()
                    await self._process.wait()
            except Exception as e:
                print(f"[MCP] Error during shutdown: {e}")
            finally:
                self._process = None

        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

        self._tools = []
        print("[MCP] Server stopped")

    async def _read_responses(self) -> None:
        if not self._process or not self._process.stdout:
            return

        try:
            while self._running:
                line = await self._process.stdout.readline()
                if not line:
                    break

                try:
                    response = json.loads(line.decode())
                    request_id = response.get("id")

                    if request_id is not None and request_id in self._pending_requests:
                        future = self._pending_requests.pop(request_id)
                        if not future.done():
                            future.set_result(response)
                except json.JSONDecodeError:
                    pass

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[MCP] Reader error: {e}")

    async def _send_request(self, method: str, params: Optional[dict] = None) -> Optional[dict]:
        if not self._process or not self._process.stdin:
            return None

        self._request_id += 1
        request_id = self._request_id

        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params:
            request["params"] = params

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        try:
            data = json.dumps(request) + "\n"
            self._process.stdin.write(data.encode())
            await self._process.stdin.drain()

            response = await asyncio.wait_for(
                future,
                timeout=self._config.request_timeout
            )
            return response

        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            print(f"[MCP] Request timeout: {method}")
            return None
        except Exception as e:
            self._pending_requests.pop(request_id, None)
            print(f"[MCP] Request error: {e}")
            return None

    async def _initialize(self) -> bool:
        response = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "darvis",
                "version": "0.1.0"
            }
        })

        if response and "result" in response:
            await self._send_request("notifications/initialized")
            return True
        return False

    async def _list_tools(self) -> list[ToolSchema]:
        response = await self._send_request("tools/list")

        if not response or "result" not in response:
            return []

        tools = []
        for tool_data in response["result"].get("tools", []):
            tools.append(ToolSchema(
                name=tool_data.get("name", ""),
                description=tool_data.get("description", ""),
                input_schema=tool_data.get("inputSchema", {})
            ))

        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Optional[str]:
        if not self._running:
            return None

        response = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments
        })

        if not response:
            return None

        if "error" in response:
            error = response["error"]
            return f"Error: {error.get('message', 'Unknown error')}"

        result = response.get("result", {})
        content = result.get("content", [])

        if content and isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            return "\n".join(text_parts)

        return str(result)

    def get_tools_for_prompt(self) -> str:
        if not self._tools:
            return ""

        tools_desc = []
        for tool in self._tools:
            params = tool.input_schema.get("properties", {})
            required = tool.input_schema.get("required", [])

            param_list = []
            for param_name, param_info in params.items():
                param_desc = param_info.get("description", "")
                param_type = param_info.get("type", "any")
                is_required = param_name in required
                req_str = " (required)" if is_required else " (optional)"
                param_list.append(f"    - {param_name}: {param_type}{req_str} - {param_desc}")

            params_str = "\n".join(param_list) if param_list else "    No parameters"

            tools_desc.append(f"""Tool: {tool.name}
Description: {tool.description}
Parameters:
{params_str}""")

        return "\n\n".join(tools_desc)
