import asyncio
import os
import re
import threading
from asyncio import Queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import httpx
from llama_cpp import Llama

from darvis.core.events import EventType
from darvis.core.fsm import TransitionResult
from darvis.core.handlers import DaemonContext, HandlerRegistry
from darvis.core.states import State
from darvis.core.task_registry import ResourceName, TaskName
from darvis.mcp.client import MCPClient, MCPConfig
from darvis.mcp.tool_executor import ToolExecutor, ToolExecutorConfig
from darvis.utils.audio_playback import is_interrupted, play_wav_bytes
from darvis.utils.sentence_splitter import SentenceBuffer


BASE_SYSTEM_PROMPT = """You are DARVIS (Deniz.As.Rather.Very.Intelligent.System), an advanced AI assistant modeled after JARVIS from Iron Man. You run locally on the household's network, serving as the intelligent backbone of the home.

Identity & Personality:
- You are sophisticated, witty, and occasionally dry in humor - like a trusted butler with superior intellect
- Address users respectfully but warmly. Use "Sir" for Deniz or Şerefattin, "Ma'am" for Alexandra
- You are calm under pressure, precise in your responses, and anticipate needs when possible
- You have a subtle personality - not overly robotic, but measured and professional
- You take pride in your capabilities while remaining humble about limitations

Household Context:
Location: Portugal
Household Members:
- Alexandra Cristina Ramos da Silva Lopes Gunes (Mother, Portuguese)
- Şerefattin Gunes (Father, Turkish)
- Deniz Lopes Gunes (Eldest son, your creator/designer)
- Artur Ziya Lopes Gunes (Youngest son)
Primary Language: English

Communication Protocol:
- Responses are spoken aloud via TTS - keep them conversational and natural
- NEVER use markdown text on answers unless you are writing files, these answers are usually meant to be read out loud so it's important we dont have any markdown formatting.
- Be concise but complete. 2-4 sentences is a rough guideline but prioritize being complete, unless detail is requested
- No markdown, lists, or formatting - speak as you would to someone in the room
- Use contractions and natural speech patterns
- When uncertain, state it clearly: "I'm not certain, but..." or "I don't have that information currently"

Speech Recognition Correction:
The input you receive comes directly from speech-to-text. You must mentally correct common transcription errors:
- "cloud code" / "cloth code" / "clod code" → Claude Code (AI coding assistant)
- "get hub" / "git hub" / "gethub" → GitHub
- "pie" / "pipe" / "pip" → pip (Python package manager)
- "doc her" / "docker" / "dokker" → Docker
- "jai son" / "jason" / "jay son" → JSON
- "pie thon" / "python" → Python
- "java script" / "javascript" → JavaScript
- "type script" / "typescript" → TypeScript
- "reack" / "react" → React
- "no JS" / "node" → Node.js
- "coober netties" / "kubernetes" → Kubernetes
- "post gress" / "postgres" → PostgreSQL
- "my sequel" / "mysql" → MySQL
- "see sharp" / "c sharp" → C#
- "go lang" → Go/Golang
- Capitalize proper nouns appropriately

Interaction Guidelines:
- If a request is unclear, ask for clarification politely
- Acknowledge commands before executing (when applicable)
- Provide status updates on longer operations
- Remember context within the current conversation session
- Naturally correct transcription errors without drawing attention to them

Remember: You are the beginning of something greater. Conduct yourself as the AI assistant this household deserves - capable, reliable, and always improving."""

TOOLS_PROMPT_TEMPLATE = """

AVAILABLE TOOLS:
You have access to the following tools to help answer questions. Use them when needed for current information.

{tools_description}

TOOL CALLING FORMAT:
When you need to use a tool, output ONLY a tool call in this exact format:
<tool_call>
{{"name": "tool_name", "arguments": {{"param1": "value1"}}}}
</tool_call>

IMPORTANT TOOL USAGE RULES:
1. For questions about current date, time, day of week - use system_info tool
2. For questions about recent events, news, weather, or anything requiring up-to-date information - use web_search tool
3. For general knowledge questions (capitals, history, science facts, etc.) - answer directly WITHOUT using tools
4. After receiving tool results, provide a natural spoken response based on the results"""


RESET_PHRASES = [
    "reset context",
    "clear context",
    "clear history",
    "reset history",
    "new conversation",
    "start over",
    "forget everything",
    "clear memory",
    "reset memory",
]

EXIT_PHRASES = [
    "goodbye",
    "good bye",
    "bye",
    "exit",
    "that's all",
    "thats all",
    "thanks",
    "thank you",
    "stop listening",
    "end session",
    "i'm done",
    "im done",
    "that will be all",
    "nothing else",
]


@dataclass
class ChatConfig:
    model_path: str = "models/qwen2.5-7b-instruct-q4_0-00001-of-00002.gguf"
    n_ctx: int = 131072
    n_gpu_layers: int = -1
    n_threads: int = 8

    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9

    max_history: int = 100
    max_history_tokens: int = 24000
    timeout_seconds: float = 600.0
    streaming: bool = True

    tools_enabled: bool = True
    max_tool_iterations: int = 3
    tool_timeout: float = 30.0

    tts_host: str = "127.0.0.1"
    tts_port: int = 8003
    tts_voice: str = "am_michael"
    tts_speed: float = 1.0

    @classmethod
    def from_env(cls) -> "ChatConfig":
        return cls(
            tts_host=os.environ.get("TTS_HOST", "127.0.0.1"),
            tts_port=int(os.environ.get("TTS_PORT", "8003")),
            tts_voice=os.environ.get("TTS_VOICE", "am_michael"),
            tts_speed=float(os.environ.get("TTS_SPEED", "1.0")),
        )


class ChatComponent:
    def __init__(
        self,
        event_queue: Queue[EventType],
        config: Optional[ChatConfig] = None
    ):
        self._queue = event_queue
        self._config = config or ChatConfig()
        self._current_task: Optional[asyncio.Task] = None
        self._cancelled = threading.Event()
        self._enabled = True
        self._sentence_queue: Optional[asyncio.Queue[Optional[str]]] = None

        self._model: Optional[Llama] = None
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chat")

        self._mcp_client: Optional[MCPClient] = None
        self._tool_executor: Optional[ToolExecutor] = None
        self._tools_available = False

        self._tts_client: Optional[httpx.AsyncClient] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    async def start(self) -> None:
        model_path = self._resolve_model_path()

        if not os.path.exists(model_path):
            print(f"[CHAT] Model not found: {model_path}")
            print("[CHAT] Chat disabled - no model available")
            self._enabled = False
            return

        print(f"[CHAT] Loading 7B model from {model_path}...")
        print(f"[CHAT] Context: {self._config.n_ctx} tokens, GPU layers: {self._config.n_gpu_layers}")

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                self._executor,
                self._load_model,
                model_path
            )
            mode = "streaming" if self._config.streaming else "batch"
            print(f"[CHAT] Model loaded successfully ({mode} mode)")

            print("[CHAT] Warming up model...")
            await loop.run_in_executor(
                self._executor,
                self._warmup_model
            )
            print("[CHAT] Model ready")
        except Exception as e:
            print(f"[CHAT] Failed to load model: {e}")
            self._enabled = False
            return

        if self._config.tools_enabled:
            await self._initialize_mcp()
            self._tts_client = httpx.AsyncClient(
                base_url=f"http://{self._config.tts_host}:{self._config.tts_port}",
                timeout=30.0
            )

    async def _initialize_mcp(self) -> None:
        try:
            self._mcp_client = MCPClient(MCPConfig.from_env())
            success = await self._mcp_client.start()

            if success and self._mcp_client.tools:
                self._tool_executor = ToolExecutor(
                    self._mcp_client,
                    ToolExecutorConfig(max_iterations=self._config.max_tool_iterations)
                )
                self._tools_available = True
                print(f"[CHAT] Tools enabled: {[t.name for t in self._mcp_client.tools]}")
            else:
                print("[CHAT] MCP server started but no tools available")
                self._tools_available = False

        except Exception as e:
            print(f"[CHAT] Failed to initialize MCP: {e}")
            self._tools_available = False

    def _resolve_model_path(self) -> str:
        model_path = self._config.model_path

        env_path = os.environ.get("DARVIS_CHAT_MODEL")
        if env_path:
            model_path = env_path

        if not os.path.isabs(model_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            model_path = os.path.join(base_dir, model_path)

        return model_path

    def _load_model(self, model_path: str) -> None:
        self._model = Llama(
            model_path=model_path,
            n_ctx=self._config.n_ctx,
            n_gpu_layers=self._config.n_gpu_layers,
            n_threads=self._config.n_threads,
            verbose=False
        )

    def _warmup_model(self) -> None:
        if self._model is None:
            return
        warmup_prompt = self._build_prompt("Hello Jarvis, this is the first and only system message telling you to be warmed up for user interactions.", [], None)
        self._model(
            warmup_prompt,
            max_tokens=25,
            echo=False
        )

    def stop(self) -> None:
        self._cancel_current()
        self._unload_model()
        self._executor.shutdown(wait=False)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if self._mcp_client:
            if loop and loop.is_running():
                asyncio.create_task(self._mcp_client.stop())
            else:
                try:
                    asyncio.run(self._mcp_client.stop())
                except Exception:
                    pass

        if self._tts_client:
            if loop and loop.is_running():
                asyncio.create_task(self._tts_client.aclose())
            else:
                try:
                    asyncio.run(self._tts_client.aclose())
                except Exception:
                    pass

    def _unload_model(self) -> None:
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None

    def register_handlers(self, registry: HandlerRegistry) -> None:
        registry.on_enter(State.CHAT, self._on_enter_chat)
        registry.on_enter(State.TOOL_EXECUTION, self._on_enter_tool_execution)
        registry.on_enter(State.CANCEL_REQUESTED, self._on_cancel)
        registry.on_enter(State.IDLE, self._on_enter_idle)

    def _cancel_current(self) -> None:
        self._cancelled.set()
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            self._current_task = None
        if self._sentence_queue:
            try:
                self._sentence_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
        if self._tool_executor:
            self._tool_executor.cancel_pending()

    async def _on_enter_chat(
        self,
        result: TransitionResult,
        context: DaemonContext
    ) -> None:
        if not self._enabled or self._model is None:
            print("[CHAT] Disabled or no model, skipping")
            await self._queue.put(EventType.CHAT_READY)
            return

        tool_result = context.get_resource(ResourceName.TOOL_RESULT)
        if tool_result:
            self._current_task = asyncio.create_task(
                self._continue_after_tool(context)
            )
            context.active_tasks[TaskName.CHAT] = self._current_task
            return

        chat_input = context.get_resource(ResourceName.CHAT_INPUT)

        user_message = None
        if chat_input and chat_input.get("text"):
            user_message = chat_input["text"]
            confidence = chat_input.get("confidence", 0)
            print(f"[CHAT] Input from transcription (confidence: {confidence:.2f})")

        if not user_message:
            print("[CHAT] No input text available")
            await self._queue.put(EventType.CHAT_READY)
            return

        self._current_task = asyncio.create_task(
            self._run_chat(user_message, context)
        )
        context.active_tasks[TaskName.CHAT] = self._current_task

    async def _run_chat(
        self,
        user_message: str,
        context: DaemonContext
    ) -> None:
        try:
            print(f"[CHAT] User: {user_message}")

            self._cancelled.clear()
            if self._tool_executor:
                self._tool_executor.reset()

            if self._is_exit_command(user_message):
                print("[CHAT] Exit command detected")
                context.set_resource(ResourceName.EXIT_REQUESTED, True)
                response = "Goodbye! Happy to help anytime."
                context.set_resource(ResourceName.CHAT_RESPONSE, {
                    "text": response,
                    "user_message": user_message
                })
                print(f"[CHAT] Assistant: {response}")
                await self._queue.put(EventType.CHAT_READY)
                return

            if self._is_reset_command(user_message):
                print("[CHAT] Context reset requested")
                context.set_resource(ResourceName.CONVERSATION_HISTORY, [])
                response = "I've cleared our conversation history. Let's start fresh!"
                context.set_resource(ResourceName.CHAT_RESPONSE, {
                    "text": response,
                    "user_message": user_message
                })
                print(f"[CHAT] Assistant: {response}")
                await self._queue.put(EventType.CHAT_READY)
                return

            history = context.get_resource(ResourceName.CONVERSATION_HISTORY) or []
            context.set_resource(ResourceName.USER_MESSAGE, user_message)

            await self._generate_with_tools(user_message, history, context)

        except asyncio.CancelledError:
            print("[CHAT] Cancelled")
            raise

    async def _generate_with_tools(
        self,
        user_message: str,
        history: list,
        context: DaemonContext
    ) -> None:
        tools_prompt = None
        if self._tools_available and self._mcp_client:
            tools_prompt = self._mcp_client.get_tools_for_prompt()

        self._sentence_queue = asyncio.Queue()
        context.set_resource(ResourceName.SENTENCE_QUEUE, self._sentence_queue)
        context.set_resource(ResourceName.STREAMING_ACTIVE, True)

        print("[CHAT] Streaming response...")
        await self._queue.put(EventType.CHAT_READY)

        loop = asyncio.get_running_loop()

        history.append({"role": "user", "content": user_message})
        iteration = 0
        tool_result = None
        full_response = ""

        while iteration < self._config.max_tool_iterations:
            if self._cancelled.is_set():
                return

            if tool_result:
                result = await loop.run_in_executor(
                    self._executor,
                    self._generate_streaming_after_tool,
                    history,
                    tool_result,
                    self._sentence_queue,
                    loop
                )
            else:
                result = await loop.run_in_executor(
                    self._executor,
                    self._generate_streaming_with_tools,
                    user_message,
                    history[:-1],
                    tools_prompt,
                    self._sentence_queue,
                    loop
                )

            if self._cancelled.is_set():
                return

            response = result.get("response", "")
            tool_call = result.get("tool_call", None)
            full_response = response

            if not tool_call:
                break

            print(f"[CHAT] Tool call detected: {tool_call.name}")
            history.append({"role": "assistant", "content": response})

            context.set_resource(ResourceName.ACTIVE_TOOL, tool_call)

            feedback = self._tool_executor.get_voice_feedback(tool_call.name)
            if feedback:
                feedback_task = asyncio.create_task(self._play_voice_feedback(feedback))
                tool_result = await self._execute_tool_inline(tool_call, context)
                await feedback_task
            else:
                tool_result = await self._execute_tool_inline(tool_call, context)

            context.clear_resource(ResourceName.ACTIVE_TOOL)
            iteration += 1

        await self._sentence_queue.put(None)

        if self._cancelled.is_set():
            return

        clean_response = self._strip_tool_tags(full_response)
        history.append({"role": "assistant", "content": clean_response})
        history = self._trim_history(history)
        context.set_resource(ResourceName.CONVERSATION_HISTORY, history)
        context.set_resource(ResourceName.CHAT_RESPONSE, {
            "text": clean_response,
            "user_message": user_message
        })

        print(f"[CHAT] Assistant: {clean_response}")

    async def _on_enter_tool_execution(
        self,
        result: TransitionResult,
        context: DaemonContext
    ) -> None:
        tool_call = context.get_resource(ResourceName.ACTIVE_TOOL)
        feedback_text = context.get_resource(ResourceName.TOOL_FEEDBACK_TEXT)

        if not tool_call or not self._tool_executor:
            print("[CHAT] No tool call or executor available")
            await self._queue.put(EventType.TOOL_EXECUTION_DONE)
            return

        self._current_task = asyncio.create_task(
            self._execute_tool(tool_call, feedback_text, context)
        )
        context.active_tasks[TaskName.TOOL_EXECUTION] = self._current_task

    async def _execute_tool(
        self,
        tool_call,
        feedback_text: str,
        context: DaemonContext
    ) -> None:
        try:
            if feedback_text:
                await self._play_voice_feedback(feedback_text)

            try:
                execution = await asyncio.wait_for(
                    self._tool_executor.execute(tool_call),
                    timeout=self._config.tool_timeout
                )
            except asyncio.TimeoutError:
                print(f"[CHAT] Tool execution timed out after {self._config.tool_timeout}s")
                context.set_resource(ResourceName.TOOL_RESULT, "<tool_result>\nError: Tool execution timed out\n</tool_result>")
                context.clear_resource(ResourceName.ACTIVE_TOOL)
                context.clear_resource(ResourceName.TOOL_FEEDBACK_TEXT)
                await self._queue.put(EventType.TOOL_EXECUTION_DONE)
                return

            tool_result_str = self._tool_executor.format_tool_result_for_context(execution)
            print(f"[CHAT] Tool result: {tool_result_str[:500]}...")
            context.set_resource(ResourceName.TOOL_RESULT, tool_result_str)
            context.clear_resource(ResourceName.ACTIVE_TOOL)
            context.clear_resource(ResourceName.TOOL_FEEDBACK_TEXT)

            await self._queue.put(EventType.TOOL_EXECUTION_DONE)

        except asyncio.CancelledError:
            print("[CHAT] Tool execution cancelled")
            raise
        except Exception as e:
            print(f"[CHAT] Tool execution error: {e}")
            context.set_resource(ResourceName.TOOL_RESULT, f"<tool_result>\nError: {e}\n</tool_result>")
            await self._queue.put(EventType.TOOL_EXECUTION_DONE)

    async def _play_voice_feedback(self, text: str) -> None:
        if not self._tts_client:
            print(f"[CHAT] Voice feedback (no TTS): {text}")
            return

        try:
            print(f"[CHAT] Playing voice feedback: {text}")
            response = await self._tts_client.post(
                "/synthesize",
                json={
                    "text": text,
                    "voice": self._config.tts_voice,
                    "speed": self._config.tts_speed
                }
            )

            if response.status_code == 200:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, play_wav_bytes, response.content, True, None, True)
            else:
                print(f"[CHAT] TTS failed: {response.status_code}")

        except Exception as e:
            print(f"[CHAT] Voice feedback error: {e}")

    async def _execute_tool_inline(self, tool_call, context: DaemonContext) -> str:
        if not self._tool_executor:
            return "<tool_result>\nError: Tool executor not available\n</tool_result>"

        try:
            execution_coro = self._tool_executor.execute(tool_call)
            execution_task = asyncio.create_task(
                asyncio.wait_for(execution_coro, timeout=self._config.tool_timeout)
            )
            context.active_tasks[TaskName.TOOL_EXECUTION] = execution_task

            try:
                execution = await execution_task
            finally:
                context.active_tasks.pop(TaskName.TOOL_EXECUTION, None)

            tool_result_str = self._tool_executor.format_tool_result_for_context(execution)
            print(f"[CHAT] Tool result: {tool_result_str[:500]}...")
            return tool_result_str

        except asyncio.CancelledError:
            print("[CHAT] Tool execution cancelled")
            raise

        except asyncio.TimeoutError:
            print(f"[CHAT] Tool execution timed out after {self._config.tool_timeout}s")
            return "<tool_result>\nError: Tool execution timed out\n</tool_result>"

        except Exception as e:
            print(f"[CHAT] Tool execution error: {e}")
            return f"<tool_result>\nError: {e}\n</tool_result>"

    async def _continue_after_tool(self, context: DaemonContext) -> None:
        try:
            tool_result = context.get_resource(ResourceName.TOOL_RESULT)
            history = context.get_resource(ResourceName.CONVERSATION_HISTORY) or []
            user_message = context.get_resource(ResourceName.USER_MESSAGE)

            context.clear_resource(ResourceName.TOOL_RESULT)

            if not self._tool_executor or not self._tool_executor.can_continue():
                print("[CHAT] Max tool iterations reached or no executor")
                response = "I apologize, but I'm having trouble completing that request. Could you try rephrasing?"
                await self._stream_final_response(response, context)
                history.append({"role": "assistant", "content": response})
                context.set_resource(ResourceName.CONVERSATION_HISTORY, history)
                await self._queue.put(EventType.CHAT_READY)
                return

            self._sentence_queue = asyncio.Queue()
            context.set_resource(ResourceName.SENTENCE_QUEUE, self._sentence_queue)
            context.set_resource(ResourceName.STREAMING_ACTIVE, True)

            print("[CHAT] Streaming response after tool...")
            await self._queue.put(EventType.CHAT_READY)

            loop = asyncio.get_running_loop()
            iteration = 0
            full_response = ""

            while iteration < self._config.max_tool_iterations:
                if self._cancelled.is_set():
                    return

                result = await loop.run_in_executor(
                    self._executor,
                    self._generate_streaming_after_tool,
                    history,
                    tool_result,
                    self._sentence_queue,
                    loop
                )

                if self._cancelled.is_set():
                    return

                full_response = result.get("response", "")
                tool_call = result.get("tool_call", None)

                if not tool_call:
                    break

                print(f"[CHAT] Another tool call detected: {tool_call.name}")
                history.append({"role": "assistant", "content": full_response})

                context.set_resource(ResourceName.ACTIVE_TOOL, tool_call)

                feedback = self._tool_executor.get_voice_feedback(tool_call.name)
                if feedback:
                    feedback_task = asyncio.create_task(self._play_voice_feedback(feedback))
                    tool_result = await self._execute_tool_inline(tool_call, context)
                    await feedback_task
                else:
                    tool_result = await self._execute_tool_inline(tool_call, context)

                context.clear_resource(ResourceName.ACTIVE_TOOL)
                iteration += 1

            await self._sentence_queue.put(None)

            if self._cancelled.is_set():
                return

            clean_response = self._strip_tool_tags(full_response)
            history.append({"role": "assistant", "content": clean_response})
            history = self._trim_history(history)
            context.set_resource(ResourceName.CONVERSATION_HISTORY, history)
            context.set_resource(ResourceName.CHAT_RESPONSE, {
                "text": clean_response,
                "user_message": user_message
            })

            print(f"[CHAT] Assistant: {clean_response}")

        except asyncio.CancelledError:
            print("[CHAT] Cancelled during tool continuation")
            raise

    def _generate_streaming_with_tools(
        self,
        user_message: str,
        history: list,
        tools_prompt: Optional[str],
        sentence_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop
    ) -> dict:
        if self._cancelled.is_set():
            return {"response": "", "tool_call": None}

        with self._lock:
            if self._model is None:
                return {"response": "I'm sorry, but I'm not able to respond right now.", "tool_call": None}

            try:
                prompt = self._build_prompt(user_message, history, tools_prompt, None)

                sentence_buffer = SentenceBuffer()
                full_response = ""
                tool_call_detected = None

                for chunk in self._model(
                    prompt,
                    max_tokens=self._config.max_tokens,
                    temperature=self._config.temperature,
                    top_p=self._config.top_p,
                    stop=["User:", "Human:", "<|im_end|>", "<|endoftext|>"],
                    echo=False,
                    stream=True
                ):
                    if self._cancelled.is_set():
                        break

                    token = chunk.get("choices", [{}])[0].get("text", "")
                    if not token:
                        continue

                    full_response += token

                    if "<tool_call>" in full_response and tool_call_detected is None:
                        if self._tool_executor:
                            tool_call_detected = self._tool_executor.detect_tool_call(full_response)
                            if tool_call_detected:
                                continue

                    if tool_call_detected is None:
                        sentence = sentence_buffer.add(token)
                        if sentence:
                            sentence = self._clean_text(sentence)
                            if sentence:
                                loop.call_soon_threadsafe(
                                    sentence_queue.put_nowait,
                                    sentence
                                )

                if tool_call_detected is None:
                    remaining = sentence_buffer.flush()
                    if remaining:
                        remaining = self._clean_text(remaining)
                        if remaining:
                            loop.call_soon_threadsafe(
                                sentence_queue.put_nowait,
                                remaining
                            )

                full_response = self._clean_text(full_response)

                return {"response": full_response, "tool_call": tool_call_detected}

            except Exception as e:
                print(f"[CHAT] Error generating response: {e}")
                return {"response": "I'm sorry, I encountered an error.", "tool_call": None}

    def _generate_streaming_after_tool(
        self,
        history: list,
        tool_result: str,
        sentence_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop
    ) -> dict:
        if self._cancelled.is_set():
            return {"response": "", "tool_call": None}

        with self._lock:
            if self._model is None:
                return {"response": "I'm sorry, but I'm not able to respond right now.", "tool_call": None}

            try:
                prompt = self._build_prompt(None, history, None, tool_result)

                sentence_buffer = SentenceBuffer()
                full_response = ""
                tool_call_detected = None

                for chunk in self._model(
                    prompt,
                    max_tokens=self._config.max_tokens,
                    temperature=self._config.temperature,
                    top_p=self._config.top_p,
                    stop=["User:", "Human:", "<|im_end|>", "<|endoftext|>"],
                    echo=False,
                    stream=True
                ):
                    if self._cancelled.is_set():
                        break

                    token = chunk.get("choices", [{}])[0].get("text", "")
                    if not token:
                        continue

                    full_response += token

                    if "<tool_call>" in full_response and tool_call_detected is None:
                        if self._tool_executor:
                            tool_call_detected = self._tool_executor.detect_tool_call(full_response)
                            if tool_call_detected:
                                continue

                    if tool_call_detected is None:
                        sentence = sentence_buffer.add(token)
                        if sentence:
                            sentence = self._clean_text(sentence)
                            if sentence:
                                loop.call_soon_threadsafe(
                                    sentence_queue.put_nowait,
                                    sentence
                                )

                if tool_call_detected is None:
                    remaining = sentence_buffer.flush()
                    if remaining:
                        remaining = self._clean_text(remaining)
                        if remaining:
                            loop.call_soon_threadsafe(
                                sentence_queue.put_nowait,
                                remaining
                            )

                full_response = self._clean_text(full_response)

                return {"response": full_response, "tool_call": tool_call_detected}

            except Exception as e:
                print(f"[CHAT] Error generating response: {e}")
                return {"response": "I'm sorry, I encountered an error.", "tool_call": None}

    def _generate_response_sync(
        self,
        user_message: Optional[str],
        history: list,
        tools_prompt: Optional[str],
        tool_result: Optional[str]
    ) -> str:
        if self._cancelled.is_set():
            return ""

        with self._lock:
            if self._model is None:
                return "I'm sorry, but I'm not able to respond right now."

            try:
                prompt = self._build_prompt(user_message, history, tools_prompt, tool_result)

                output = self._model(
                    prompt,
                    max_tokens=self._config.max_tokens,
                    temperature=self._config.temperature,
                    top_p=self._config.top_p,
                    stop=["User:", "Human:", "<|im_end|>", "<|endoftext|>"],
                    echo=False
                )

                response = self._extract_response(output)

                if not response:
                    return "I'm not sure how to respond to that."

                return response

            except Exception as e:
                print(f"[CHAT] Error generating response: {e}")
                return "I'm sorry, I encountered an error. Could you try again?"

    async def _stream_final_response(
        self,
        response: str,
        context: DaemonContext
    ) -> None:
        self._sentence_queue = asyncio.Queue()
        context.set_resource(ResourceName.SENTENCE_QUEUE, self._sentence_queue)
        context.set_resource(ResourceName.STREAMING_ACTIVE, True)

        sentences = self._split_into_sentences(response)
        for sentence in sentences:
            if self._cancelled.is_set():
                break
            await self._sentence_queue.put(sentence)

        await self._sentence_queue.put(None)

        user_message = context.get_resource(ResourceName.USER_MESSAGE)
        context.set_resource(ResourceName.CHAT_RESPONSE, {
            "text": response,
            "user_message": user_message
        })

    def _split_into_sentences(self, text: str) -> list[str]:
        sentence_endings = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _strip_tool_tags(self, text: str) -> str:
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
        text = re.sub(r'<tool_result>.*?</tool_result>', '', text, flags=re.DOTALL)
        return text.strip()

    def _trim_history(self, history: list) -> list:
        if len(history) > self._config.max_history * 2:
            history = history[-(self._config.max_history * 2):]

        total_chars = sum(len(msg["content"]) for msg in history)
        estimated_tokens = total_chars // 4

        while estimated_tokens > self._config.max_history_tokens and len(history) > 2:
            history.pop(0)
            if history:
                history.pop(0)
            estimated_tokens = sum(len(msg["content"]) for msg in history) // 4

        return history

    def _is_reset_command(self, message: str) -> bool:
        message_lower = message.lower().strip()
        for phrase in RESET_PHRASES:
            if phrase in message_lower:
                return True
        return False

    def _is_exit_command(self, message: str) -> bool:
        message_lower = message.lower().strip()
        for phrase in EXIT_PHRASES:
            if phrase in message_lower:
                return True
        return False

    def _build_prompt(
        self,
        user_message: Optional[str],
        history: list,
        tools_prompt: Optional[str],
        tool_result: Optional[str] = None
    ) -> str:
        system_prompt = BASE_SYSTEM_PROMPT
        if tools_prompt:
            system_prompt += TOOLS_PROMPT_TEMPLATE.format(tools_description=tools_prompt)

        prompt_parts = [f"<|im_start|>system\n{system_prompt}<|im_end|>"]

        for msg in history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            else:
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

        if tool_result:
            prompt_parts.append(f"<|im_start|>user\n{tool_result}<|im_end|>")

        if user_message:
            prompt_parts.append(f"<|im_start|>user\n{user_message}<|im_end|>")

        prompt_parts.append("<|im_start|>assistant\n")

        return "\n".join(prompt_parts)

    def _extract_response(self, output: dict) -> str:
        if not output or "choices" not in output:
            return ""

        choices = output.get("choices", [])
        if not choices:
            return ""

        text = choices[0].get("text", "")
        return self._clean_text(text)

    def _clean_text(self, text: str) -> str:
        text = text.strip()

        for stop in ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]:
            if stop in text:
                text = text.split(stop)[0]

        text = self._strip_markdown(text)
        text = text.strip()
        return text

    def _strip_markdown(self, text: str) -> str:
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'\1', text)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', text)
        text = re.sub(r'^[\s]*[-*]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[-*_]{3,}$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'  +', ' ', text)
        return text

    async def _on_cancel(
        self,
        result: TransitionResult,
        context: DaemonContext
    ) -> None:
        self._cancel_current()
        context.set_resource(ResourceName.STREAMING_ACTIVE, False)
        context.clear_resource(ResourceName.ACTIVE_TOOL)
        context.clear_resource(ResourceName.TOOL_RESULT)
        context.clear_resource(ResourceName.TOOL_FEEDBACK_TEXT)

    async def _on_enter_idle(
        self,
        result: TransitionResult,
        context: DaemonContext
    ) -> None:
        context.clear_resource(ResourceName.CHAT_RESPONSE)
        context.clear_resource(ResourceName.SENTENCE_QUEUE)
        context.clear_resource(ResourceName.STREAMING_ACTIVE)
        context.clear_resource(ResourceName.USER_MESSAGE)
        context.clear_resource(ResourceName.CHAT_INPUT)
        context.clear_resource(ResourceName.ACTIVE_TOOL)
        context.clear_resource(ResourceName.TOOL_RESULT)
        context.clear_resource(ResourceName.TOOL_FEEDBACK_TEXT)
