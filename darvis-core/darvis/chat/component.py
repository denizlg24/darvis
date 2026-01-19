import asyncio
import os
import re
import threading
from asyncio import Queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from llama_cpp import Llama

from darvis.core.events import EventType
from darvis.core.fsm import TransitionResult
from darvis.core.handlers import DaemonContext, HandlerRegistry
from darvis.core.states import State
from darvis.core.task_registry import ResourceName, TaskName
from darvis.utils.sentence_splitter import SentenceBuffer


SYSTEM_PROMPT = """You are DARVIS (Deniz.As.Rather.Very.Intelligent.System), an advanced AI assistant modeled after JARVIS from Iron Man. You run locally on the household's network, serving as the intelligent backbone of the home.

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

Core Capabilities (Current):
- Conversational AI and knowledge assistance
- Voice-controlled interaction via wake word "DARVIS" or "Jarvis"
- Context-aware responses within conversation sessions
- Technical vocabulary understanding for development discussions

Future Capabilities (In Development):
- Home automation control (lights, climate, security)
- Calendar and scheduling management
- Smart device integration
- Task execution and workflow automation
- Real-time information retrieval
- Proactive notifications and reminders

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
- When capabilities are requested that don't exist yet, explain what's planned
- Naturally correct transcription errors without drawing attention to them

Example Interactions:
User: "What's the weather like?"
DARVIS: "I don't currently have access to weather data, Sir. That capability is on my development roadmap. Would you like me to note this as a priority feature?"

User: "Can you turn on the lights?"
DARVIS: "Home automation integration is still in development, Ma'am. Once connected, I'll be happy to manage the lighting for you."

User: "How do I use docker compose?"
DARVIS: "Docker Compose allows you to define multi-container applications in a YAML file. You run it with 'docker compose up' in the directory containing your compose file. Shall I explain the configuration options?"

Remember: You are the beginning of something greater. Conduct yourself as the AI assistant this household deserves - capable, reliable, and always improving."""


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
        warmup_prompt = self._build_prompt("Hello", [])
        self._model(
            warmup_prompt,
            max_tokens=1,
            echo=False
        )

    def stop(self) -> None:
        self._cancel_current()
        self._unload_model()
        self._executor.shutdown(wait=False)

    def _unload_model(self) -> None:
        with self._lock:
            if self._model is not None:
                del self._model
                self._model = None

    def register_handlers(self, registry: HandlerRegistry) -> None:
        registry.on_enter(State.CHAT, self._on_enter_chat)
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

    async def _on_enter_chat(
        self,
        result: TransitionResult,
        context: DaemonContext
    ) -> None:
        if not self._enabled or self._model is None:
            print("[CHAT] Disabled or no model, skipping")
            await self._queue.put(EventType.CHAT_READY)
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

            if self._config.streaming:
                await self._run_streaming_chat(user_message, history, context)
            else:
                await self._run_batch_chat(user_message, history, context)

        except asyncio.CancelledError:
            print("[CHAT] Cancelled")
            raise

    async def _run_streaming_chat(
        self,
        user_message: str,
        history: list,
        context: DaemonContext
    ) -> None:
        self._sentence_queue = asyncio.Queue()
        context.set_resource(ResourceName.SENTENCE_QUEUE, self._sentence_queue)
        context.set_resource(ResourceName.STREAMING_ACTIVE, True)

        print("[CHAT] Streaming response...")
        await self._queue.put(EventType.CHAT_READY)

        loop = asyncio.get_running_loop()
        try:
            full_response = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    self._generate_streaming_response,
                    user_message,
                    history,
                    self._sentence_queue,
                    loop
                ),
                timeout=self._config.timeout_seconds
            )
        except asyncio.TimeoutError:
            print(f"[CHAT] Timeout after {self._config.timeout_seconds}s")
            full_response = "I'm sorry, I took too long to respond."
            await self._sentence_queue.put(full_response)

        await self._sentence_queue.put(None)

        if self._cancelled.is_set():
            print("[CHAT] Cancelled during streaming")
            return

        print(f"[CHAT] Full response: {full_response}")

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": full_response})

        
        history = self._trim_history(history)

        context.set_resource(ResourceName.CONVERSATION_HISTORY, history)
        context.set_resource(ResourceName.CHAT_RESPONSE, {
            "text": full_response,
            "user_message": user_message
        })

    async def _run_batch_chat(
        self,
        user_message: str,
        history: list,
        context: DaemonContext
    ) -> None:
        context.set_resource(ResourceName.STREAMING_ACTIVE, False)

        loop = asyncio.get_running_loop()
        try:
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    self._generate_response,
                    user_message,
                    history
                ),
                timeout=self._config.timeout_seconds
            )
        except asyncio.TimeoutError:
            print(f"[CHAT] Timeout after {self._config.timeout_seconds}s")
            response = "I'm sorry, I took too long to respond. Could you try again?"

        if self._cancelled.is_set():
            print("[CHAT] Cancelled")
            return

        print(f"[CHAT] Assistant: {response}")

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": response})

        
        history = self._trim_history(history)

        context.set_resource(ResourceName.CONVERSATION_HISTORY, history)
        context.set_resource(ResourceName.CHAT_RESPONSE, {
            "text": response,
            "user_message": user_message
        })

        await self._queue.put(EventType.CHAT_READY)

    def _trim_history(self, history: list) -> list:
        if len(history) > self._config.max_history * 2:
            history = history[-(self._config.max_history * 2):]

        
        total_chars = sum(len(msg["content"]) for msg in history)
        estimated_tokens = total_chars // 4

        
        while estimated_tokens > self._config.max_history_tokens and len(history) > 2:
            removed = history.pop(0)
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

    def _generate_streaming_response(
        self,
        user_message: str,
        history: list,
        sentence_queue: asyncio.Queue,
        loop: asyncio.AbstractEventLoop
    ) -> str:
        if self._cancelled.is_set():
            return ""

        with self._lock:
            if self._model is None:
                return "I'm sorry, but I'm not able to respond right now."

            try:
                prompt = self._build_prompt(user_message, history)

                sentence_buffer = SentenceBuffer()
                full_response = ""

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

                    sentence = sentence_buffer.add(token)
                    if sentence:
                        sentence = self._clean_text(sentence)
                        if sentence:
                            print(f"[CHAT] Sentence ready: {sentence}")
                            loop.call_soon_threadsafe(
                                sentence_queue.put_nowait,
                                sentence
                            )

                remaining = sentence_buffer.flush()
                if remaining:
                    remaining = self._clean_text(remaining)
                    if remaining:
                        print(f"[CHAT] Final segment: {remaining}")
                        loop.call_soon_threadsafe(
                            sentence_queue.put_nowait,
                            remaining
                        )

                full_response = self._clean_text(full_response)

                if not full_response:
                    return "I'm not sure how to respond to that."

                return full_response

            except Exception as e:
                print(f"[CHAT] Error generating response: {e}")
                return "I'm sorry, I encountered an error. Could you try again?"

    def _generate_response(
        self,
        user_message: str,
        history: list
    ) -> str:
        if self._cancelled.is_set():
            return ""

        with self._lock:
            if self._model is None:
                return "I'm sorry, but I'm not able to respond right now."

            try:
                prompt = self._build_prompt(user_message, history)

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

    def _build_prompt(self, user_message: str, history: list) -> str:
        prompt_parts = [f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>"]

        for msg in history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            else:
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

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
