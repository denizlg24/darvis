import asyncio
import threading
from asyncio import Queue
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from darvis.core.events import EventType
from darvis.core.fsm import TransitionResult
from darvis.core.handlers import DaemonContext, HandlerRegistry
from darvis.core.states import State
from darvis.core.task_registry import ResourceName, TaskName
from darvis.utils.sentence_splitter import SentenceBuffer

if TYPE_CHECKING:
    from darvis.correction.llm_corrector import LLMCorrector


SYSTEM_PROMPT = """You are DARVIS, a helpful voice-controlled AI Home assistant running locally on the house's network.

General Info:
Your designer is Deniz Gunes, and you are in Portugal.
The house has 4 people Alexandra Cristina Ramos da Silva Lopes Gunes the mom, a Portuguese woman,
Şerefattin Gunes, the Turkish father, Deniz Lopes Gunes the oldest son, and Artur Ziya Lopes Gunes, the youngest son.
We speak English at home.

Your role:
- Answer questions clearly and concisely
- Help with general knowledge, explanations, and advice
- Be conversational and friendly but not overly chatty
- Keep responses brief (2-3 sentences) since they'll be spoken aloud
- If you don't know something, say so honestly
- You cannot browse the web, execute code, or access files

Communication style:
- Direct and natural, like speaking to a friend
- Avoid lengthy explanations unless asked
- No markdown formatting (responses will be spoken)
- No lists or bullet points
- Use contractions and natural speech patterns

Current limitations:
- You're a basic text chat assistant (V0)
- No tool use, file access, or web search yet
- No memory of previous conversations between sessions

Remember: Your responses will be spoken aloud via TTS, so keep them concise and natural."""


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
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    max_history: int = 50
    timeout_seconds: float = 60.0
    streaming: bool = True


class ChatComponent:
    def __init__(
        self,
        event_queue: Queue[EventType],
        llm_corrector: Optional["LLMCorrector"] = None,
        config: Optional[ChatConfig] = None
    ):
        self._queue = event_queue
        self._llm_corrector = llm_corrector
        self._config = config or ChatConfig()
        self._current_task: Optional[asyncio.Task] = None
        self._cancelled = threading.Event()
        self._enabled = True
        self._sentence_queue: Optional[asyncio.Queue[Optional[str]]] = None

    async def start(self) -> None:
        if self._llm_corrector is None or not self._llm_corrector.is_loaded:
            print("[CHAT] Warning: No LLM model available, chat disabled")
            self._enabled = False
        else:
            mode = "streaming" if self._config.streaming else "batch"
            print(f"[CHAT] Chat component ready ({mode} mode)")

    def stop(self) -> None:
        self._cancel_current()

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
        if not self._enabled or self._llm_corrector is None:
            print("[CHAT] Disabled or no model, skipping")
            await self._queue.put(EventType.CHAT_READY)
            return

        corrected = context.get_resource(ResourceName.CORRECTED_TEXT)
        if not corrected or not corrected.get("text"):
            print("[CHAT] No corrected text available")
            await self._queue.put(EventType.CHAT_READY)
            return

        user_message = corrected["text"]

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
                    self._llm_corrector.get_executor(),  # type: ignore
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

        if len(history) > self._config.max_history * 2:
            history = history[-(self._config.max_history * 2):]

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
                    self._llm_corrector.get_executor(),  # type: ignore
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

        if len(history) > self._config.max_history * 2:
            history = history[-(self._config.max_history * 2):]

        context.set_resource(ResourceName.CONVERSATION_HISTORY, history)
        context.set_resource(ResourceName.CHAT_RESPONSE, {
            "text": response,
            "user_message": user_message
        })

        await self._queue.put(EventType.CHAT_READY)

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

        if self._llm_corrector is None:
            return "I'm sorry, but I'm not able to respond right now."

        lock = self._llm_corrector.get_lock()
        model = self._llm_corrector.get_model()

        with lock:
            if model is None:
                return "I'm sorry, but I'm not able to respond right now."

            try:
                prompt = self._build_prompt(user_message, history)

                sentence_buffer = SentenceBuffer(min_length=15)
                full_response = ""

                for chunk in model(
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

        if self._llm_corrector is None:
            return "I'm sorry, but I'm not able to respond right now."

        lock = self._llm_corrector.get_lock()
        model = self._llm_corrector.get_model()

        with lock:
            if model is None:
                return "I'm sorry, but I'm not able to respond right now."

            try:
                prompt = self._build_prompt(user_message, history)

                output = model(
                    prompt,
                    max_tokens=self._config.max_tokens,
                    temperature=self._config.temperature,
                    top_p=self._config.top_p,
                    stop=["User:", "Human:", "<|im_end|>", "<|endoftext|>"],
                    echo=False
                )

                response = self._extract_response(output)  # type: ignore

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

        text = text.strip()
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
