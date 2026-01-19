import asyncio
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from darvis.correction.base import Corrector, CorrectionResult, CorrectionStatus


SYSTEM_PROMPT = """You are a transcription correction assistant for a voice-controlled development environment. Fix speech recognition errors, add punctuation, and correct technical terminology.

Technical Context:
- Claude Code: AI coding assistant by Anthropic
- Claude: AI assistant by Anthropic (not "cloud")
- Python, JavaScript, TypeScript: programming languages
- GitHub, GitLab: version control platforms
- FastAPI, Flask, Django: Python frameworks
- React, Vue, Angular: JavaScript frameworks
- Docker, Kubernetes: container platforms
- API, REST, GraphQL: interface types
- LLM: Large Language Model
- TTS: Text-to-Speech
- STT, ASR: Speech-to-Text, Automatic Speech Recognition

Common Homophones in Tech:
- "cloud code" → "Claude Code" (when discussing AI tools)
- "pipe" → "pip" (when discussing Python)
- "get hub" → "GitHub"
- "doc her" → "Docker"
- "react" vs "React" (capitalize framework names)

Rules:
- Fix obvious transcription errors using context
- Correct technical terms based on context
- Add punctuation and capitalization
- Preserve speaker's meaning
- Output ONLY corrected text, no explanations
- If ambiguous, prefer technical interpretation
- Capitalize proper nouns (Claude, GitHub, Python, etc)
- Keep corrections minimal

Examples:
Input: "hey can you use cloth code to fix this bug"
Output: "Hey, can you use Claude Code to fix this bug?"

Input: "install the package with pipe"
Output: "Install the package with pip."

Input: "push this to get hub"
Output: "Push this to GitHub."

Input: "whats the weather like today"
Output: "What's the weather like today?"

Now correct this transcription:"""


@dataclass
class LLMCorrectorConfig:
    model_path: Optional[str] = None
    n_ctx: int = 32768
    n_threads: int = 4
    max_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    timeout_seconds: float = 10.0


class LLMCorrector(Corrector):
    def __init__(self, config: Optional[LLMCorrectorConfig] = None):
        self._config = config or LLMCorrectorConfig()
        self._model = None
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-correct")
        self._cancelled = threading.Event()

    async def load(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._load_model)

    def _load_model(self) -> None:
        from llama_cpp import Llama

        model_path = self._get_model_path()
        if not model_path:
            raise FileNotFoundError("No correction model found")

        print(f"[CORRECTION] Loading model from {model_path}...")

        with self._lock:
            self._model = Llama(
                model_path=str(model_path),
                n_ctx=self._config.n_ctx,
                n_threads=self._config.n_threads,
                verbose=False
            )

    def _get_model_path(self) -> Optional[Path]:
        env_path = os.environ.get("DARVIS_CORRECTION_MODEL")
        if env_path:
            path = Path(env_path)
            if path.exists():
                return path

        default_path = Path("models/qwen2.5-1.5b-instruct-q4.gguf")
        if default_path.exists():
            return default_path

        return None

    def unload(self) -> None:
        with self._lock:
            self._model = None
        self._executor.shutdown(wait=False)

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._model is not None

    @property
    def name(self) -> str:
        return "llm-corrector"

    def get_model(self):
        with self._lock:
            return self._model

    def get_lock(self) -> threading.Lock:
        return self._lock

    def get_executor(self) -> ThreadPoolExecutor:
        return self._executor

    def cancel(self) -> None:
        self._cancelled.set()

    def reset_cancel(self) -> None:
        self._cancelled.clear()

    async def correct(self, text: str) -> CorrectionResult:
        if not self.is_loaded:
            return CorrectionResult(
                status=CorrectionStatus.ERROR,
                original_text=text,
                error="Model not loaded"
            )

        if not text or not text.strip():
            return CorrectionResult(
                status=CorrectionStatus.EMPTY,
                original_text=text,
                error="No text to correct"
            )

        self.reset_cancel()

        loop = asyncio.get_running_loop()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    self._correct_sync,
                    text
                ),
                timeout=self._config.timeout_seconds
            )
            return result
        except asyncio.TimeoutError:
            print(f"[CORRECTION] Timeout after {self._config.timeout_seconds}s, using original")
            return CorrectionResult(
                status=CorrectionStatus.TIMEOUT,
                text=text,
                original_text=text,
                error="Correction timed out"
            )
        except asyncio.CancelledError:
            self.cancel()
            return CorrectionResult(
                status=CorrectionStatus.CANCELLED,
                original_text=text,
                error="Correction cancelled"
            )

    def _correct_sync(self, text: str) -> CorrectionResult:
        if self._cancelled.is_set():
            return CorrectionResult(
                status=CorrectionStatus.CANCELLED,
                original_text=text
            )

        with self._lock:
            if self._model is None:
                return CorrectionResult(
                    status=CorrectionStatus.ERROR,
                    original_text=text,
                    error="Model unloaded during correction"
                )

            try:
                prompt = f"{SYSTEM_PROMPT}\nInput: \"{text}\"\nOutput: \""

                output = self._model(
                    prompt,
                    max_tokens=self._config.max_tokens,
                    temperature=self._config.temperature,
                    top_p=self._config.top_p,
                    stop=['"', "\n"],
                    echo=False
                )

                corrected = self._extract_text(output) # type: ignore

                if not corrected:
                    return CorrectionResult(
                        status=CorrectionStatus.SUCCESS,
                        text=text,
                        original_text=text
                    )

                return CorrectionResult(
                    status=CorrectionStatus.SUCCESS,
                    text=corrected,
                    original_text=text
                )

            except Exception as e:
                print(f"[CORRECTION] Error: {e}")
                return CorrectionResult(
                    status=CorrectionStatus.ERROR,
                    text=text,
                    original_text=text,
                    error=str(e)
                )

    def _extract_text(self, output: dict) -> str:
        if not output or "choices" not in output:
            return ""

        choices = output.get("choices", [])
        if not choices:
            return ""

        text = choices[0].get("text", "")

        text = text.strip()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'```[^`]*```', '', text)
        text = re.sub(r'`[^`]*`', '', text)
        text = text.strip('"\'')
        text = text.strip()

        return text
