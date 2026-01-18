from darvis.core.states import State
from darvis.core.events import EventType

TRANSITIONS = {
    (State.INIT, EventType.LOADED): State.IDLE,
    (State.CANCEL_REQUESTED, EventType.CANCEL_PROCESSED): State.IDLE,
    (State.IDLE, EventType.WAKE_WORD): State.LISTENING,
    (State.LISTENING, EventType.LISTEN_START): State.VOICE_CAPTURE,
    (State.LISTENING, EventType.SILENCE_DETECTED): State.IDLE,
    (State.VOICE_CAPTURE, EventType.SILENCE_DETECTED): State.TRANSCRIBING,
    (State.TRANSCRIBING, EventType.TRANSCRIPTION_READY): State.CORRECTING,
    (State.CORRECTING, EventType.CORRECTION_READY): State.CHAT,
    (State.CHAT, EventType.CHAT_READY): State.TTS,
    (State.TTS, EventType.TTS_DONE): State.IDLE,
    (State.TTS, EventType.WAKE_WORD): State.LISTENING,
}
