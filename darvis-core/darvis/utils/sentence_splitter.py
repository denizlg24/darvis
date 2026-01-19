import re
from typing import Optional, Tuple

ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
    "etc", "vs", "i.e", "e.g", "inc", "ltd", "co",
    "st", "ave", "blvd", "rd", "apt", "no",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "u.s", "u.k", "a.m", "p.m",
}

SENTENCE_ENDINGS = {'.', '!', '?'}

MIN_SENTENCE_LENGTH = 80


class SentenceBuffer:
    def __init__(self, min_length: int = MIN_SENTENCE_LENGTH):
        self._buffer = ""
        self._min_length = min_length

    def add(self, token: str) -> Optional[str]:
        self._buffer += token
        return self._try_extract_sentence()

    def flush(self) -> Optional[str]:
        text = self._buffer.strip()
        self._buffer = ""
        return text if text else None

    def clear(self) -> None:
        self._buffer = ""

    def _try_extract_sentence(self) -> Optional[str]:
        text = self._buffer

        for i, char in enumerate(text):
            if char not in SENTENCE_ENDINGS:
                continue

            potential_end = i + 1

            if potential_end >= len(text):
                continue

            next_char = text[potential_end] if potential_end < len(text) else ""

            if next_char not in (' ', '\n', '\r', '"', "'", ')'):
                continue

            if self._is_abbreviation(text, i):
                continue

            sentence = text[:potential_end].strip()

            if len(sentence) < self._min_length:
                continue

            if potential_end < len(text):
                after = text[potential_end:].lstrip()
                if after and after[0].islower():
                    continue

            self._buffer = text[potential_end:].lstrip()
            return sentence

        return None

    def _is_abbreviation(self, text: str, period_pos: int) -> bool:
        if text[period_pos] != '.':
            return False

        before = text[:period_pos].split()
        if not before:
            return False

        last_word = before[-1].lower().rstrip('.')

        if last_word in ABBREVIATIONS:
            return True

        if len(last_word) <= 2 and last_word.replace('.', '').isalpha():
            return True

        return False


def split_into_sentences(text: str, min_length: int = MIN_SENTENCE_LENGTH) -> list[str]:
    sentences = []
    buffer = SentenceBuffer(min_length)

    for char in text:
        sentence = buffer.add(char)
        if sentence:
            sentences.append(sentence)

    remaining = buffer.flush()
    if remaining:
        sentences.append(remaining)

    return sentences


def is_sentence_complete(text: str) -> bool:
    text = text.strip()

    if not text:
        return False

    if text[-1] not in SENTENCE_ENDINGS:
        return False

    if len(text) < MIN_SENTENCE_LENGTH:
        return False

    words = text.split()
    if not words:
        return False

    last_word = words[-1].lower().rstrip('.!?')
    if last_word in ABBREVIATIONS:
        return False

    return True
