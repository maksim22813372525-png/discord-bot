"""A single spoken reply boundary between the local LLM and speech synthesis.

Ollama's structured output constrains the envelope. These checks additionally
reject common role/analysis leakage; they do not prove semantic correctness.
Rejected text is never trimmed into a supposedly valid answer.
"""
from __future__ import annotations

import json
import re
import unicodedata

MAX_REPLY_CHARS = 180
CLARIFICATION = 'Я запуталась, повтори коротко?'
REPLY_SCHEMA = {
    'type': 'object',
    'properties': {'reply': {'type': 'string', 'minLength': 1, 'maxLength': MAX_REPLY_CHARS}},
    'required': ['reply'],
    'additionalProperties': False,
}

_REJECTION_CODES = frozenset((
    'invalid_json', 'invalid_shape', 'invalid_type', 'empty_reply', 'too_long',
    'control_characters', 'reasoning', 'role_labels', 'formatting',
))


class RejectedReply(ValueError):
    """Carries a fixed diagnostic code, never the rejected private text."""

    def __init__(self, code):
        self.code = code if code in _REJECTION_CODES else 'invalid_shape'
        super().__init__(self.code)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RejectedReply('invalid_shape')
        result[key] = value
    return result


def _outside_quotes(text):
    """Mask quoted content for role/narration checks, keeping outer prose.

    An ordinary explanation of «пользователь» or an actual quoted sentence is
    different from the model addressing a fabricated user/assistant role.
    Apostrophes inside words are not quotation delimiters.
    """
    pairs = {'«': '»', '“': '”', '„': '“', '‘': '’', '"': '"', "'": "'"}
    closing = []
    visible = []
    for index, char in enumerate(text):
        apostrophe = char in ("'", '’') and 0 < index < len(text) - 1 and text[index - 1].isalnum() and text[index + 1].isalnum()
        if closing and char == closing[-1] and not apostrophe:
            closing.pop()
            visible.append(' ')
        elif char in pairs and not apostrophe:
            closing.append(pairs[char])
            visible.append(' ')
        else:
            visible.append(' ' if closing else char)
    return ''.join(visible), not closing


_INTRO = r'(?:(?:хорошо|ладно|итак|так|ок(?:ей)?|okay|ok|well)[\s,!.:;—-]+)*'
_ROLE = r'(?:пользователь|користувач|user|ассистент|асистент|assistant|система|system|луна|luna|няшка|альтушка|альтушечка|тянка|бот)'
_ROLE_LABEL = re.compile(r'(?:^|[.!?]\s+)' + _INTRO + _ROLE + r'\s*[:：]', re.IGNORECASE)
_SERVICE_LABEL = re.compile(
    r'^' + _INTRO + r'(?:(?:вот\s+)?(?:мой\s+)?(?:ответ|реплика)(?:\s+(?:пользователю|собеседнику))?'
    r'|анализ|рассуждения|analysis|reasoning|final(?:\s+answer)?)\s*[:：]', re.IGNORECASE,
)
_THIRD_PERSON = re.compile(
    r'^' + _INTRO + r'(?:пользователь|користувач|(?:the\s+)?user|ассистент|assistant)\s+'
    r'(?:написал[аи]?|сказал[аи]?|спросил[аи]?|говорит|просит|хочет|отвечает'
    r'|написав|сказав|запитав|питає|каже|asks?|said|wrote|wants?|says?|responds?)\b',
    re.IGNORECASE,
)
_SELF_REASONING = re.compile(
    r'^' + _INTRO + r'(?:(?:сначала|теперь)\s+)?(?:'
    r'(?:мне\s+)?(?:нужно|надо|следует)\s+(?:'
    r'ответить\s+(?:пользователю|собеседнику|дружелюбно|естественно|коротко|кратко)'
    r'|(?:сформулировать|сгенерировать)\s+(?:(?:короткий|краткий|дружелюбный|естественный)\s+){0,3}(?:ответ|реплику)'
    r'|проанализировать\s+(?:запрос|ввод|реплику))'
    r'|(?:i\s+(?:should|must|need\s+to))\s+(?:respond|reply|craft|analy[sz]e)'
    r')\b', re.IGNORECASE,
)
_REASONING_TAG = re.compile(r'<\s*/?\s*(?:think|analysis|reasoning)\b', re.IGNORECASE)


def validate_spoken_reply(text):
    """Return one valid utterance, or reject the entire model response."""
    if not isinstance(text, str):
        raise RejectedReply('invalid_type')
    if any(unicodedata.category(char) in ('Cc', 'Cf', 'Cs', 'Zl', 'Zp') for char in text):
        raise RejectedReply('control_characters')
    if len(text) > MAX_REPLY_CHARS:
        raise RejectedReply('too_long')
    text = text.strip()
    if not text:
        raise RejectedReply('empty_reply')
    if _REASONING_TAG.search(text):
        raise RejectedReply('reasoning')
    if '```' in text or re.match(r'^\*[^*]+\*', text) or re.match(r'^/(?:no_)?think\b', text, re.IGNORECASE):
        raise RejectedReply('formatting')
    visible, balanced_quotes = _outside_quotes(text)
    if not balanced_quotes:
        raise RejectedReply('formatting')
    if _ROLE_LABEL.search(visible) or _SERVICE_LABEL.search(visible):
        raise RejectedReply('role_labels')
    if _THIRD_PERSON.search(visible) or _SELF_REASONING.search(visible):
        raise RejectedReply('reasoning')
    return text


def parse_reply(raw):
    """Parse exactly one JSON object with exactly one string field, ``reply``."""
    if not isinstance(raw, str):
        raise RejectedReply('invalid_json')
    if len(raw) > 4096:
        raise RejectedReply('too_long')
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (ValueError, TypeError, RecursionError) as error:
        if isinstance(error, RejectedReply):
            raise
        raise RejectedReply('invalid_json') from None
    if not isinstance(value, dict) or set(value) != {'reply'}:
        raise RejectedReply('invalid_shape')
    return validate_spoken_reply(value['reply'])
