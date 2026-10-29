import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai_reply import CLARIFICATION, MAX_REPLY_CHARS, REPLY_SCHEMA, RejectedReply, parse_reply, validate_spoken_reply


class ReplyContractTests(unittest.TestCase):
    def test_contract_has_one_required_spoken_field(self):
        self.assertEqual(REPLY_SCHEMA['required'], ['reply'])
        self.assertEqual(set(REPLY_SCHEMA['properties']), {'reply'})
        self.assertFalse(REPLY_SCHEMA['additionalProperties'])
        self.assertEqual(REPLY_SCHEMA['properties']['reply']['maxLength'], MAX_REPLY_CHARS)
        self.assertEqual(validate_spoken_reply(CLARIFICATION), CLARIFICATION)

    def test_actual_narration_report_is_rejected_instead_of_salvaged(self):
        text = 'Хорошо, пользователь написал "जिके". Сначала нужно понять, что он имел в виду. Нужно ответить дружелюбно.'
        with self.assertRaises(RejectedReply) as caught:
            parse_reply(json.dumps({'reply': text}))
        self.assertEqual(caught.exception.code, 'reasoning')
        self.assertNotIn('जिके', str(caught.exception))

    def test_plain_reasoning_and_user_role_narration_are_rejected(self):
        for text in (
            'Пользователь сказал привет, поэтому отвечу приветливо.',
            'Так, пользователь спросил, как у меня дела.',
            'The user said hello, I should reply casually.',
            'Користувач сказав привіт. Треба відповісти.',
            'Мне нужно сформулировать короткий ответ.',
            'Сначала нужно проанализировать запрос.',
            'I should respond in a friendly way.',
            'Хорошо. Нужно ответить коротко.',
        ):
            with self.subTest(text=text), self.assertRaises(RejectedReply):
                validate_spoken_reply(text)

    def test_role_labels_and_fabricated_dialogue_do_not_reach_speech(self):
        for text in (
            'Луна: Приветики!',
            'User: Hi! Assistant: Hello!',
            'Привет! Пользователь: Как дела? Луна: Хорошо.',
            'Ответ: Ого, хорош!',
            'Мой ответ пользователю: Приветики!',
            'Analysis: choose a warm reply.',
            'Final answer: Привет!',
            '*улыбается* Приветики!',
            '```json {"reply":"Привет"}```',
            '/no_think Привет!',
            '/think Сначала нужно понять вопрос.',
        ):
            with self.subTest(text=text), self.assertRaises(RejectedReply):
                validate_spoken_reply(text)

    def test_closed_and_unterminated_reasoning_are_both_rejected(self):
        for text in ('<think>Пользователь сказал привет.</think> Приветики!',
                     '<think>Надо понять вопрос', 'Привет! </think>',
                     '<THINK >Нужен ответ', '<analysis>plan</analysis> Привет!'):
            with self.subTest(text=text), self.assertRaises(RejectedReply):
                parse_reply(json.dumps({'reply': text}))

    def test_legitimate_words_and_quoted_speech_are_preserved(self):
        for text in (
            'Приветики!', 'М? Я тут.', 'Ого, хорош. Только не зазнавайся, звезда.',
            'Я сказал тебе, что сегодня устал.',
            'Ты сказал, что устал. Давай выдохнем.',
            'Слово «пользователь» означает человека, который пользуется программой.',
            'Пользователь — человек, который пользуется программой.',
            '«Пользователь сказал привет» — это обычная цитата.',
            'Он сказал: «Я устал». Может, просто дать ему отдохнуть?',
            'Мне нужно понять, что ты имеешь в виду. Повтори коротко?',
            'Нужно ответить ему честно, если не хочешь играть.',
            'Сначала нужно проанализировать ситуацию, а потом уже спорить.',
            'Нужно сформулировать цель, иначе вы будете спорить вечно.',
            'Луна — спутник Земли. А я тут просто отвлеклась.',
            'Ты спросил "как дела?". Уже веселее.',
        ):
            with self.subTest(text=text):
                self.assertEqual(parse_reply(json.dumps({'reply': text})), text)

    def test_json_must_be_complete_and_have_exactly_one_field(self):
        for raw in ('Приветики!', '{"reply":"Привет', '```json\n{"reply":"Привет"}\n```',
                    '{"reply":"Привет"} trailing', '[]', 'null', '42',
                    '{"reply":"Привет","analysis":"plan"}',
                    '{"reply":"Привет","reply":"Пока"}', '{}', '[' * 1500 + ']' * 1500):
            with self.subTest(raw=raw), self.assertRaises(RejectedReply):
                parse_reply(raw)

    def test_open_quote_cannot_hide_narration_but_actual_nested_quotes_are_allowed(self):
        for text in ('"Пользователь сказал привет', '«User: hi! Assistant: hello!'):
            with self.subTest(text=text), self.assertRaises(RejectedReply):
                validate_spoken_reply(text)
        for text in ('Он сказал: «Слово “пользователь” тут ни при чём».',
                     'Он сказал: ‘I’m tired’. Может, дать ему отдохнуть?'):
            self.assertEqual(validate_spoken_reply(text), text)

    def test_spoken_field_must_be_nonempty_bounded_text_without_controls(self):
        for value in (None, False, 3, [], {}, '', '  ', 'а' * (MAX_REPLY_CHARS + 1),
                      'Привет\nКак дела?', 'Привет\t!', 'Привет\x00!',
                      'Привет\u202e!', 'Привет\ud800!'):
            with self.subTest(value=repr(value)), self.assertRaises(RejectedReply):
                parse_reply(json.dumps({'reply': value}))

    def test_only_outer_spaces_are_normalized_and_rejections_expose_fixed_codes(self):
        self.assertEqual(parse_reply('{"reply":"  Приветики!  "}'), 'Приветики!')
        with self.assertRaises(RejectedReply) as caught:
            parse_reply('{"reply": "Приватный текст", "secret": "do-not-log"}')
        self.assertEqual(str(caught.exception), 'invalid_shape')
        self.assertNotIn('do-not-log', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
