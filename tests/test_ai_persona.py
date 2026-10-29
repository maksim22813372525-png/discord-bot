from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai_persona import direct_reply, ALIASES, SYSTEM_PROMPT


class PersonaTests(unittest.TestCase):
    def test_user_greeting_regression(self):
        for text in ('привет', 'Приветики!', 'пр', 'привееет'):
            answer = direct_reply(text)
            self.assertIsNotNone(answer, text)
            self.assertLess(len(answer), 30)
            self.assertNotIn('сленг', answer.casefold())
            self.assertNotIn('сообщен', answer.casefold())

    def test_all_requested_aliases_get_a_short_response(self):
        for alias in ('Няшка', 'Альтушка', 'Альтушечка', 'Тянка'):
            for text in (alias, alias.upper() + '!', 'Эй, ' + alias,
                         alias + ', ты тут?', alias + ', привет!'):
                with self.subTest(text=text):
                    answer = direct_reply(text)
                    self.assertIsNotNone(answer)
                    self.assertLess(len(answer), 30)
            self.assertIn(alias.casefold(), ALIASES)

    def test_real_questions_and_unknown_words_still_go_to_the_llm(self):
        for text in ('Привет, как дела?', 'Альтушка, я забрал раунд в кс',
                     'Няшка, ты тут была вчера?', 'Тянка, что думаешь про игру?',
                     'прекрасно', 'алгебра', '', '...'):
            self.assertIsNone(direct_reply(text), text)

    def test_ukrainian_and_english_are_understood_without_stripping_ukrainian_letters(self):
        for text in ('Привіт!', 'Альтушечка, привітик', 'Hi!', 'Hello', 'Няшка, ти тут?',
                     'Hey, Luna!', 'Altushka', 'Альтушечко'):
            self.assertIsNotNone(direct_reply(text), text)
        for text in ('How are you?', 'Альтушка, як справи?', 'Луна, I won раунд', 'привітання'):
            self.assertIsNone(direct_reply(text), text)
        self.assertIn('украинский, английский', SYSTEM_PROMPT)

    def test_persona_requires_only_a_direct_spoken_reply_without_fake_dialogue(self):
        self.assertIn('Текстовый чат', SYSTEM_PROMPT)
        self.assertIn('от своего лица', SYSTEM_PROMPT)
        self.assertIn('единственным полем reply', SYSTEM_PROMPT)
        self.assertNotIn('Пример реплики:', SYSTEM_PROMPT)
        self.assertNotIn('Пример ответа:', SYSTEM_PROMPT)
        self.assertIn('ответь честно', SYSTEM_PROMPT)


if __name__ == '__main__':
    unittest.main()
