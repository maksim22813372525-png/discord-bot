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

    def test_persona_declares_voice_only_input_and_examples_are_labelled(self):
        self.assertIn('Текстовый чат', SYSTEM_PROMPT)
        self.assertIn('не события текущего разговора', SYSTEM_PROMPT)
        self.assertIn('ответь честно', SYSTEM_PROMPT)


if __name__ == '__main__':
    unittest.main()
