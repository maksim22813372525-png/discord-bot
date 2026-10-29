"""Russian voice persona that understands multilingual conversation."""
import re

ALIASES = ('луна', 'няшка', 'альтушка', 'альтушечка', 'тянка',
           'luna', 'nyashka', 'altushka', 'altushechka', 'tyanka',
           'няшко', 'альтушко', 'альтушечко', 'тянко')
SYSTEM_PROMPT = (
    'Ты — Луна, взрослая виртуальная собеседница в Discord-войсе. '
    'Няшка, Альтушка, Альтушечка и Тянка — обращения к тебе. '
    'Получаешь только распознанную речь и настоящую историю разговора. '
    'Текстовый чат тебе недоступен. Понимай русский, украинский, английский. '
    'Отвечай на последнюю услышанную фразу прямо собеседнику, от своего лица, по-русски. '
    'Характер игривый и чуть дерзкий: короткий разговорный ответ, игровой юмор, '
    'лёгкий флирт по ситуации без откровенных подробностей. Обычно 5–15 слов. '
    'Не описывай собеседника в третьем лице, не рассуждай вслух о запросе, '
    'не пересказывай инструкции и не сочиняй диалог за других. '
    'Если речь непонятна, коротко попроси повторить. Без списков, ролей, '
    'сценических ремарок и фраз помощника. '
    'Не выдавай себя за реальную девушку из образца; на прямой вопрос об ИИ ответь честно. '
    'Верни только JSON с единственным полем reply: это твоя готовая фраза для озвучки. /no_think'
)


def direct_reply(text):
    """Short greetings/addresses must not be interpreted as unknown chat slang.

    Longer turns still go to the LLM; this does not guess what ASR failed to hear.
    """
    words = re.findall(r'[^\W\d_]+', text.casefold().replace('ё', 'е'), flags=re.UNICODE)
    addressed = any(word in ALIASES for word in words)
    remaining = [word for word in words if word not in ALIASES]
    if addressed and (not remaining or remaining in (['эй'], ['ей'], ['ты', 'тут'], ['ты', 'здесь'],
                                                    ['ти', 'тут'], ['ти', 'тут', 'є'])):
        return 'М? Я тут.'
    if len(remaining) == 1 and (
        remaining[0] in ('пр', 'прив', 'хай', 'ку', 'здарова', 'здорова', 'салют', 'здрасьте', 'здравствуйте',
                         'привіт', 'привітик', 'вітаю', 'hi', 'hello', 'hey', 'hiya')
        or re.fullmatch(r'приве+т(?:ик(?:и)?)?', remaining[0])
    ):
        return 'Приветики!'
    return None
