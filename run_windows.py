"""Ask for a token locally, without writing it to disk or echoing it."""

import getpass
import os
from pathlib import Path
import runpy


def main():
    os.chdir(Path(__file__).resolve().parent)
    if not os.environ.get('DISCORD_TOKEN'):
        print('Встав токен бота і натисни Enter. Символи не показуватимуться — це нормально.')
        token = getpass.getpass('Discord token: ').strip()
        if not token:
            raise SystemExit('Токен не введено. Запусти start.bat ще раз.')
        os.environ['DISCORD_TOKEN'] = token
    print('Підключення до Discord… Для зупинки натисни Ctrl+C або закрий це вікно.')
    try:
        runpy.run_path('bot.py', run_name='__main__')
    finally:
        os.environ.pop('DISCORD_TOKEN', None)


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print('\nЗапуск завершено.')
