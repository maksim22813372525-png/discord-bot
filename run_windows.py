"""Install checksum-verified portable Node.js and launch the voice bot."""

import getpass
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import urllib.request
import zipfile

NODE_VERSION = '24.22.2'
NODE_HASHES = {
    'x64': 'ca1bff5d8ead13c9f2ea404a176db46bd430704d5b5113734049fa365e7799e2',
    'arm64': 'f47968c213105077f64c6510f007632c0ca2574802a0b5067c327c49abe1c468',
}


def prepare_node(root):
    if platform.system() != 'Windows':
        raise RuntimeError('На Linux/macOS використовуй npm ci та npm start.')
    machine = platform.machine().lower()
    if machine in ('amd64', 'x86_64'):
        arch = 'x64'
    elif machine in ('arm64', 'aarch64'):
        arch = 'arm64'
    else:
        raise RuntimeError('Потрібна 64-бітна Windows (x64 або ARM64).')
    runtime = root / '.runtime'
    folder = runtime / f'node-v{NODE_VERSION}-win-{arch}'
    node = folder / 'node.exe'
    npm = folder / 'node_modules/npm/bin/npm-cli.js'
    verified = folder / '.verified-runtime'
    if not node.is_file() or not npm.is_file() or not verified.is_file() or verified.read_text() != NODE_HASHES[arch]:
        runtime.mkdir(exist_ok=True)
        archive = runtime / 'node-download.zip'
        filename = f'node-v{NODE_VERSION}-win-{arch}.zip'
        print('Завантажую Node.js з nodejs.org. Перший запуск може тривати кілька хвилин…', flush=True)
        try:
            digest = hashlib.sha256()
            with urllib.request.urlopen(f'https://nodejs.org/dist/v{NODE_VERSION}/{filename}', timeout=60) as response:
                with archive.open('wb') as output:
                    while chunk := response.read(1024 * 1024):
                        digest.update(chunk)
                        output.write(chunk)
            if digest.hexdigest() != NODE_HASHES[arch]:
                raise RuntimeError('Перевірка SHA-256 Node.js не пройдена. Файл не запускатиметься.')
            with zipfile.ZipFile(archive) as zipped:
                for entry in zipped.infolist():
                    if not (runtime / entry.filename).resolve().is_relative_to(folder.resolve()):
                        raise RuntimeError('Небезпечний шлях у ZIP Node.js.')
                zipped.extractall(runtime)
            verified.write_text(NODE_HASHES[arch])
        finally:
            archive.unlink(missing_ok=True)
    subprocess.run([str(node), '--version'], check=True)
    return node, npm


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    node, npm = prepare_node(root)
    os.environ['PATH'] = str(node.parent) + os.pathsep + os.environ.get('PATH', '')
    fingerprint = hashlib.sha256((root / 'package-lock.json').read_bytes()).hexdigest()
    marker = root / 'node_modules/.voice-bot-lock'
    if not marker.is_file() or marker.read_text() != fingerprint:
        print('Встановлюю голосові бібліотеки…', flush=True)
        install_env = dict(os.environ)
        install_env.pop('DISCORD_TOKEN', None)
        subprocess.run([str(node), str(npm), 'ci', '--omit=dev', '--no-audit', '--no-fund'], check=True, env=install_env)
        marker.write_text(fingerprint)
    if not os.environ.get('DISCORD_TOKEN'):
        print('Встав токен бота і натисни Enter. Символи не показуватимуться — це нормально.')
        token = getpass.getpass('Discord token: ').strip()
        if not token:
            raise SystemExit('Токен не введено. Запусти start.bat ще раз.')
        os.environ['DISCORD_TOKEN'] = token
    print('Підключення до Discord… Для зупинки натисни Ctrl+C або закрий це вікно.')
    try:
        result = subprocess.run([str(node), 'bot.js'], check=False)
        if result.returncode:
            raise SystemExit(result.returncode)
    finally:
        os.environ.pop('DISCORD_TOKEN', None)


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print('\nЗапуск завершено.')
    except Exception as error:
        raise SystemExit(f'Не вдалося запустити: {error}') from None
