"""Prepare the local AI stack, then supervise its worker and Discord client.

The reference stays on this machine. Installers do not inherit the Discord or
worker credential. Ollama and the Python worker are contacted only on loopback.
"""

import getpass
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings

from run_windows import prepare_node

OLLAMA_URL = 'http://127.0.0.1:11434'
OLLAMA_MODEL = 'qwen3:4b'
TORCH_VERSION = '2.9.1'
TORCH_INDEX = 'https://download.pytorch.org/whl/cu128'
INSTALL_STACK = f'python3.12;torch={TORCH_VERSION};torchaudio={TORCH_VERSION};{TORCH_INDEX}'
_LOCAL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def require_runtime():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            'Для AI-версії потрібен Python 3.12 (64-bit). Встанови його з '
            'https://www.python.org/downloads/release/python-31210/ '
            '(Windows installer 64-bit), потім відкрий start_ai.bat.'
        )
    if platform.system() != 'Windows' or platform.machine().lower() not in ('amd64', 'x86_64'):
        raise RuntimeError('Цей AI-запускач підтримує Windows x64.')


def dependency_env():
    env = dict(os.environ)
    for name in ('DISCORD_TOKEN', 'AI_WORKER_TOKEN', 'AI_WORKER_URL'):
        env.pop(name, None)
    env['PYTHONUTF8'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    env['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'
    return env


def prepare_node_packages(root, node, npm, env):
    fingerprint = hashlib.sha256((root / 'package-lock.json').read_bytes()).hexdigest()
    marker = root / 'node_modules/.voice-bot-lock'
    if marker.is_file() and marker.read_text() == fingerprint:
        return
    print('Встановлюю бібліотеки Discord…', flush=True)
    subprocess.run(
        [str(node), str(npm), 'ci', '--omit=dev', '--no-audit', '--no-fund'],
        cwd=root, env=env, check=True,
    )
    marker.parent.mkdir(exist_ok=True)
    marker.write_text(fingerprint)


def prepare_python_dependencies(root, env):
    folder = root / '.ai-venv'
    python = folder / 'Scripts/python.exe'
    if not python.is_file():
        print('Створюю окреме Python-середовище для AI…', flush=True)
        subprocess.run([sys.executable, '-m', 'venv', str(folder)], env=env, check=True)
    try:
        subprocess.run(
            [str(python), '-c', 'import sys; sys.exit(sys.version_info[:2] != (3, 12))'],
            env=env, check=True,
        )
    except subprocess.CalledProcessError:
        raise RuntimeError(
            'Папка .ai-venv створена для іншої версії Python. '
            'Розпакуй AI-бота в нову папку й відкрий start_ai.bat через Python 3.12.'
        ) from None
    requirements = root / 'ai-requirements.txt'
    fingerprint = hashlib.sha256(requirements.read_bytes() + INSTALL_STACK.encode()).hexdigest()
    marker = folder / '.voice-ai-installed'
    if not marker.is_file() or marker.read_text() != fingerprint:
        print('Встановлюю PyTorch CUDA та голосові моделі. Перший запуск завантажить кілька ГБ…', flush=True)
        subprocess.run(
            [str(python), '-m', 'pip', 'install', '--disable-pip-version-check',
             f'torch=={TORCH_VERSION}', f'torchaudio=={TORCH_VERSION}', '--index-url', TORCH_INDEX],
            cwd=root, env=env, check=True,
        )
        subprocess.run(
            [str(python), '-m', 'pip', 'install', '--disable-pip-version-check', '-r', str(requirements)],
            cwd=root, env=env, check=True,
        )
        subprocess.run([str(python), '-m', 'pip', 'check'], cwd=root, env=env, check=True)
        marker.write_text(fingerprint)
    return python


def gpu_preflight(python, env):
    if env.get('VOICE_DEVICE', '').strip().lower() == 'cpu':
        print('CPU-режим: перевірку CUDA пропущено. Генерація голосу може бути повільною.', flush=True)
        return
    print('Перевіряю CUDA на твоїй відеокарті…', flush=True)
    code = (
        'import torch\n'
        'if not torch.cuda.is_available():\n'
        ' raise SystemExit("CUDA недоступна. Онови драйвер NVIDIA й перезапусти Windows.")\n'
        'x = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)\n'
        'y = (x @ x).sum().item()\n'
        'torch.cuda.synchronize()\n'
        'print("GPU:", torch.cuda.get_device_name(0))\n'
        'print("Відеопам’ять:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), "ГБ")\n'
    )
    subprocess.run([str(python), '-c', code], env=env, check=True)


def open_local(request, timeout=10):
    # This direct route is limited to a local service; public downloads retain
    # their normal inherited proxy and TLS verification in prepare_node/pip.
    url = request.full_url if isinstance(request, urllib.request.Request) else request
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1':
        raise ValueError('Очікувався локальний HTTP-сервіс на 127.0.0.1.')
    return _LOCAL_OPENER.open(request, timeout=timeout)


def ensure_ollama():
    try:
        with open_local(OLLAMA_URL + '/api/tags') as response:
            data = json.load(response)
        names = {item.get('name', item.get('model')) for item in data['models']}
    except (OSError, ValueError, KeyError, TypeError):
        raise RuntimeError(
            'Ollama не відповідає. Встанови її з https://ollama.com/download/windows '
            'і запусти застосунок Ollama, потім знову відкрий start_ai.bat.'
        ) from None
    if OLLAMA_MODEL in names:
        return
    print(f'Завантажую локальну мовну модель {OLLAMA_MODEL} через Ollama…', flush=True)
    request = urllib.request.Request(
        OLLAMA_URL + '/api/pull',
        data=json.dumps({'model': OLLAMA_MODEL, 'stream': True}).encode(),
        headers={'Content-Type': 'application/json'}, method='POST',
    )
    success = False
    previous = None
    try:
        with open_local(request, timeout=120) as response:
            for line in response:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get('error'):
                    raise RuntimeError('Ollama не змогла завантажити модель. Перевір інтернет і вільне місце.')
                status = str(event.get('status', ''))
                total, completed = event.get('total'), event.get('completed', 0)
                percentage = int(completed * 100 / total) if isinstance(total, int) and total > 0 else None
                progress = (status, percentage)
                if progress != previous:
                    print(status + (f' {percentage}%' if percentage is not None else ''), flush=True)
                    previous = progress
                if status == 'success':
                    success = True
    except (OSError, ValueError, TypeError):
        raise RuntimeError('Не вдалося завершити завантаження Ollama. Перевір інтернет і повтори запуск.') from None
    if not success:
        raise RuntimeError('Ollama не підтвердила завершення завантаження. Спробуй запуск після перевірки її стану.')


def reference_path(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        value = value[1:-1]
    path = Path(value).expanduser()
    if not value or not path.is_file():
        raise RuntimeError('Файл голосового зразка не знайдено. Скопіюй повний шлях до MP4 або WAV.')
    if path.suffix.lower() not in {'.wav', '.mp3', '.mp4', '.m4a', '.webm', '.ogg', '.flac', '.aac', '.mov'}:
        raise RuntimeError('Вибери аудіо/відеофайл: наприклад MP4, WAV або MP3.')
    return path.resolve()


def choose_reference():
    value = os.environ.get('VOICE_REFERENCE_PATH')
    if not value:
        print('Голосовий зразок залишиться тільки на твоєму ПК.', flush=True)
        value = input('Перетягни MP4/WAV у це вікно або встав повний шлях і натисни Enter: ')
    return reference_path(value)


def read_discord_token():
    token = os.environ.get('DISCORD_TOKEN', '').strip()
    if token:
        return token
    print('Встав токен бота. Символи не показуватимуться — це нормально.', flush=True)
    with warnings.catch_warnings():
        # getpass otherwise falls back to echoed input when no console exists.
        warnings.simplefilter('error', getpass.GetPassWarning)
        try:
            return getpass.getpass('Discord token: ').strip()
        except getpass.GetPassWarning:
            raise RuntimeError(
                'Приховане введення недоступне. Відкрий start_ai.bat у звичайному вікні Windows.'
            ) from None


def safe_print(line, private_values=()):
    for value in private_values:
        if value:
            line = line.replace(value, '[приховано]')
    print(line.rstrip('\r\n'), flush=True)


def worker_url(value):
    try:
        parsed = urllib.parse.urlsplit(value)
        port = parsed.port
    except ValueError:
        raise RuntimeError('AI-процес повернув неправильну локальну адресу.') from None
    if (parsed.scheme != 'http' or parsed.hostname != '127.0.0.1'
            or parsed.username or parsed.password or parsed.path not in ('', '/')
            or parsed.query or parsed.fragment or not port):
        raise RuntimeError('AI-процес повернув неправильну локальну адресу.')
    return value.rstrip('/')


def wait_for_worker_ready(worker, private_values=()):
    for line in worker.stdout:
        if line.startswith('AI_WORKER_READY '):
            return worker_url(line[len('AI_WORKER_READY '):].strip())
        safe_print(line, private_values)
    raise RuntimeError('Голосова модель не запустилася. Перевір повідомлення вище; Discord ще не підключено.')


def drain_worker(worker, private_values=()):
    try:
        for line in worker.stdout:
            safe_print(line, private_values)
    except (OSError, ValueError):
        pass  # The supervised worker can be terminated while the pipe is read.


def stop_process(process):
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def supervise_children(worker, node):
    """Keep Discord connected only while the local AI worker remains alive."""
    while True:
        node_status = node.poll()
        worker_status = worker.poll()
        if node_status is not None:
            return node_status
        if worker_status is not None:
            raise RuntimeError(
                'Локальний AI-процес зупинився. Discord-бота буде відключено. '
                'Перевір повідомлення вище й повтори запуск.'
            )
        time.sleep(0.2)


def main():
    require_runtime()
    root = Path(__file__).resolve().parent
    print('Версія бота: ' + (root / 'AI_VERSION').read_text(encoding='utf-8').strip(), flush=True)
    env = dependency_env()
    node, npm = prepare_node(root)
    env['PATH'] = str(node.parent) + os.pathsep + env.get('PATH', '')
    prepare_node_packages(root, node, npm, env)
    python = prepare_python_dependencies(root, env)
    gpu_preflight(python, env)
    ensure_ollama()
    reference = choose_reference()
    secret = secrets.token_urlsafe(32)
    worker_env = dict(env, AI_WORKER_TOKEN=secret, VOICE_REFERENCE_PATH=str(reference))
    worker = node_process = None
    private_values = [secret]
    try:
        print('Готую локальні голосові моделі. Перший запуск може тривати кілька хвилин…', flush=True)
        worker = subprocess.Popen(
            [str(python), '-u', str(root / 'ai_worker.py'), '--port', '0'],
            cwd=root, env=worker_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace',
        )
        url = wait_for_worker_ready(worker, private_values)
        threading.Thread(target=drain_worker, args=(worker, private_values), daemon=True).start()
        token = read_discord_token()
        if not token:
            raise RuntimeError('Токен не введено. Запусти start_ai.bat ще раз.')
        private_values.append(token)
        node_env = dict(env, AI_WORKER_URL=url, AI_WORKER_TOKEN=secret, DISCORD_TOKEN=token)
        print('Підключаю AI-співрозмовницю до Discord…', flush=True)
        node_process = subprocess.Popen(
            [str(node), str(root / 'ai-bot.js')], cwd=root, env=node_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace',
        )
        threading.Thread(target=drain_worker, args=(node_process, private_values), daemon=True).start()
        status = supervise_children(worker, node_process)
        if status:
            raise SystemExit(status)
    finally:
        try:
            if node_process is not None:
                stop_process(node_process)
        finally:
            if worker is not None:
                stop_process(worker)


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print('\nAI-бот зупинено.')
    except Exception as error:
        raise SystemExit(f'Не вдалося запустити AI-бота: {error}') from None
