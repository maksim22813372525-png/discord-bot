import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_ai_windows as launcher


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'ai-requirements.txt').write_text('qwen-tts==0.1.1\n')
        (self.root / 'package-lock.json').write_text('{"lockfileVersion":3}')
        self.python = self.root / '.ai-venv/Scripts/python.exe'
        self.python.parent.mkdir(parents=True)
        self.python.touch()

    def test_python314_is_rejected_before_installation(self):
        with patch.object(launcher.sys, 'version_info', (3, 14, 0)), patch.object(launcher.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'Python 3.12'):
                launcher.require_runtime()
            run.assert_not_called()

    def test_explicit_cpu_mode_skips_cuda_check(self):
        with patch.object(launcher.subprocess, 'run') as run:
            launcher.gpu_preflight(self.python, {'VOICE_DEVICE': 'cpu'})
            run.assert_not_called()

    def test_default_and_cuda_devices_run_cuda_check(self):
        for env in ({}, {'VOICE_DEVICE': 'cuda:0'}):
            with self.subTest(env=env), patch.object(launcher.subprocess, 'run') as run:
                launcher.gpu_preflight(self.python, env)
                run.assert_called_once()
                self.assertIn('torch.cuda.synchronize()', run.call_args.args[0][-1])

    def test_installation_environment_does_not_contain_tokens_or_mutate_parent(self):
        with patch.dict(os.environ, {'DISCORD_TOKEN': 'discord-private', 'AI_WORKER_TOKEN': 'worker-private', 'AI_WORKER_URL': 'http://127.0.0.1:1'}):
            env = launcher.dependency_env()
            self.assertNotIn('DISCORD_TOKEN', env)
            self.assertNotIn('AI_WORKER_TOKEN', env)
            self.assertNotIn('AI_WORKER_URL', env)
            self.assertEqual(os.environ['DISCORD_TOKEN'], 'discord-private')

    def test_token_input_refuses_echoed_fallback_when_console_is_missing(self):
        with patch.dict(os.environ, {'DISCORD_TOKEN': ''}), patch.object(
            launcher.getpass, 'getpass', side_effect=launcher.getpass.GetPassWarning('console unavailable')
        ):
            with self.assertRaisesRegex(RuntimeError, 'Приховане введення недоступне'):
                launcher.read_discord_token()

    def test_install_marker_is_written_only_after_both_pip_steps_and_check(self):
        with patch.object(launcher.subprocess, 'run', return_value=Mock(returncode=0)) as run:
            result = launcher.prepare_python_dependencies(self.root, {})
            self.assertEqual(result, self.python)
            calls = [call.args[0] for call in run.call_args_list]
            self.assertEqual(len(calls), 4)
            self.assertIn('torch==2.9.1', calls[1])
            self.assertIn(launcher.TORCH_INDEX, calls[1])
            self.assertIn('-r', calls[2])
            self.assertEqual(calls[3][-2:], ['pip', 'check'])
            marker = self.root / '.ai-venv/.voice-ai-installed'
            self.assertTrue(marker.is_file())
            launcher.prepare_python_dependencies(self.root, {})
            self.assertEqual(run.call_count, 5)  # Interpreter check only on the second run.

    def test_failed_requirement_install_does_not_leave_success_marker(self):
        def run(command, **kwargs):
            if '-r' in command:
                raise subprocess.CalledProcessError(1, command)
            return Mock(returncode=0)
        with patch.object(launcher.subprocess, 'run', side_effect=run):
            with self.assertRaises(subprocess.CalledProcessError):
                launcher.prepare_python_dependencies(self.root, {})
        self.assertFalse((self.root / '.ai-venv/.voice-ai-installed').exists())

    def test_node_dependency_cache_does_not_claim_success_on_failure(self):
        with patch.object(launcher.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, ['npm'])):
            with self.assertRaises(subprocess.CalledProcessError):
                launcher.prepare_node_packages(self.root, Path('node.exe'), Path('npm.js'), {})
        self.assertFalse((self.root / 'node_modules/.voice-bot-lock').exists())

    def test_installed_ollama_model_is_not_pulled_again(self):
        body = json.dumps({'models': [{'name': 'qwen3:4b'}]}).encode()
        with patch.object(launcher, 'open_local', return_value=io.BytesIO(body)) as open_local:
            launcher.ensure_ollama()
            self.assertEqual(open_local.call_count, 1)

    def test_missing_ollama_model_is_pulled_and_success_is_required(self):
        tags = io.BytesIO(b'{"models":[]}')
        stream = io.BytesIO(b'{"status":"pulling manifest"}\n{"status":"success"}\n')
        with patch.object(launcher, 'open_local', side_effect=[tags, stream]) as open_local:
            launcher.ensure_ollama()
            request = open_local.call_args_list[1].args[0]
            self.assertEqual(json.loads(request.data)['model'], 'qwen3:4b')
        with patch.object(launcher, 'open_local', side_effect=[io.BytesIO(b'{"models":[]}'), io.BytesIO(b'{"status":"pulling manifest"}\n')]):
            with self.assertRaisesRegex(RuntimeError, 'не підтвердила'):
                launcher.ensure_ollama()

    def test_only_local_service_urls_are_permitted(self):
        for url in ['https://127.0.0.1:12', 'http://example.com:12', 'http://127.0.0.1:word', 'http://private@127.0.0.1:12', 'http://127.0.0.1:12/path']:
            with self.subTest(url=url), self.assertRaises(RuntimeError):
                launcher.worker_url(url)
        self.assertEqual(launcher.worker_url('http://127.0.0.1:4567/'), 'http://127.0.0.1:4567')
        with self.assertRaises(ValueError):
            launcher.open_local('https://example.com/api/tags')

    def test_reference_with_spaces_and_paired_quotes_is_a_local_file(self):
        file = self.root / 'my voice.mp4'
        file.write_bytes(b'fixture')
        self.assertEqual(launcher.reference_path(f' "{file}" '), file.resolve())
        with self.assertRaisesRegex(RuntimeError, 'не знайдено'):
            launcher.reference_path('https://example.com/reference.mp4')

    def test_ready_line_is_parsed_and_logs_redact_worker_token(self):
        worker = Mock(stdout=io.StringIO('loading secret-value\nAI_WORKER_READY http://127.0.0.1:9876\n'))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(launcher.wait_for_worker_ready(worker, ('secret-value',)), 'http://127.0.0.1:9876')
        self.assertNotIn('secret-value', output.getvalue())
        self.assertIn('[приховано]', output.getvalue())
        with self.assertRaisesRegex(RuntimeError, 'не запустилася'):
            launcher.wait_for_worker_ready(Mock(stdout=io.StringIO('')))

    def test_forwarded_logs_redact_both_worker_and_discord_tokens(self):
        process = Mock(stdout=io.StringIO('error worker-private and discord-private\n'))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            launcher.drain_worker(process, ['worker-private', 'discord-private'])
        self.assertNotIn('worker-private', output.getvalue())
        self.assertNotIn('discord-private', output.getvalue())
        self.assertEqual(output.getvalue().count('[приховано]'), 2)

    def test_stop_process_terminates_only_its_owned_process_and_handles_timeout(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired(['worker'], 10), 0]
        launcher.stop_process(process)
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()

    def main_patches(self, node_status=0, worker_status=None, interrupt=False):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        for name in ('require_runtime', 'prepare_node_packages', 'gpu_preflight', 'ensure_ollama'):
            stack.enter_context(patch.object(launcher, name))
        stack.enter_context(patch.object(launcher, 'prepare_node', return_value=(Path('node.exe'), Path('npm.js'))))
        stack.enter_context(patch.object(launcher, 'prepare_python_dependencies', return_value=self.python))
        stack.enter_context(patch.object(launcher, 'choose_reference', return_value=self.root / 'voice.mp4'))
        stack.enter_context(patch.object(launcher, 'wait_for_worker_ready', return_value='http://127.0.0.1:1234'))
        stack.enter_context(patch.object(launcher.threading, 'Thread'))
        stack.enter_context(patch.object(launcher.secrets, 'token_urlsafe', return_value='ephemeral-worker-secret'))
        worker, node = Mock(), Mock()
        worker.poll.return_value = worker_status
        node.poll.return_value = node_status
        popen = stack.enter_context(patch.object(launcher.subprocess, 'Popen', side_effect=[worker, node]))
        if interrupt:
            stack.enter_context(patch.object(launcher, 'supervise_children', side_effect=KeyboardInterrupt()))
        stop = stack.enter_context(patch.object(launcher, 'stop_process'))
        return popen, worker, node, stop

    def test_worker_never_gets_discord_token_and_is_stopped_after_node_exit(self):
        popen, worker, node, stop = self.main_patches()
        with patch.dict(os.environ, {'DISCORD_TOKEN': 'private-discord-token'}):
            launcher.main()
        worker_env = popen.call_args_list[0].kwargs['env']
        self.assertNotIn('DISCORD_TOKEN', worker_env)
        self.assertEqual(worker_env['AI_WORKER_TOKEN'], 'ephemeral-worker-secret')
        node_env = popen.call_args_list[1].kwargs['env']
        self.assertEqual(node_env['DISCORD_TOKEN'], 'private-discord-token')
        self.assertEqual(node_env['AI_WORKER_URL'], 'http://127.0.0.1:1234')
        self.assertEqual(node_env['AI_WORKER_TOKEN'], worker_env['AI_WORKER_TOKEN'])
        self.assertEqual([c.args[0] for c in stop.call_args_list], [node, worker])

    def test_ctrl_c_while_node_runs_stops_both_owned_children(self):
        popen, worker, node, stop = self.main_patches(interrupt=True)
        with patch.dict(os.environ, {'DISCORD_TOKEN': 'private-token'}):
            with self.assertRaises(KeyboardInterrupt):
                launcher.main()
        self.assertEqual([c.args[0] for c in stop.call_args_list], [node, worker])

    def test_worker_exit_after_ready_disconnects_node_and_stops_both_children(self):
        popen, worker, node, stop = self.main_patches(node_status=None)
        worker.poll.side_effect = [None, 1]
        with patch.dict(os.environ, {'DISCORD_TOKEN': 'private-token'}), patch.object(launcher.time, 'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'AI-процес зупинився'):
                launcher.main()
        self.assertEqual(popen.call_count, 2)
        sleep.assert_called_once_with(0.2)
        self.assertEqual([c.args[0] for c in stop.call_args_list], [node, worker])

    def test_supervisor_waits_while_alive_and_returns_node_exit_status(self):
        worker, node = Mock(), Mock()
        worker.poll.return_value = None
        node.poll.side_effect = [None, None, 7]
        with patch.object(launcher.time, 'sleep') as sleep:
            self.assertEqual(launcher.supervise_children(worker, node), 7)
        self.assertEqual(sleep.call_count, 2)


if __name__ == '__main__':
    unittest.main()
