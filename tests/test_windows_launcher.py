import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_windows as launcher


def archive(extra=None):
    output = io.BytesIO()
    folder = f'node-v{launcher.NODE_VERSION}-win-x64/'
    with zipfile.ZipFile(output, 'w') as z:
        z.writestr(folder + 'node.exe', b'fixture-not-an-executable')
        z.writestr(folder + 'node_modules/npm/bin/npm-cli.js', b'fixture')
        if extra:
            z.writestr(extra, b'never-extract')
    return output.getvalue()


class LauncherTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for target, value in [('platform.system', 'Windows'), ('platform.machine', 'AMD64')]:
            mock = patch('run_windows.' + target, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)
        self.run = patch('run_windows.subprocess.run')
        self.runner = self.run.start()
        self.addCleanup(self.run.stop)

    def download(self, payload, expected=None):
        digest = expected or hashlib.sha256(payload).hexdigest()
        return patch('run_windows.urllib.request.urlopen', side_effect=lambda *a, **kw: io.BytesIO(payload)), patch.dict(launcher.NODE_HASHES, {'x64': digest})

    def test_verified_download_is_reused(self):
        network, hashes = self.download(archive())
        with network as get, hashes:
            node, npm = launcher.prepare_node(self.root)
            self.assertTrue(node.is_file())
            self.assertTrue(npm.is_file())
            launcher.prepare_node(self.root)
            self.assertEqual(get.call_count, 1)
            self.assertEqual(self.runner.call_count, 2)

    def test_corrupt_download_cannot_execute(self):
        network, hashes = self.download(archive(), '0' * 64)
        with network, hashes:
            with self.assertRaisesRegex(RuntimeError, 'SHA-256'):
                launcher.prepare_node(self.root)
        self.runner.assert_not_called()
        self.assertEqual(list(self.root.rglob('node.exe')), [])
        self.assertFalse((self.root / '.runtime/node-download.zip').exists())

    def test_zip_cannot_escape_runtime(self):
        network, hashes = self.download(archive('../../outside'))
        with network, hashes:
            with self.assertRaisesRegex(RuntimeError, 'ZIP'):
                launcher.prepare_node(self.root)
        self.runner.assert_not_called()
        self.assertEqual(list(self.root.rglob('node.exe')), [])

    def test_partial_extraction_is_not_trusted(self):
        folder = self.root / '.runtime' / f'node-v{launcher.NODE_VERSION}-win-x64'
        (folder / 'node_modules/npm/bin').mkdir(parents=True)
        (folder / 'node.exe').write_bytes(b'partial')
        (folder / 'node_modules/npm/bin/npm-cli.js').write_bytes(b'partial')
        network, hashes = self.download(archive())
        with network as get, hashes:
            launcher.prepare_node(self.root)
            self.assertEqual(get.call_count, 1)
        self.assertEqual((folder / 'node.exe').read_bytes(), b'fixture-not-an-executable')


if __name__ == '__main__':
    unittest.main()
