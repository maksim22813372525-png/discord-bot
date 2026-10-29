import unittest
from memory import Memory


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.memory = Memory(':memory:')
        self.addCleanup(self.memory.db.close)

    def test_channel_isolation_and_disable(self):
        self.memory.remember(1, 7, 'ignored')
        self.assertIsNone(self.memory.choose(1))
        self.memory.enable(1, 1, 2)
        self.memory.remember(1, 7, 'hello')
        self.assertEqual(self.memory.choose(1), 'hello')
        self.assertIsNone(self.memory.choose(2))
        self.memory.disable(1)
        self.memory.remember(1, 7, 'ignored again')
        self.assertEqual(self.memory.choose(1), 'hello')
        self.assertEqual(self.memory.channels(), [])

    def test_forget_and_retention(self):
        self.memory.enable(1, 1, 2)
        for i in range(1005):
            self.memory.remember(1, 7, str(i))
        rows = self.memory.db.execute('SELECT text FROM phrases').fetchall()
        self.assertEqual(len(rows), 1000)
        self.assertNotIn(('0',), rows)
        self.memory.remember(1, 8, 'keep')
        self.memory.forget(1, 7)
        self.assertEqual(self.memory.choose(1), 'keep')
        self.memory.forget(1)
        self.assertIsNone(self.memory.choose(1))


if __name__ == '__main__':
    unittest.main()
