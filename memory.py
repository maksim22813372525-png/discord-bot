import random
import sqlite3


class Memory:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute('CREATE TABLE IF NOT EXISTS channels (id INTEGER PRIMARY KEY, low INTEGER, high INTEGER)')
        self.db.execute('CREATE TABLE IF NOT EXISTS phrases (id INTEGER PRIMARY KEY, channel INTEGER, author INTEGER, text TEXT)')
        self.db.commit()

    def channels(self):
        return self.db.execute('SELECT id, low, high FROM channels').fetchall()

    def enable(self, channel, low, high):
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO channels VALUES (?, ?, ?)', (channel, low, high))

    def disable(self, channel):
        with self.db:
            self.db.execute('DELETE FROM channels WHERE id = ?', (channel,))

    def remember(self, channel, author, text):
        if not text.strip() or not self.db.execute('SELECT 1 FROM channels WHERE id = ?', (channel,)).fetchone():
            return
        with self.db:
            self.db.execute('INSERT INTO phrases(channel, author, text) VALUES (?, ?, ?)', (channel, author, text[:2000]))
            self.db.execute('DELETE FROM phrases WHERE channel = ? AND id NOT IN (SELECT id FROM phrases WHERE channel = ? ORDER BY id DESC LIMIT 1000)', (channel, channel))

    def choose(self, channel):
        rows = self.db.execute('SELECT text FROM phrases WHERE channel = ?', (channel,)).fetchall()
        return random.choice(rows)[0] if rows else None

    def forget(self, channel, author=None):
        with self.db:
            if author is None:
                self.db.execute('DELETE FROM phrases WHERE channel = ?', (channel,))
            else:
                self.db.execute('DELETE FROM phrases WHERE channel = ? AND author = ?', (channel, author))
