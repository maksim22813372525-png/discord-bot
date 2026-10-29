import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { EventEmitter, once } from 'node:events';
import { PassThrough, Readable } from 'node:stream';
import { setImmediate as tick } from 'node:timers/promises';
import { AudioPlayerStatus, StreamType, VoiceConnectionStatus, createAudioResource, generateDependencyReport } from '@discordjs/voice';
import { ClipStore, Phrase, SILENCE, MAX_FRAMES } from '../clips.js';
import { VoiceSession } from '../voice-session.js';
import { commands } from '../bot.js';

const packet = Buffer.from([0x78, 0x12, 0x34, 0x56]);
const packets = () => Array.from({ length: 20 }, () => Buffer.from(packet));
const temporary = t => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'voice-bot-test-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  return directory;
};

test('short pauses and original Opus bytes are preserved; silence and clicks are discarded', () => {
  const phrase = new Phrase(0);
  for (let i = 0; i < 15; i++) phrase.add(packet, i * 20);
  phrase.add(packet, 500);
  phrase.add(SILENCE, 520);
  const result = phrase.finish();
  assert.equal(result.length, 26);
  assert.deepEqual(result[0], packet);
  assert.deepEqual(result[15], SILENCE);
  assert.deepEqual(result.at(-1), packet);
  const silent = new Phrase(0);
  for (let i = 0; i < 20; i++) silent.add(SILENCE, i * 20);
  assert.deepEqual(silent.finish(), []);
  const click = new Phrase(0);
  click.add(packet, 0);
  assert.deepEqual(click.finish(), []);
});

test('recordings are limited to eight seconds', () => {
  const phrase = new Phrase(0);
  for (let i = 0; i < 1000; i++) phrase.add(packet, i * 20);
  assert.equal(phrase.finish().length, MAX_FRAMES);
});

test('clips survive restart and are isolated by guild and voice channel', t => {
  const directory = temporary(t);
  const store = new ClipStore(directory);
  store.add('guild1', 'room1', 'user1', packets());
  store.add('guild2', 'room1', 'user2', packets());
  const restored = new ClipStore(directory);
  assert.deepEqual(restored.choose('guild1', 'room1').packets, packets());
  assert.equal(restored.choose('guild1', 'room2'), null);
  assert.equal(restored.choose('guild3', 'room1'), null);
  store.clear('guild1');
  assert.equal(store.choose('guild1', 'room1'), null);
  assert.equal(store.choose('guild2', 'room1').user, 'user2');
});

test('forget_me deletes only that user and persists opt-out until explicitly reversed', t => {
  const directory = temporary(t);
  const store = new ClipStore(directory);
  store.add('g', 'c', 'u', packets());
  store.add('g', 'c', 'other', packets());
  store.ignore('g', 'u');
  store.add('g', 'c', 'u', packets());
  assert.deepEqual(store.data.clips.map(c => c.user), ['other']);
  const restored = new ClipStore(directory);
  assert.equal(restored.isIgnored('g', 'u'), true);
  restored.ignore('g', 'u', false);
  restored.add('g', 'c', 'u', packets());
  assert.equal(restored.data.clips.length, 2);
});

test('retains the most recent 100 clips per guild', t => {
  const store = new ClipStore(temporary(t));
  store.add('other-guild', 'c', 'u', packets());
  for (let i = 0; i < 105; i++) store.add('g', 'c', String(i), packets());
  assert.equal(store.data.clips.filter(c => c.guild === 'g').length, 100);
  assert.equal(store.data.clips.filter(c => c.guild === 'g')[0].user, '5');
  assert.equal(store.choose('other-guild', 'c').user, 'u');
});

function sessionFixture(t) {
  const store = new ClipStore(temporary(t));
  const members = new Map([
    ['u', { user: { bot: false }, voice: { channelId: 'c' } }],
    ['bot', { user: { bot: true }, voice: { channelId: 'c' } }],
    ['elsewhere', { user: { bot: false }, voice: { channelId: 'elsewhere' } }],
  ]);
  const streams = new Map();
  const errors = [];
  const session = new VoiceSession({
    channel: { id: 'c', guild: { id: 'g', members: { cache: members } } },
    store, minimum: 1, maximum: 5, onClose() {}, onError: error => errors.push(error),
  });
  session.connection = {
    state: { status: VoiceConnectionStatus.Ready },
    receiver: { speaking: new EventEmitter(), subscribe(user) {
      const stream = new PassThrough({ objectMode: true });
      streams.set(user, stream);
      return stream;
    } },
    destroy() { this.state.status = VoiceConnectionStatus.Destroyed; },
  };
  t.after(() => session.stop());
  return { store, streams, session, errors };
}

test('receiver packets become playable original clips; bots and other rooms are excluded', async t => {
  const { session, streams, store, errors } = sessionFixture(t);
  session.capture('bot');
  session.capture('elsewhere');
  assert.equal(streams.size, 0);
  session.capture('u');
  for (const p of packets()) streams.get('u').write(p);
  streams.get('u').end();
  await tick();
  assert.deepEqual(store.choose('g', 'c').packets, packets());
  assert.deepEqual(errors, []);
});

test('stop, clear, and opt-out cancel pending buffers so deleted audio cannot reappear', async t => {
  for (const action of ['stop', 'clear', 'ignore']) {
    const { session, streams, store } = sessionFixture(t);
    session.capture('u');
    for (const p of packets()) streams.get('u').write(p);
    if (action === 'stop') session.stop();
    else {
      session.discard(action === 'ignore' ? 'u' : undefined);
      if (action === 'ignore') store.ignore('g', 'u');
      else store.clear('g');
    }
    await tick();
    assert.equal(store.choose('g', 'c'), null, action);
    assert.equal(streams.get('u').destroyed, true, action);
    assert.equal(session.captures.size, 0, action);
  }
});

test('no incoming clips are captured during playback or after opt-out', t => {
  const { session, store, streams } = sessionFixture(t);
  store.ignore('g', 'u');
  session.capture('u');
  assert.equal(streams.size, 0);
  store.ignore('g', 'u', false);
  const original = session.player;
  session.player = { state: { status: AudioPlayerStatus.Playing }, stop() {} };
  session.capture('u');
  assert.equal(streams.size, 0);
  session.player = original;
});

test('real Discord audio resource passes stored Opus packets through without a codec or FFmpeg', async () => {
  const resource = createAudioResource(Readable.from(packets()), { inputType: StreamType.Opus });
  if (!resource.playStream.readableLength) await once(resource.playStream, 'readable');
  assert.deepEqual(resource.read(), packet);
  resource.playStream.destroy();
  assert.match(generateDependencyReport(), /davey: 0\./);
});

test('random scheduler respects the minimum interval and stops on voice_off', async t => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  t.mock.method(Math, 'random', () => 0);
  const { session, store } = sessionFixture(t);
  store.add('g', 'c', 'u', packets());
  session.channel.members = { some: () => true };
  const played = [];
  session.player = {
    state: { status: AudioPlayerStatus.Idle },
    play(resource) { played.push(resource); }, stop() {},
  };
  session.schedule();
  t.mock.timers.tick(59_999);
  assert.equal(played.length, 0);
  t.mock.timers.tick(1);
  assert.equal(played.length, 1);
  session.stop();
  t.mock.timers.tick(600_000);
  assert.equal(played.length, 1);
  played[0].playStream.destroy();
});

test('only voice commands are registered and administrative commands require ManageGuild', () => {
  const definitions = commands().map(c => c.toJSON());
  assert.deepEqual(definitions.map(c => c.name), ['voice_on', 'voice_off', 'voice_clear', 'forget_me', 'remember_me']);
  for (const c of definitions.slice(0, 3)) assert.equal(c.default_member_permissions, '32');
});
