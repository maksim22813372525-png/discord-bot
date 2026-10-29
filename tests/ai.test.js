import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter, once } from 'node:events';
import { PassThrough } from 'node:stream';
import { setImmediate as tick } from 'node:timers/promises';
import OpusScript from 'opusscript';
import { AudioPlayerStatus, VoiceConnectionStatus } from '@discordjs/voice';
import { ConversationSession, LocalWorker } from '../ai-session.js';
import { definitions } from '../ai-bot.js';

function fixture(t, post) {
  const streams = new Map();
  const errors = [];
  const members = new Map([
    ['u', { user: { bot: false }, voice: { channelId: 'c' } }],
    ['bot', { user: { bot: true }, voice: { channelId: 'c' } }],
  ]);
  const calls = [];
  const worker = { async post(route, data, signal) {
    calls.push({ route, data });
    return post ? post(route, data, signal) : { skipped: true };
  } };
  const session = new ConversationSession({
    channel: { id: 'c', guild: { id: 'g', members: { cache: members } } },
    worker, onClose() {}, onError: e => errors.push(e),
  });
  session.connection = {
    state: { status: VoiceConnectionStatus.Ready },
    receiver: { speaking: new EventEmitter(), subscribe(user) {
      const stream = new PassThrough({ objectMode: true }); streams.set(user, stream); return stream;
    } },
    destroy() { this.state.status = VoiceConnectionStatus.Destroyed; },
  };
  t.after(() => session.stop());
  return { session, streams, errors, calls };
}

function opusSpeech() {
  const encoder = new OpusScript(48000, 2, OpusScript.Application.VOIP);
  try {
    const raw = Buffer.alloc(960 * 2 * 2);
    for (let i = 0; i < 960; i++) {
      const value = Math.round(Math.sin(i * 2 * Math.PI * 440 / 48000) * 5000);
      raw.writeInt16LE(value, i * 4); raw.writeInt16LE(value, i * 4 + 2);
    }
    return Array.from({ length: 20 }, () => Buffer.from(encoder.encode(raw, 960)));
  } finally { encoder.delete(); }
}

test('real Opus receiver frames become bounded stereo PCM requests', async t => {
  const { session, streams, calls, errors } = fixture(t);
  session.capture('u');
  for (const packet of opusSpeech()) streams.get('u').write(packet);
  streams.get('u').end();
  await tick();
  const request = calls.find(c => c.route === '/reply');
  assert.equal(request.data.conversation, 'g:c');
  const pcm = Buffer.from(request.data.audio_pcm, 'base64');
  assert.equal(pcm.length, 20 * 3840);
  assert.ok(pcm.some(value => value !== 0));
  assert.deepEqual(errors, []);
});

test('bots, opted-out members, and playback are excluded', t => {
  const { session, streams } = fixture(t);
  session.capture('bot');
  session.ignored.add('u'); session.capture('u');
  session.ignored.delete('u');
  const original = session.player;
  session.player = { state: { status: AudioPlayerStatus.Playing }, stop() {} };
  session.capture('u'); session.player = original;
  assert.equal(streams.size, 0);
});

test('speech starting during inference is captured and processed when inference finishes', async t => {
  let resolve;
  const result = new Promise(r => { resolve = r; });
  let requests = 0;
  const { session, streams, calls, errors } = fixture(t, route =>
    route === '/reply' && ++requests === 1 ? result : { skipped: true });
  const first = session.respond('u', Buffer.alloc(3840));
  session.capture('u');
  for (const packet of opusSpeech()) streams.get('u').write(packet);
  streams.get('u').end();
  await tick();
  assert.ok(session.pending);
  assert.equal(calls.filter(c => c.route === '/reply').length, 1);
  resolve({ skipped: true });
  await first;
  await tick();
  const replies = calls.filter(c => c.route === '/reply');
  assert.equal(replies.length, 2);
  assert.equal(Buffer.from(replies[1].data.audio_pcm, 'base64').length, 20 * 3840);
  assert.equal(session.pending, null);
  assert.deepEqual(errors, []);
});

test('only the latest bounded pending turn is processed after playback', async t => {
  const { session, calls } = fixture(t);
  const original = session.player;
  session.player = { state: { status: AudioPlayerStatus.Playing }, stop() {} };
  await session.respond('u', Buffer.from([1, 0, 1, 0]));
  await session.respond('u', Buffer.alloc(4_000_000, 2));
  assert.equal(calls.length, 0);
  assert.equal(session.pending.pcm.length, 3_840_000);
  session.player = original;
  original.emit(AudioPlayerStatus.Idle);
  await tick();
  assert.equal(calls.length, 1);
  assert.equal(Buffer.from(calls[0].data.audio_pcm, 'base64').length, 3_840_000);
  assert.equal(Buffer.from(calls[0].data.audio_pcm, 'base64')[0], 2);
});

test('forget and stop discard pending speech before playback idle can drain it', async t => {
  for (const action of ['forget', 'stop']) {
    const { session, calls } = fixture(t);
    session.busy = true;
    await session.respond('u', Buffer.alloc(3840));
    assert.ok(session.pending);
    // stop() synchronously emits Idle in the real player.
    session.player.stop = () => {
      session.busy = false;
      session.player.emit(AudioPlayerStatus.Idle);
    };
    await session[action]();
    await tick();
    assert.equal(session.pending, null);
    assert.equal(calls.filter(c => c.route === '/reply').length, 0);
  }
});

test('starting playback finishes an ongoing capture before it can record bot audio', async t => {
  let resolve;
  const result = new Promise(r => { resolve = r; });
  const { session, streams, calls } = fixture(t, route => route === '/reply' ? result : { ok: true });
  let resource;
  session.player = { state: { status: AudioPlayerStatus.Idle }, stop() {}, play(r) {
    resource = r;
    assert.equal(session.captures.size, 0);
    this.state.status = AudioPlayerStatus.Playing;
  } };
  const first = session.respond('u', Buffer.alloc(3840));
  session.capture('u');
  for (const packet of opusSpeech()) streams.get('u').write(packet);
  resolve({ audio_pcm: Buffer.alloc(3840).toString('base64'), sample_rate: 48000, channels: 2 });
  await first;
  assert.equal(streams.get('u').destroyed, true);
  assert.ok(session.pending);
  assert.equal(calls.filter(c => c.route === '/reply').length, 1);
  resource.playStream.destroy();
});

test('clearing cancels active recordings and late responses cannot play or reappear', async t => {
  let resolve;
  const result = new Promise(r => { resolve = r; });
  const { session } = fixture(t, route => route === '/reply' ? result : { ok: true });
  const played = [];
  session.player = { state: { status: AudioPlayerStatus.Idle }, stop() {}, play: r => played.push(r) };
  const pending = session.respond('u', Buffer.alloc(3840));
  await session.forget('u');
  resolve({ audio_pcm: Buffer.alloc(3840).toString('base64'), sample_rate: 48000, channels: 2 });
  await pending;
  assert.equal(played.length, 0);
  assert.equal(session.ignored.has('u'), true);
});

test('synthesized PCM is encoded for Discord by the real audio resource', async t => {
  const raw = Buffer.alloc(3840 * 20);
  for (let i = 0; i < raw.length / 4; i++) {
    const value = Math.round(Math.sin(i * 2 * Math.PI * 440 / 48000) * 5000);
    raw.writeInt16LE(value, i * 4); raw.writeInt16LE(value, i * 4 + 2);
  }
  const { session, errors } = fixture(t, () => ({ audio_pcm: raw.toString('base64'), sample_rate: 48000, channels: 2 }));
  let resource;
  session.player = { state: { status: AudioPlayerStatus.Idle }, stop() {}, play: r => { resource = r; } };
  await session.respond('u', Buffer.alloc(3840));
  assert.ok(resource);
  if (!resource.playStream.readableLength) await once(resource.playStream, 'readable');
  const opus = resource.read();
  assert.ok(Buffer.isBuffer(opus) && opus.length);
  const decoder = new OpusScript(48000, 2, OpusScript.Application.VOIP);
  try { assert.equal(decoder.decode(opus).length, 3840); }
  finally { decoder.delete(); resource.playStream.destroy(); }
  assert.deepEqual(errors, []);
});

test('invalid model audio reports a failure and releases the inference slot', async t => {
  const { session, errors } = fixture(t, () => ({ audio_pcm: 'AA==', sample_rate: 48000, channels: 2 }));
  await session.respond('u', Buffer.alloc(3840));
  assert.equal(errors.length, 1);
  assert.equal(session.busy, false);
});

test('worker calls use local endpoints and ephemeral authentication', async () => {
  assert.throws(() => new LocalWorker('https://example.com', 'token'));
  const calls = [];
  const worker = new LocalWorker('http://127.0.0.1:1234', 'test-only', async (...args) => {
    calls.push(args); return { ok: true, async json() { return { skipped: true }; } };
  });
  await worker.post('/reply', { conversation: 'g:c' });
  assert.equal(calls[0][0], 'http://127.0.0.1:1234/reply');
  assert.equal(calls[0][1].headers.Authorization, 'Bearer test-only');
});

test('AI command registration preserves runtime admin requirements', () => {
  assert.deepEqual(definitions.map(c => c.name), ['talk_on', 'talk_off', 'talk_clear', 'talk_heard', 'forget_me', 'remember_me']);
  for (const c of definitions.slice(0, 3)) assert.equal(c.default_member_permissions, '32');
  assert.equal(definitions.find(c => c.name === 'talk_heard').default_member_permissions, undefined);
});
