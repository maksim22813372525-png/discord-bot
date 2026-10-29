import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter, once } from 'node:events';
import { PassThrough } from 'node:stream';
import { setImmediate as tick } from 'node:timers/promises';
import OpusScript from 'opusscript';
import { AudioPlayerStatus, VoiceConnectionStatus } from '@discordjs/voice';
import { ConversationSession, LocalWorker, PLAYBACK_TAIL_MS } from '../ai-session.js';
import { definitions, registerCommands } from '../ai-bot.js';
import { formatDiagnostic } from '../ai-status.js';

function fixture(t, post, now) {
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
    worker, onClose() {}, onError: e => errors.push(e), now,
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

function opusSpeech(frameSize = 960, packets = 20) {
  const encoder = new OpusScript(48000, 2, OpusScript.Application.VOIP);
  try {
    const raw = Buffer.alloc(frameSize * 2 * 2);
    for (let i = 0; i < frameSize; i++) {
      const value = Math.round(Math.sin(i * 2 * Math.PI * 440 / 48000) * 5000);
      raw.writeInt16LE(value, i * 4); raw.writeInt16LE(value, i * 4 + 2);
    }
    return Array.from({ length: packets }, () => Buffer.from(encoder.encode(raw, frameSize)));
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

test('short speech is measured in decoded samples, including 60 ms Opus packets', async t => {
  for (const [frameSize, packets] of [[960, 12], [2880, 4]]) {
    const { session, streams, calls, errors } = fixture(t);
    session.capture('u');
    for (const packet of opusSpeech(frameSize, packets)) streams.get('u').write(packet);
    streams.get('u').end();
    await tick();
    const request = calls.find(c => c.route === '/reply');
    assert.ok(request, `${packets} x ${frameSize / 48} ms packets must reach ASR`);
    assert.equal(Buffer.from(request.data.audio_pcm, 'base64').length, 48_000 * 4 * 0.24);
    const diagnostic = session.diagnostic('u');
    assert.equal(diagnostic.received_packets, packets);
    assert.equal(diagnostic.duration_ms, 240);
    assert.equal(diagnostic.stage, 'skipped');
    assert.deepEqual(errors, []);
  }
});

test('receive diagnostics distinguish no packets, receiving, and rejected short audio', async t => {
  const { session, streams, calls } = fixture(t);
  assert.equal(session.diagnostic('u').stage, 'none');
  session.capture('u');
  assert.equal(session.diagnostic('u').stage, 'receiving');
  assert.equal(session.diagnostic('u').received_packets, 0);
  for (const packet of opusSpeech(960, 4)) streams.get('u').write(packet);
  streams.get('u').end();
  await tick();
  const diagnostic = session.diagnostic('u');
  assert.equal(diagnostic.stage, 'too_short');
  assert.equal(diagnostic.duration_ms, 80);
  assert.equal(diagnostic.received_packets, 4);
  assert.equal(calls.filter(c => c.route === '/reply').length, 0);
  assert.equal(session.diagnostic('other').received_packets, 0);
  assert.equal(session.diagnostic('other').member_known, false);
  assert.equal(session.diagnostic('other').same_channel, false);
  assert.equal('audio_pcm' in diagnostic, false);
  assert.equal('heard' in diagnostic, false);
  assert.equal('answer' in diagnostic, false);
});

test('diagnostics expose local deafness, opt-out, channel and request failure flags', async t => {
  const { session, errors } = fixture(t, () => { throw new Error('test worker failure'); });
  session.guild.members.me = { voice: { selfDeaf: true, serverDeaf: true } };
  session.ignored.add('u');
  let diagnostic = session.diagnostic('u');
  assert.equal(diagnostic.self_deaf, true);
  assert.equal(diagnostic.server_deaf, true);
  assert.equal(diagnostic.ignored, true);
  assert.equal(diagnostic.same_channel, true);
  session.ignored.delete('u');
  await session.respond('u', Buffer.alloc(3840));
  diagnostic = session.diagnostic('u');
  assert.equal(diagnostic.stage, 'error');
  assert.equal(diagnostic.reason, 'worker_failed');
  assert.equal(errors.length, 1);
  assert.equal(diagnostic.busy, false);
});

test('receive diagnostics are bounded and cleared by forget/stop', async t => {
  const { session } = fixture(t);
  for (let i = 0; i < 130; i++) session.note(`member-${i}`, 'receiving', { received_packets: i });
  assert.equal(session.receiveDiagnostics.size, 100);
  assert.equal(session.diagnostic('member-0').stage, 'none');
  const previousEpoch = session.epoch;
  await session.forget();
  assert.equal(session.receiveDiagnostics.size, 0);
  session.note('u', 'playing', {}, previousEpoch);
  assert.equal(session.receiveDiagnostics.size, 0);
  session.note('u', 'receiving');
  session.stop();
  session.note('u', 'error');
  assert.equal(session.receiveDiagnostics.size, 0);
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

test('delayed microphone echo is blocked after real playback but fresh speech is accepted after the guard', async t => {
  let now = 1000;
  const { session, streams, calls } = fixture(t, undefined, () => now);
  session.playbackUser = 'u';
  session.note('u', 'playing');
  session.player.emit(AudioPlayerStatus.Idle);
  session.capture('u');
  assert.equal(streams.size, 0);
  assert.equal(session.diagnostic('u').stage, 'echo_blocked');
  assert.equal(session.diagnostic('u').cooldown_ms, PLAYBACK_TAIL_MS);
  now += PLAYBACK_TAIL_MS - 1;
  session.capture('u');
  assert.equal(streams.size, 0);
  now += 1;
  session.capture('u');
  for (const packet of opusSpeech()) streams.get('u').write(packet);
  streams.get('u').end();
  await tick();
  assert.equal(calls.filter(c => c.route === '/reply').length, 1);
  assert.equal(session.diagnostic('u').cooldown_ms, 0);
});

test('idle without actual bot playback does not delay microphone capture', t => {
  const { session, streams } = fixture(t, undefined, () => 1000);
  session.player.emit(AudioPlayerStatus.Idle);
  session.player.emit(AudioPlayerStatus.Idle);
  session.capture('u');
  assert.equal(streams.size, 1);
  assert.equal(session.diagnostic('u').cooldown_ms, 0);
});

test('a genuine pending turn is processed immediately while new playback-tail recordings are blocked', async t => {
  const { session, calls, streams } = fixture(t, undefined, () => 1000);
  session.busy = true;
  await session.respond('u', Buffer.alloc(3840));
  session.busy = false;
  session.playbackUser = 'u';
  session.player.emit(AudioPlayerStatus.Idle);
  session.capture('u');
  await tick();
  assert.equal(calls.filter(c => c.route === '/reply').length, 1);
  assert.equal(session.pending, null);
  assert.equal(streams.size, 0);
});

test('forgetting during playback guards its tail without restoring deleted diagnostics', async t => {
  let now = 1000;
  const { session, streams } = fixture(t, undefined, () => now);
  await session.forget();
  assert.equal(session.diagnostic('u').cooldown_ms, 0);
  session.playbackUser = 'u';
  session.note('u', 'playing');
  await session.forget();
  assert.equal(session.diagnostic('u').stage, 'none');
  assert.equal(session.diagnostic('u').cooldown_ms, PLAYBACK_TAIL_MS);
  session.capture('u');
  assert.equal(streams.size, 0);
  now += PLAYBACK_TAIL_MS;
  session.capture('u');
  assert.equal(streams.size, 1);
  session.stop();
  assert.equal(session.receiveDiagnostics.size, 0);
  assert.equal(session.captureAfter, 0);
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
  assert.deepEqual(definitions.map(c => c.name), ['talk_on', 'talk_off', 'talk_clear', 'talk_heard', 'talk_test', 'forget_me', 'remember_me']);
  for (const c of definitions.slice(0, 3)) assert.equal(c.default_member_permissions, '32');
  assert.equal(definitions.find(c => c.name === 'talk_heard').default_member_permissions, undefined);
});

test('commands refresh in each server as well as the global command list', async () => {
  const registered = [];
  const commands = name => ({ async set(value) { registered.push({ name, value }); } });
  await registerCommands({ application: { commands: commands('global') },
    guilds: { cache: new Map([['g', { commands: commands('g') }], ['h', { commands: commands('h') }]]) } });
  assert.deepEqual(registered.map(c => c.name), ['global', 'g', 'h']);
  for (const target of registered) {
    assert.ok(target.value.some(c => c.name === 'talk_heard'));
    assert.ok(target.value.some(c => c.name === 'talk_test'));
  }
});

test('voice test bypasses microphone and uses the normal playable audio path', async t => {
  let resource;
  const { session, calls } = fixture(t, () => ({ audio_pcm: Buffer.alloc(3840).toString('base64'), sample_rate: 48000, channels: 2 }));
  session.player = { state: { status: AudioPlayerStatus.Idle }, stop() {}, play(r) { resource = r; } };
  assert.equal(await session.testVoice('u'), true);
  assert.deepEqual(calls[0], { route: '/test', data: { conversation: 'g:c', user: 'u' } });
  assert.equal(session.diagnostic('u').received_packets, 0);
  resource.playStream.destroy();
  session.busy = true;
  await assert.rejects(session.testVoice('u'), /текущего ответа/);
  session.busy = false;
  session.ignored.add('u');
  await assert.rejects(session.testVoice('u'), /remember_me/);
  assert.equal(calls.filter(c => c.route === '/test').length, 1);
});

test('private diagnostics distinguish missing audio, input mute, recognition and synthesis failures', () => {
  const base = { same_channel: true, received_packets: 0, duration_ms: 0, stage: 'none' };
  const none = { status: 'none' };
  assert.match(formatDiagnostic('test-version', base, none), /аудио из Discord пока не пришло/);
  assert.match(formatDiagnostic('v', { ...base, ignored: true }, none), /remember_me/);
  assert.match(formatDiagnostic('v', { ...base, server_deaf: true }, none), /deafen/);
  assert.match(formatDiagnostic('v', { ...base, user_muted: true }, none), /микрофон выключен/);
  const failure = formatDiagnostic('v', { ...base, stage: 'error', reason: 'decode_failed' }, none);
  assert.match(failure, /Ошибка при обработке звука/);
  assert.doesNotMatch(failure, /аудио из Discord пока не пришло/);
  assert.match(formatDiagnostic('v', { ...base, stage: 'playback_blocked' }, none), /во время ответа бота/);
  assert.match(formatDiagnostic('v', { ...base, stage: 'echo_blocked' }, none), /остаток звука/);
  const heard = { ...base, received_packets: 30, duration_ms: 1800, stage: 'requesting' };
  assert.match(formatDiagnostic('v', heard, { status: 'recognizing' }), /Распознаю речь/);
  assert.match(formatDiagnostic('v', heard, { status: 'synthesizing', heard: 'How are you?', language: 'en' }), /How are you/);
  assert.match(formatDiagnostic('v', heard, { status: 'failed' }), /завершилась с ошибкой/);
  assert.match(formatDiagnostic('v', heard, { status: 'skipped' }), /речь не распознана/);
  assert.match(formatDiagnostic('v', heard, { status: 'skipped', reason: 'unsupported_language' }), /неподдерживаемый язык/);
  assert.match(formatDiagnostic('v', heard, { status: 'answered', reason: 'rejected_reply' }), /Рассуждение модели отброшено/);
  assert.match(formatDiagnostic('v', heard, { status: 'answered', model_metrics: { load_seconds: 0.1, tokens_per_second: 40 } }), /40 токенов\/с/);
});
