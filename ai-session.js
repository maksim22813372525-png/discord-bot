import { Readable } from 'node:stream';
import OpusScript from 'opusscript';
import {
  AudioPlayerStatus, EndBehaviorType, NoSubscriberBehavior, StreamType,
  VoiceConnectionStatus, createAudioPlayer, createAudioResource, entersState, joinVoiceChannel,
} from '@discordjs/voice';
import { SILENCE } from './clips.js';

export const PLAYBACK_TAIL_MS = 500;

export class LocalWorker {
  constructor(url, token, fetcher = fetch) {
    const parsed = new URL(url);
    if (parsed.protocol !== 'http:' || !['127.0.0.1', 'localhost', '[::1]'].includes(parsed.hostname)) {
      throw new Error('Голосова служба має працювати локально.');
    }
    this.url = url.replace(/\/$/, '');
    this.token = token;
    this.fetch = fetcher;
  }

  async post(route, data, signal) {
    const response = await this.fetch(this.url + route, {
      method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${this.token}` },
      body: JSON.stringify(data), signal,
    });
    if (!response.ok) throw new Error(`Локальна модель: HTTP ${response.status}. Перевір вікно запуску.`);
    return response.json();
  }
}

export class ConversationSession {
  constructor({ channel, worker, onClose, onError, ignored = new Set(), now = () => performance.now() }) {
    Object.assign(this, { channel, worker, onClose, onError, ignored, now });
    this.guild = channel.guild;
    this.key = `${this.guild.id}:${channel.id}`;
    this.captures = new Map();
    this.closed = false;
    this.busy = false;
    this.pending = null;
    this.receiveDiagnostics = new Map();
    this.playbackUser = null;
    this.captureAfter = 0;
    this.epoch = 0;
    this.player = createAudioPlayer({ behaviors: { noSubscriber: NoSubscriberBehavior.Stop } });
    this.player.on('error', error => this.fail(error));
    this.player.on(AudioPlayerStatus.Idle, () => {
      if (this.playbackUser) {
        const user = this.playbackUser;
        this.playbackUser = null;
        // A participant's microphone can deliver the loudspeaker tail after
        // Discord has finished playback. Do not begin a new recording yet.
        this.captureAfter = this.now() + PLAYBACK_TAIL_MS;
        if (this.receiveDiagnostics.get(user)?.stage === 'playing') this.note(user, 'completed');
      }
      this.drainPending();
    });
    this.onSpeaking = user => this.capture(user);
  }

  async start() {
    this.connection = joinVoiceChannel({ channelId: this.channel.id, guildId: this.guild.id,
      adapterCreator: this.guild.voiceAdapterCreator, selfDeaf: false, selfMute: false });
    this.connection.on('error', error => this.fail(error));
    this.connection.on(VoiceConnectionStatus.Disconnected, () => this.fail(new Error('Войс від’єднано. Виконай /talk_on знову.')));
    this.connection.on(VoiceConnectionStatus.Destroyed, () => this.stop());
    await entersState(this.connection, VoiceConnectionStatus.Ready, 25_000);
    if (this.closed) throw new Error('Запуск скасовано.');
    this.connection.subscribe(this.player);
    this.connection.receiver.speaking.on('start', this.onSpeaking);
  }

  canListen(user) {
    const member = this.guild.members.cache.get(user);
    return !this.closed && member && !member.user.bot && !this.ignored.has(user)
      && member.voice.channelId === this.channel.id;
  }

  note(user, stage, values = {}, epoch = this.epoch) {
    if (this.closed || epoch !== this.epoch) return;
    const previous = this.receiveDiagnostics.get(user) || {
      received_packets: 0, duration_ms: 0, capture_count: 0,
    };
    if (!Object.hasOwn(values, 'reason')) delete previous.reason;
    this.receiveDiagnostics.delete(user);
    this.receiveDiagnostics.set(user, { ...previous, ...values, stage });
    // Counters contain no audio/text, and departed participants cannot grow
    // the process state without a bound.
    if (this.receiveDiagnostics.size > 100) this.receiveDiagnostics.delete(this.receiveDiagnostics.keys().next().value);
  }

  diagnostic(user) {
    const member = this.guild.members.cache.get(user);
    const ownVoice = this.guild.members.me?.voice;
    return {
      stage: 'none', received_packets: 0, duration_ms: 0, capture_count: 0,
      ...this.receiveDiagnostics.get(user),
      connection: this.connection?.state.status || 'not_connected',
      player: this.player.state.status,
      busy: this.busy,
      same_channel: member?.voice.channelId === this.channel.id,
      member_known: Boolean(member),
      ignored: this.ignored.has(user),
      self_deaf: Boolean(ownVoice?.selfDeaf),
      server_deaf: Boolean(ownVoice?.serverDeaf),
      self_mute: Boolean(ownVoice?.selfMute),
      server_mute: Boolean(ownVoice?.serverMute),
      user_muted: Boolean(member?.voice.selfMute || member?.voice.serverMute),
      cooldown_ms: Math.max(0, Math.ceil(this.captureAfter - this.now())),
    };
  }

  capture(user) {
    // Start events are edge-triggered: dropping one while inference is busy
    // loses the whole ongoing utterance, even when inference later finishes.
    if (!this.canListen(user) || this.captures.has(user)) return;
    if (this.player.state.status !== AudioPlayerStatus.Idle) {
      this.note(user, 'playback_blocked');
      return;
    }
    if (this.now() < this.captureAfter) {
      // Suppress this whole start event rather than re-subscribing mid-tail.
      // A fresh utterance after the guard will create a new speaking start.
      this.note(user, 'echo_blocked');
      return;
    }
    const stream = this.connection.receiver.subscribe(user, {
      end: { behavior: EndBehaviorType.AfterSilence, duration: 900 },
    });
    const decoder = new OpusScript(48000, 2, OpusScript.Application.VOIP);
    const chunks = [];
    let bytes = 0;
    let speechSamples = 0;
    let decodedSamples = 0;
    let last = performance.now();
    const epoch = this.epoch;
    this.note(user, 'receiving', {
      duration_ms: 0, capture_count: (this.receiveDiagnostics.get(user)?.capture_count || 0) + 1,
    }, epoch);
    const capture = { done: false, timer: null, stream };
    const finish = save => {
      if (capture.done) return;
      capture.done = true;
      clearTimeout(capture.timer);
      this.captures.delete(user);
      stream.destroy();
      decoder.delete();
      if (save && !this.closed && !this.ignored.has(user) && speechSamples >= 48_000 * 0.2) {
        this.note(user, 'captured', { duration_ms: Math.round(decodedSamples / 48) }, epoch);
        this.respond(user, Buffer.concat(chunks));
      } else if (save) {
        this.note(user, 'too_short', { duration_ms: Math.round(decodedSamples / 48) }, epoch);
      }
    };
    capture.cancel = () => finish(false);
    capture.finish = () => finish(true);
    this.captures.set(user, capture);
    stream.on('data', packet => {
      if (capture.done) return;
      try {
        const now = performance.now();
        const gap = Math.min(800, now - last);
        if (gap > 120) {
          const padding = Buffer.alloc(Math.floor((gap - 20) / 20) * 3840);
          chunks.push(padding); bytes += padding.length;
        }
        last = now;
        const pcm = decoder.decode(packet);
        chunks.push(Buffer.from(pcm)); bytes += pcm.length;
        const samples = pcm.length / 4;
        decodedSamples += samples;
        if (!packet.equals(SILENCE)) speechSamples += samples;
        this.note(user, 'receiving', {
          duration_ms: Math.round(decodedSamples / 48),
          received_packets: (this.receiveDiagnostics.get(user)?.received_packets || 0) + 1,
        }, epoch);
        if (bytes >= 48_000 * 2 * 2 * 20) finish(true);
      } catch (error) { finish(false); this.note(user, 'error', { reason: 'decode_failed' }, epoch); this.onError(error); }
    });
    stream.once('end', () => finish(true));
    stream.once('close', () => finish(false));
    stream.once('error', error => { finish(false); this.note(user, 'error', { reason: 'receive_failed' }, epoch); this.onError(error); });
    capture.timer = setTimeout(() => finish(true), 20_000);
  }

  async testVoice(user) {
    if (!this.canListen(user)) throw new Error('Зайди в голосовой канал бота; /remember_me включает прослушивание тебя.');
    if (this.busy || this.player.state.status !== AudioPlayerStatus.Idle) {
      throw new Error('Дождись текущего ответа, затем повтори /talk_test.');
    }
    return this.respond(user, null, true);
  }

  async respond(user, pcm, testVoice = false) {
    if (!this.canListen(user)) return;
    if (this.busy || this.player.state.status !== AudioPlayerStatus.Idle) {
      if (testVoice) return;
      // Keep just the latest complete utterance, not a backlog of old speech.
      if (this.pending && this.pending.user !== user) this.note(this.pending.user, 'skipped', { reason: 'replaced_pending' });
      this.pending = { user, pcm: pcm.subarray(0, 3_840_000), epoch: this.epoch };
      this.note(user, 'queued');
      return;
    }
    this.busy = true;
    const epoch = this.epoch;
    this.note(user, 'requesting');
    const controller = this.request = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 300_000);
    try {
      const response = await this.worker.post(testVoice ? '/test' : '/reply', {
        conversation: this.key, user,
        ...(testVoice ? {} : { audio_pcm: pcm.subarray(0, 3_840_000).toString('base64') }),
      }, controller.signal);
      if (this.closed || epoch !== this.epoch || this.ignored.has(user)) return;
      if (response.skipped) {
        this.note(user, 'skipped', { reason: response.busy ? 'worker_busy' : 'not_recognized' }, epoch);
        return false;
      }
      if (response.sample_rate !== 48000 || response.channels !== 2 || typeof response.audio_pcm !== 'string') {
        throw new Error('Некоректна відповідь голосової моделі.');
      }
      const audio = Buffer.from(response.audio_pcm, 'base64');
      if (!audio.length || audio.length % 4 || audio.length > 7_680_000) throw new Error('Некоректний PCM.');
      // End any recording started while thinking before our own audio can
      // reach a participant's microphone. Its captured prefix can be pending.
      for (const capture of this.captures.values()) capture.finish();
      this.playbackUser = user;
      this.note(user, 'playing', {}, epoch);
      this.player.play(createAudioResource(Readable.from([audio]), { inputType: StreamType.Raw }));
      return true;
    } catch (error) {
      this.note(user, 'error', { reason: error.name === 'AbortError' ? 'request_timeout' : 'worker_failed' }, epoch);
      if (error.name !== 'AbortError' && !this.closed) this.onError(error);
    } finally {
      clearTimeout(timeout);
      if (this.request === controller) {
        this.request = null;
        this.busy = false;
        this.drainPending();
      }
    }
  }

  drainPending() {
    if (this.closed || this.busy || this.player.state.status !== AudioPlayerStatus.Idle || !this.pending) return;
    const pending = this.pending;
    this.pending = null;
    if (pending.epoch === this.epoch && this.canListen(pending.user)) {
      void this.respond(pending.user, pending.pcm);
    }
  }

  async forget(user) {
    this.epoch++;
    this.pending = null;
    this.receiveDiagnostics.clear();
    if (this.playbackUser) this.captureAfter = this.now() + PLAYBACK_TAIL_MS;
    this.playbackUser = null;
    this.request?.abort();
    for (const capture of this.captures.values()) capture.cancel();
    this.player.stop(true);
    if (user) this.ignored.add(user);
    await this.worker.post('/forget', { conversation: this.key, user });
  }

  fail(error) {
    if (this.closed) return;
    this.stop();
    this.onError(error);
  }

  stop() {
    if (this.closed) return;
    this.closed = true;
    this.epoch++;
    this.pending = null;
    this.receiveDiagnostics.clear();
    this.playbackUser = null;
    this.captureAfter = 0;
    this.request?.abort();
    for (const capture of this.captures.values()) capture.cancel();
    this.player.stop(true);
    this.connection?.receiver.speaking.off('start', this.onSpeaking);
    if (this.connection && this.connection.state.status !== VoiceConnectionStatus.Destroyed) this.connection.destroy();
    this.worker.post('/forget', { conversation: this.key }).catch(() => {});
    this.onClose();
  }
}
