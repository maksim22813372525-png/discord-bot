import { Readable } from 'node:stream';
import OpusScript from 'opusscript';
import {
  AudioPlayerStatus, EndBehaviorType, NoSubscriberBehavior, StreamType,
  VoiceConnectionStatus, createAudioPlayer, createAudioResource, entersState, joinVoiceChannel,
} from '@discordjs/voice';
import { SILENCE } from './clips.js';

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
  constructor({ channel, worker, onClose, onError, ignored = new Set() }) {
    Object.assign(this, { channel, worker, onClose, onError, ignored });
    this.guild = channel.guild;
    this.key = `${this.guild.id}:${channel.id}`;
    this.captures = new Map();
    this.closed = false;
    this.busy = false;
    this.epoch = 0;
    this.player = createAudioPlayer({ behaviors: { noSubscriber: NoSubscriberBehavior.Stop } });
    this.player.on('error', error => this.fail(error));
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

  capture(user) {
    const member = this.guild.members.cache.get(user);
    if (this.closed || this.busy || this.player.state.status !== AudioPlayerStatus.Idle
      || this.captures.has(user) || !member || member.user.bot || this.ignored.has(user)
      || member.voice.channelId !== this.channel.id) return;
    const stream = this.connection.receiver.subscribe(user, {
      end: { behavior: EndBehaviorType.AfterSilence, duration: 900 },
    });
    const decoder = new OpusScript(48000, 2, OpusScript.Application.VOIP);
    const chunks = [];
    let bytes = 0;
    let speechFrames = 0;
    let last = performance.now();
    const capture = { done: false, timer: null, stream };
    const finish = save => {
      if (capture.done) return;
      capture.done = true;
      clearTimeout(capture.timer);
      this.captures.delete(user);
      stream.destroy();
      decoder.delete();
      if (save && !this.closed && !this.ignored.has(user) && speechFrames >= 15) {
        this.respond(user, Buffer.concat(chunks));
      }
    };
    capture.cancel = () => finish(false);
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
        if (!packet.equals(SILENCE)) speechFrames++;
        if (bytes >= 48_000 * 2 * 2 * 20) finish(true);
      } catch (error) { finish(false); this.onError(error); }
    });
    stream.once('end', () => finish(true));
    stream.once('close', () => finish(false));
    stream.once('error', error => { finish(false); this.onError(error); });
    capture.timer = setTimeout(() => finish(true), 20_000);
  }

  async respond(user, pcm) {
    if (this.closed || this.busy || this.ignored.has(user)) return;
    this.busy = true;
    const epoch = this.epoch;
    const controller = this.request = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 300_000);
    // Discard overlapping utterances rather than queuing old conversations.
    for (const capture of this.captures.values()) capture.cancel();
    try {
      const response = await this.worker.post('/reply', {
        conversation: this.key, user, audio_pcm: pcm.subarray(0, 3_840_000).toString('base64'),
      }, controller.signal);
      if (this.closed || epoch !== this.epoch || this.ignored.has(user) || response.skipped) return;
      if (response.sample_rate !== 48000 || response.channels !== 2 || typeof response.audio_pcm !== 'string') {
        throw new Error('Некоректна відповідь голосової моделі.');
      }
      const audio = Buffer.from(response.audio_pcm, 'base64');
      if (!audio.length || audio.length % 4 || audio.length > 7_680_000) throw new Error('Некоректний PCM.');
      this.player.play(createAudioResource(Readable.from([audio]), { inputType: StreamType.Raw }));
    } catch (error) {
      if (error.name !== 'AbortError' && !this.closed) this.onError(error);
    } finally {
      clearTimeout(timeout);
      if (this.request === controller) { this.request = null; this.busy = false; }
    }
  }

  async forget(user) {
    this.epoch++;
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
    this.request?.abort();
    for (const capture of this.captures.values()) capture.cancel();
    this.player.stop(true);
    this.connection?.receiver.speaking.off('start', this.onSpeaking);
    if (this.connection && this.connection.state.status !== VoiceConnectionStatus.Destroyed) this.connection.destroy();
    this.worker.post('/forget', { conversation: this.key }).catch(() => {});
    this.onClose();
  }
}
