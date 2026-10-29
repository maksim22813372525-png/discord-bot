import { Readable } from 'node:stream';
import {
  AudioPlayerStatus, EndBehaviorType, NoSubscriberBehavior, StreamType,
  VoiceConnectionStatus, createAudioPlayer, createAudioResource, entersState, joinVoiceChannel,
} from '@discordjs/voice';
import { Phrase } from './clips.js';

export class VoiceSession {
  constructor({ channel, store, minimum, maximum, onClose, onError }) {
    Object.assign(this, { channel, store, minimum, maximum, onClose, onError });
    this.guild = channel.guild;
    this.captures = new Map();
    this.closed = false;
    this.player = createAudioPlayer({ behaviors: { noSubscriber: NoSubscriberBehavior.Stop } });
    this.player.on('error', error => this.fail(error));
    this.onSpeaking = user => this.capture(user);
  }

  async start() {
    this.connection = joinVoiceChannel({
      channelId: this.channel.id, guildId: this.guild.id,
      adapterCreator: this.guild.voiceAdapterCreator, selfDeaf: false, selfMute: false,
    });
    this.connection.on('error', error => this.fail(error));
    this.connection.on(VoiceConnectionStatus.Disconnected, () => {
      this.fail(new Error('Голосове з’єднання втрачено. Виконай /voice_on ще раз.'));
    });
    this.connection.on(VoiceConnectionStatus.Destroyed, () => this.stop());
    await entersState(this.connection, VoiceConnectionStatus.Ready, 25_000);
    if (this.closed) throw new Error('Запуск скасовано.');
    this.connection.subscribe(this.player);
    this.connection.receiver.speaking.on('start', this.onSpeaking);
    this.schedule();
  }

  capture(user) {
    const member = this.guild.members.cache.get(user);
    if (this.closed || this.player.state.status !== AudioPlayerStatus.Idle
      || this.captures.has(user) || !member || member.user.bot
      || member.voice.channelId !== this.channel.id || this.store.isIgnored(this.guild.id, user)) return;
    const stream = this.connection.receiver.subscribe(user, {
      end: { behavior: EndBehaviorType.AfterSilence, duration: 800 },
    });
    const phrase = new Phrase(performance.now());
    const capture = { stream, timer: null, done: false };
    this.captures.set(user, capture);
    const finish = (save = true) => {
      if (capture.done) return;
      capture.done = true;
      clearTimeout(capture.timer);
      this.captures.delete(user);
      stream.destroy();
      if (save && !this.closed && !this.store.isIgnored(this.guild.id, user)) {
        try { this.store.add(this.guild.id, this.channel.id, user, phrase.finish()); }
        catch (error) { this.fail(error); }
      }
    };
    capture.cancel = () => finish(false);
    stream.on('data', packet => phrase.add(packet, performance.now()));
    stream.once('end', () => finish());
    stream.once('close', () => finish(false));
    stream.once('error', error => { finish(false); this.fail(error); });
    capture.timer = setTimeout(() => finish(), 8_000);
  }

  discard(user) {
    for (const [id, capture] of this.captures) {
      if (!user || user === id) capture.cancel();
    }
    // A currently playing clip must stop when recordings are cleared.
    this.player.stop(true);
  }

  schedule() {
    if (this.closed) return;
    const minutes = this.minimum + Math.random() * (this.maximum - this.minimum);
    this.timer = setTimeout(() => {
      try {
        if (!this.closed && this.connection.state.status === VoiceConnectionStatus.Ready
          && this.player.state.status === AudioPlayerStatus.Idle
          && this.channel.members.some(member => !member.user.bot)) {
          const clip = this.store.choose(this.guild.id, this.channel.id);
          if (clip) {
            this.discard(); // Don't capture loudspeaker feedback during playback.
            this.player.play(createAudioResource(Readable.from(clip.packets), { inputType: StreamType.Opus }));
          }
        }
        this.schedule();
      } catch (error) { this.fail(error); }
    }, minutes * 60_000);
  }

  fail(error) {
    if (this.closed) return;
    this.stop();
    this.onError(error);
  }

  stop() {
    if (this.closed) return;
    this.closed = true;
    clearTimeout(this.timer);
    this.discard();
    this.connection?.receiver.speaking.off('start', this.onSpeaking);
    if (this.connection && this.connection.state.status !== VoiceConnectionStatus.Destroyed) this.connection.destroy();
    this.onClose();
  }
}
