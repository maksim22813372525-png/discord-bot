import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

export const SILENCE = Buffer.from([0xf8, 0xff, 0xfe]);
export const MAX_FRAMES = 400; // Discord sends 20 ms Opus frames: 8 seconds.
const MAX_BYTES = 512_000;

// Preserve short pauses; never decode, synthesize, or change the speaker's voice.
export class Phrase {
  constructor(now) {
    this.started = now;
    this.frames = [];
    this.bytes = 0;
    this.voiced = 0;
  }

  add(packet, now) {
    if (!Buffer.isBuffer(packet) || !packet.length || packet.length > 4096) return;
    const target = Math.min(MAX_FRAMES, Math.floor((now - this.started) / 20));
    // Ignore small network jitter; fill genuine pauses with Opus silence.
    if (target - this.frames.length > 4) {
      while (this.frames.length < target) this.frames.push(SILENCE);
    }
    if (this.frames.length >= MAX_FRAMES || this.bytes + packet.length > MAX_BYTES) return;
    this.frames.push(Buffer.from(packet));
    this.bytes += packet.length;
    if (!packet.equals(SILENCE)) this.voiced++;
  }

  finish() {
    while (this.frames.at(-1)?.equals(SILENCE)) this.frames.pop();
    return this.voiced >= 15 ? this.frames : []; // Ignore clicks and tiny fragments.
  }
}

export class ClipStore {
  constructor(directory) {
    fs.mkdirSync(directory, { recursive: true });
    this.file = path.join(directory, 'voice-clips.json');
    this.data = fs.existsSync(this.file)
      ? JSON.parse(fs.readFileSync(this.file, 'utf8'))
      : { version: 1, clips: [], ignored: [] };
    if (this.data.version !== 1 || !Array.isArray(this.data.clips) || !Array.isArray(this.data.ignored)) {
      throw new Error('Невідомий формат voice-clips.json. Збережи копію перед відновленням бази.');
    }
  }

  save() {
    const temp = `${this.file}.tmp`;
    fs.writeFileSync(temp, JSON.stringify(this.data), { mode: 0o600 });
    fs.renameSync(temp, this.file);
  }

  isIgnored(guild, user) {
    return this.data.ignored.includes(`${guild}:${user}`);
  }

  ignore(guild, user, ignored = true) {
    const key = `${guild}:${user}`;
    this.data.ignored = this.data.ignored.filter(x => x !== key);
    if (ignored) this.data.ignored.push(key);
    if (ignored) this.clear(guild, user);
    else this.save();
  }

  add(guild, channel, user, packets) {
    if (this.isIgnored(guild, user) || packets.length < 15 || packets.length > MAX_FRAMES) return;
    if (packets.reduce((sum, p) => sum + p.length, 0) > MAX_BYTES) return;
    this.data.clips.push({ id: crypto.randomUUID(), guild, channel, user,
      packets: packets.map(p => p.toString('base64')) });
    const guildClips = this.data.clips.filter(c => c.guild === guild);
    const removed = new Set(guildClips.slice(0, -100).map(c => c.id));
    this.data.clips = this.data.clips.filter(c => !removed.has(c.id));
    this.save();
  }

  choose(guild, channel, random = Math.random) {
    const candidates = this.data.clips.filter(c => c.guild === guild && c.channel === channel
      && !this.isIgnored(guild, c.user));
    const clip = candidates[Math.floor(random() * candidates.length)];
    return clip ? { ...clip, packets: clip.packets.map(p => Buffer.from(p, 'base64')) } : null;
  }

  clear(guild, user) {
    this.data.clips = this.data.clips.filter(c => c.guild !== guild || (user && c.user !== user));
    this.save();
  }
}
