import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  ChannelType, Client, Events, GatewayIntentBits, PermissionFlagsBits, SlashCommandBuilder,
} from 'discord.js';
import { ClipStore } from './clips.js';
import { VoiceSession } from './voice-session.js';

const admin = PermissionFlagsBits.ManageGuild;
const command = (name, description, restricted = false) => {
  const c = new SlashCommandBuilder().setName(name).setDescription(description).setDMPermission(false);
  if (restricted) c.setDefaultMemberPermissions(admin);
  return c;
};

export function commands() {
  return [
    command('voice_on', 'Зайти у твій войс, записувати та випадково повторювати короткі фрази', true)
      .addIntegerOption(o => o.setName('minimum').setDescription('Мінімальний інтервал у хвилинах (типово 1)').setMinValue(1).setMaxValue(1440))
      .addIntegerOption(o => o.setName('maximum').setDescription('Максимальний інтервал у хвилинах (типово 5)').setMinValue(1).setMaxValue(1440)),
    command('voice_off', 'Зупинити запис і вийти з войсу', true),
    command('voice_clear', 'Видалити всі голосові записи цього сервера', true),
    command('forget_me', 'Видалити мої записи та більше не записувати мене на цьому сервері'),
    command('remember_me', 'Знову дозволити записувати мене на цьому сервері'),
  ];
}

export async function main() {
  const token = process.env.DISCORD_TOKEN;
  if (!token) throw new Error('Задай DISCORD_TOKEN або запусти start.bat.');
  const directory = process.env.BOT_DATA_DIR || path.join(path.dirname(fileURLToPath(import.meta.url)), 'data');
  const store = new ClipStore(directory);
  const client = new Client({
    intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildVoiceStates],
    allowedMentions: { parse: [] },
  });
  const sessions = new Map();
  client.once(Events.ClientReady, async ready => {
    try {
      await ready.application.commands.set(commands().map(c => c.toJSON()));
      console.log(`Онлайн: ${ready.user.tag}. Зайди у войс та введи /voice_on у текстовому каналі.`);
    } catch (error) {
      console.error('Не вдалося зареєструвати команди:', error.message);
      client.destroy();
      process.exitCode = 1;
    }
  });

  client.on(Events.InteractionCreate, async interaction => {
    if (!interaction.isChatInputCommand() || !interaction.inGuild()) return;
    const guildId = interaction.guildId;
    const name = interaction.commandName;
    try {
      if (!commands().some(c => c.name === name)) {
        await interaction.reply({ content: 'Це стара команда. Онови Discord і вибери /voice_on.', ephemeral: true });
        return;
      }
      if (['voice_on', 'voice_off', 'voice_clear'].includes(name) && !interaction.memberPermissions?.has(admin)) {
        await interaction.reply({ content: 'Потрібне право «Керування сервером».', ephemeral: true });
        return;
      }
      const session = sessions.get(guildId);
      if (name === 'voice_on') {
        const member = await interaction.guild.members.fetch(interaction.user.id);
        const channel = member.voice.channel;
        const minimum = interaction.options.getInteger('minimum') ?? 1;
        const maximum = interaction.options.getInteger('maximum') ?? 5;
        if (!channel || channel.type !== ChannelType.GuildVoice) {
          await interaction.reply({ content: 'Спочатку зайди у звичайний голосовий канал сервера.', ephemeral: true });
          return;
        }
        if (maximum < minimum || session) {
          await interaction.reply({ content: session ? 'Я вже у войсі. Спочатку виконай /voice_off.' : 'Максимум має бути не меншим за мінімум.', ephemeral: true });
          return;
        }
        const me = await interaction.guild.members.fetchMe();
        if (!channel.permissionsFor(me)?.has([PermissionFlagsBits.ViewChannel, PermissionFlagsBits.Connect, PermissionFlagsBits.Speak])) {
          await interaction.reply({ content: 'Дай боту права «Дивитися канали», «Приєднуватися» та «Говорити» в цьому войсі.', ephemeral: true });
          return;
        }
        await interaction.deferReply();
        // Public notice must succeed before audio reception starts.
        await interaction.editReply(`🎙️ Заходжу в <#${channel.id}>: записуватиму фрази до 8 секунд і повторюватиму їхні оригінальні записи кожні ${minimum}–${maximum} хв. До 100 записів на сервер. /forget_me — видалити свої записи й вимкнути записування себе. /voice_off — зупинити.`);
        if (sessions.has(guildId)) {
          await interaction.editReply('Бота вже запускає інший учасник. Спочатку виконай /voice_off.');
          return;
        }
        const current = new VoiceSession({ channel, store, minimum, maximum,
          onClose: () => { if (sessions.get(guildId) === current) sessions.delete(guildId); },
          onError: error => {
            console.error('Помилка войсу:', error.message);
            interaction.channel?.send('Запис зупинено через помилку войсу. Перевір вікно бота й виконай /voice_on ще раз.').catch(console.error);
          },
        });
        sessions.set(guildId, current);
        try { await current.start(); }
        catch (error) { current.stop(); throw error; }
      } else if (name === 'voice_off') {
        session?.stop();
        await interaction.reply('Запис і повторення зупинено, бот вийшов із войсу. Збережені записи можна видалити через /voice_clear.');
      } else if (name === 'voice_clear') {
        session?.discard();
        store.clear(guildId);
        await interaction.reply({ content: 'Усі голосові записи сервера видалено. Якщо бот у войсі, нові фрази продовжать записуватися.', ephemeral: true });
      } else if (name === 'forget_me') {
        session?.discard(interaction.user.id);
        store.ignore(guildId, interaction.user.id);
        await interaction.reply({ content: 'Твої записи цього сервера видалено. Більше не записуватиму тебе, доки не виконаєш /remember_me.', ephemeral: true });
      } else if (name === 'remember_me') {
        store.ignore(guildId, interaction.user.id, false);
        await interaction.reply({ content: 'Записування твого голосу знову дозволено, коли бот у войсі.', ephemeral: true });
      }
    } catch (error) {
      console.error('Помилка команди:', error.message);
      const content = 'Не вдалося виконати команду. Перевір права бота, підключення до інтернету та помилку у вікні запуску.';
      if (interaction.deferred || interaction.replied) await interaction.editReply(content).catch(console.error);
      else await interaction.reply({ content, ephemeral: true }).catch(console.error);
    }
  });

  client.on(Events.VoiceStateUpdate, (before, after) => {
    const session = sessions.get(after.guild.id);
    if (after.id === client.user?.id && session && after.channelId !== session.channel.id) {
      // An administrator moving the bot must not move recordings to another room.
      session.stop();
    }
  });

  const stop = () => { for (const session of sessions.values()) session.stop(); client.destroy(); };
  process.once('SIGINT', stop);
  process.once('SIGTERM', stop);
  client.on(Events.Error, error => console.error('Discord:', error.message));
  await client.login(token);
  return { client, stop };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => { console.error('Не вдалося запустити бота:', error.message); process.exitCode = 1; });
}
