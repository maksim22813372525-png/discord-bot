import { Client, Events, GatewayIntentBits, PermissionFlagsBits, SlashCommandBuilder, ChannelType } from 'discord.js';
import { ClipStore } from './clips.js';
import { ConversationSession, LocalWorker } from './ai-session.js';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const restrict = PermissionFlagsBits.ManageGuild;
export const definitions = [
  ['talk_on', 'Зайти у твій войс і ввімкнути локальну AI-співрозмовницю', true],
  ['talk_off', 'Зупинити розмову та вийти з войсу', true],
  ['talk_clear', 'Очистити пам’ять поточної розмови', true],
  ['talk_heard', 'Приватно показати, як бот розпізнав твою останню репліку'],
  ['forget_me', 'Не слухати мене та видалити мої репліки з пам’яті'],
  ['remember_me', 'Знову слухати мої репліки'],
].map(([name, description, admin]) => {
  const c = new SlashCommandBuilder().setName(name).setDescription(description).setDMPermission(false);
  return (admin ? c.setDefaultMemberPermissions(restrict) : c).toJSON();
});

export async function main() {
  const token = process.env.DISCORD_TOKEN;
  if (!token || !process.env.AI_WORKER_TOKEN || !process.env.AI_WORKER_URL) throw new Error('Запусти start_ai.bat.');
  const worker = new LocalWorker(process.env.AI_WORKER_URL, process.env.AI_WORKER_TOKEN);
  const store = new ClipStore(process.env.BOT_DATA_DIR || path.join(path.dirname(fileURLToPath(import.meta.url)), 'data'));
  const sessions = new Map();
  const client = new Client({ intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildVoiceStates], allowedMentions: { parse: [] } });
  client.once(Events.ClientReady, async ready => {
    try {
      await ready.application.commands.set(definitions);
      console.log(`Онлайн: ${ready.user.tag}. Зайди у войс та виконай /talk_on.`);
    } catch (error) { console.error('Команди:', error.message); client.destroy(); process.exitCode = 1; }
  });
  client.on(Events.InteractionCreate, async i => {
    if (!i.isChatInputCommand() || !i.inGuild()) return;
    try {
      const name = i.commandName;
      if (!definitions.some(c => c.name === name)) {
        await i.reply({ content: 'Обери /talk_on. Старі команди ще можуть бути у кеші Discord.', ephemeral: true }); return;
      }
      if (['talk_on', 'talk_off', 'talk_clear'].includes(name) && !i.memberPermissions?.has(restrict)) {
        await i.reply({ content: 'Потрібне право «Керування сервером».', ephemeral: true }); return;
      }
      await i.deferReply({ ephemeral: ['forget_me', 'remember_me', 'talk_heard'].includes(name) });
      let session = sessions.get(i.guildId);
      if (name === 'talk_on') {
        if (session) { await i.editReply('Розмова вже ввімкнена. Спочатку /talk_off.'); return; }
        const member = await i.guild.members.fetch(i.user.id);
        const channel = member.voice.channel;
        if (!channel || channel.type !== ChannelType.GuildVoice) { await i.editReply('Спочатку зайди у звичайний голосовий канал.'); return; }
        const me = await i.guild.members.fetchMe();
        if (!channel.permissionsFor(me)?.has([PermissionFlagsBits.ViewChannel, PermissionFlagsBits.Connect, PermissionFlagsBits.Speak])) {
          await i.editReply('Боту потрібні права «Дивитися канали», «Приєднуватися» та «Говорити».'); return;
        }
        await i.editReply(`🎙️ У <#${channel.id}> починається розмова з локальною AI-співрозмовницею російською. Репліки розпізнаються на ПК власника; історія живе лише в пам’яті процесу. /forget_me — не слухати тебе, /talk_off — зупинити. Говоріть по черзі; відповідь буде після паузи.`);
        if (sessions.has(i.guildId)) { await i.editReply('Розмову вже запустив інший учасник.'); return; }
        const ignored = new Set(store.data.ignored.filter(s => s.startsWith(i.guildId + ':')).map(s => s.split(':')[1]));
        session = new ConversationSession({ channel, worker, ignored,
          onClose: () => { if (sessions.get(i.guildId) === session) sessions.delete(i.guildId); },
          onError: error => {
            console.error('Розмова:', error.message);
            i.channel?.send('Не вдалося обробити голосову репліку. Перевір вікно бота та локальні моделі.').catch(() => {});
          },
        });
        sessions.set(i.guildId, session);
        try { await session.start(); }
        catch (error) { session.stop(); throw error; }
      } else if (name === 'talk_off') {
        session?.stop();
        await i.editReply('Розмову зупинено.');
      } else if (name === 'talk_clear') {
        await session?.forget();
        await i.editReply('Пам’ять поточної розмови очищено.');
      } else if (name === 'talk_heard') {
        if (!session) { await i.editReply('Сначала включи разговор через /talk_on.'); return; }
        const last = await worker.post('/last', { conversation: session.key, user: i.user.id });
        if (last.status === 'none') await i.editReply('Твоих распознанных реплик пока нет. Скажи фразу после запуска /talk_on.');
        else if (last.status === 'skipped') await i.editReply('В последнем фрагменте речь не распознана. Попробуй сказать фразу чётче и проверить уровень микрофона.');
        else await i.editReply(`Услышала: «${String(last.heard || '').slice(0, 700)}»\n${last.answer ? `Ответ: «${String(last.answer).slice(0, 600)}»` : 'Ответ ещё готовится.'}`);
      } else if (name === 'forget_me') {
        store.ignore(i.guildId, i.user.id);
        if (session) await session.forget(i.user.id);
        await i.editReply('Спільну пам’ять розмови очищено, включно зі згадками твоїх реплік. Не слухатиму тебе до /remember_me.');
      } else if (name === 'remember_me') {
        store.ignore(i.guildId, i.user.id, false);
        await worker.post('/allow', { conversation: i.guildId + ':', user: i.user.id });
        if (session) {
          session.ignored.delete(i.user.id);
        }
        await i.editReply('Знову слухатиму твої репліки, коли розмова ввімкнена.');
      }
    } catch (error) {
      console.error('Команда:', error.message);
      const content = 'Помилка запуску або локальної моделі. Подивись повідомлення у вікні бота.';
      if (i.deferred || i.replied) await i.editReply(content).catch(() => {});
      else await i.reply({ content, ephemeral: true }).catch(() => {});
    }
  });
  client.on(Events.VoiceStateUpdate, (before, after) => {
    const session = sessions.get(after.guild.id);
    if (after.id === client.user?.id && session && after.channelId !== session.channel.id) session.stop();
  });
  const stop = () => { for (const s of sessions.values()) s.stop(); client.destroy(); };
  process.once('SIGINT', stop); process.once('SIGTERM', stop);
  client.on(Events.Error, error => console.error('Discord:', error.message));
  await client.login(token);
  return { client, stop };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => { console.error(error.message); process.exitCode = 1; });
}
