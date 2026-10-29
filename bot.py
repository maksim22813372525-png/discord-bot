import asyncio
import logging
import os
import random
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands

from memory import Memory

logging.basicConfig(level=logging.INFO)
path = Path(os.getenv('BOT_DATA_DIR', 'data'))
path.mkdir(parents=True, exist_ok=True)
memory = Memory(path / 'phrases.sqlite3')


class EchoBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=commands.when_mentioned, intents=intents,
                         allowed_mentions=discord.AllowedMentions.none(),
                         proxy=os.getenv('HTTPS_PROXY') or None)
        self.workers = {}

    async def setup_hook(self):
        await self.tree.sync()
        for channel, low, high in memory.channels():
            self.start_worker(channel, low, high)

    def start_worker(self, channel, low, high):
        self.stop_worker(channel)
        self.workers[channel] = asyncio.create_task(self.repeat(channel, low, high))

    def stop_worker(self, channel):
        task = self.workers.pop(channel, None)
        if task:
            task.cancel()

    async def repeat(self, channel_id, low, high):
        await self.wait_until_ready()
        while not self.is_closed():
            await asyncio.sleep(random.uniform(low * 60, high * 60))
            phrase = memory.choose(channel_id)
            if phrase:
                try:
                    channel = self.get_channel(channel_id) or await self.fetch_channel(channel_id)
                    await channel.send(phrase)
                except discord.HTTPException:
                    logging.exception('Could not send a phrase to channel %s', channel_id)

    async def on_message(self, message):
        if message.guild and not message.author.bot and not message.webhook_id:
            memory.remember(message.channel.id, message.author.id, message.content)

    async def close(self):
        tasks = list(self.workers.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await super().close()


bot = EchoBot()


@bot.tree.command(name='echo_on', description='Запам’ятовувати та випадково повторювати фрази в цьому каналі')
@app_commands.guild_only()
@app_commands.default_permissions(manage_guild=True)
@app_commands.checks.has_permissions(manage_guild=True)
async def echo_on(interaction: discord.Interaction, minimum: app_commands.Range[int, 1, 10080] = 30,
                  maximum: app_commands.Range[int, 1, 10080] = 180):
    if maximum < minimum:
        await interaction.response.send_message('Максимум має бути не меншим за мінімум.', ephemeral=True)
        return
    memory.enable(interaction.channel_id, minimum, maximum)
    bot.start_worker(interaction.channel_id, minimum, maximum)
    await interaction.response.send_message(
        f'Увімкнено! Зберігатиму нові текстові повідомлення цього каналу (до 1000 фраз) '
        f'та повторюватиму випадкову фразу кожні {minimum}–{maximum} хв. '
        'Видалити свої фрази: /forget_me. Очистити всі: /echo_clear.')


@bot.tree.command(name='echo_off', description='Вимкнути збереження і повторення в цьому каналі')
@app_commands.guild_only()
@app_commands.default_permissions(manage_guild=True)
@app_commands.checks.has_permissions(manage_guild=True)
async def echo_off(interaction: discord.Interaction):
    memory.disable(interaction.channel_id)
    bot.stop_worker(interaction.channel_id)
    await interaction.response.send_message('Збереження та повторення вимкнено. Фрази можна видалити через /echo_clear.')


@bot.tree.command(name='echo_clear', description='Видалити всі збережені фрази цього каналу')
@app_commands.guild_only()
@app_commands.default_permissions(manage_guild=True)
@app_commands.checks.has_permissions(manage_guild=True)
async def echo_clear(interaction: discord.Interaction):
    memory.forget(interaction.channel_id)
    await interaction.response.send_message('Усі збережені фрази цього каналу видалено.', ephemeral=True)


@bot.tree.command(name='forget_me', description='Видалити мої збережені фрази в цьому каналі')
@app_commands.guild_only()
async def forget_me(interaction: discord.Interaction):
    memory.forget(interaction.channel_id, interaction.user.id)
    await interaction.response.send_message('Твої збережені фрази цього каналу видалено.', ephemeral=True)


@bot.tree.error
async def command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.CheckFailure):
        message = 'Для цієї команди потрібне право «Керування сервером».'
    else:
        logging.error('Command failed', exc_info=(type(error), error, error.__traceback__))
        message = 'Не вдалося виконати команду. Перевір журнал бота.'
    if interaction.response.is_done():
        await interaction.followup.send(message, ephemeral=True)
    else:
        await interaction.response.send_message(message, ephemeral=True)


if __name__ == '__main__':
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        raise SystemExit('Задай DISCORD_TOKEN у змінних середовища перед запуском.')
    bot.run(token)
