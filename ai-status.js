// Private, per-user troubleshooting. No other participant's transcript is used.
const stages = {
  recognizing: 'Распознаю речь.', thinking: 'Готовлю ответ.', synthesizing: 'Озвучиваю ответ.',
  failed: 'Обработка завершилась с ошибкой. Проверь окно бота.',
};

export function formatDiagnostic(version, input, model) {
  const lines = [`Версия: ${version}`];
  if (input.ignored) lines.push('Прослушивание тебя выключено. Введи /remember_me.');
  else if (!input.same_channel) lines.push('Зайди в тот же голосовой канал, что и бот.');
  else if (input.self_deaf || input.server_deaf) lines.push('У бота выключен звук входящих голосов. Сними заглушение (deafen) в Discord.');
  else if (input.user_muted) lines.push('Твой микрофон выключен в Discord. Включи его и повтори фразу.');
  else if (input.stage === 'error') lines.push('Ошибка при обработке звука. Проверь окно бота и попробуй /talk_test.');
  else if (input.stage === 'playback_blocked') lines.push('Ты начал говорить во время ответа бота. После его ответа сделай паузу и повтори.');
  else if (input.stage === 'echo_blocked') lines.push('Короткая пауза после голоса бота: возможный остаток звука не записывается. Повтори фразу после паузы.');
  else if (input.received_packets === 0) lines.push('Твоё аудио из Discord пока не пришло. Проверь микрофон и Push-to-Talk, затем скажи фразу после ответа бота.');
  else if (input.stage === 'too_short') lines.push('Звук получен, но фрагмент слишком короткий. Скажи целую фразу.');
  else if (input.stage === 'receiving') lines.push('Получаю твой голос. Закончи фразу и сделай паузу.');
  else if (input.stage === 'queued') lines.push('Твоя фраза ждёт завершения текущего ответа.');
  if (input.self_mute || input.server_mute) lines.push('У бота выключен микрофон в Discord — его ответ не будет слышен.');
  if (model.status === 'skipped') lines.push(model.reason === 'unsupported_language'
    ? 'Фрагмент отклонён: распознан неподдерживаемый язык или алфавит. Поддерживаются русский, украинский и английский.'
    : 'В последнем обработанном фрагменте речь не распознана. Проверь уровень микрофона.');
  if (model.reason === 'rejected_reply') lines.push('Рассуждение модели отброшено до озвучки. Прозвучала короткая просьба повторить.');
  if (stages[model.status]) lines.push(stages[model.status]);
  if (model.heard) lines.push(`Последняя расшифровка: «${String(model.heard).slice(0, 550)}»`);
  if (model.answer) lines.push(`Ответ: «${String(model.answer).slice(0, 500)}»`);
  if (model.language) lines.push(`Определённый язык: ${model.language}`);
  if (model.engine) lines.push(`Распознавание: ${model.engine}`);
  if (model.timings) {
    const labels = { recognizing: 'речь', thinking: 'ответ', synthesizing: 'голос' };
    lines.push('Время: ' + Object.entries(labels).filter(([key]) => Number.isFinite(model.timings[key]))
      .map(([key, label]) => `${label} ${model.timings[key]} с`).join(', '));
  }
  if (model.model_metrics) {
    const metrics = model.model_metrics;
    const pieces = [];
    if (Number.isFinite(metrics.load_seconds)) pieces.push(`загрузка ${metrics.load_seconds.toFixed(2)} с`);
    if (Number.isFinite(metrics.prompt_seconds)) pieces.push(`контекст ${metrics.prompt_seconds.toFixed(2)} с`);
    if (Number.isFinite(metrics.tokens_per_second)) pieces.push(`${metrics.tokens_per_second} токенов/с`);
    if (pieces.length) lines.push('Языковая модель: ' + pieces.join(', ') + '.');
  }
  lines.push(`Аудиопакетов: ${input.received_packets}; последний фрагмент: ${input.duration_ms} мс; этап: ${input.stage}.`);
  return lines.join('\n').slice(0, 1900);
}
