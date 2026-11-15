import './style.css';
import { wavBytes } from './dsp.mjs';

const $ = selector => document.querySelector(selector);
const names = { en: 'English', zh: '中文', es: 'Español', fr: 'Français', de: 'Deutsch' };
const examples = { en: 'The little bird is singing in the garden.', zh: '今天天气很好，我们一起去公园。',
  es: 'El sol brilla y los pájaros cantan.', fr: 'Le petit chat regarde par la fenêtre.', de: 'Die Sonne scheint und die Vögel singen.' };
const worker = new Worker(new URL('./worker.mjs', import.meta.url), { type: 'module' });
const base = new URL('./', document.baseURI).href;
let language = 'en', submittedLanguage = 'en', url, lastAudio, ready = false;
function count() { $('#character-count').textContent = `${$('#sentence').value.length} / 240`; }
function draw(audio) {
  const canvas = $('#waveform'), ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect(); canvas.width = rect.width * ratio; canvas.height = rect.height * ratio;
  const ctx = canvas.getContext('2d'); ctx.scale(ratio, ratio); ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.strokeStyle = '#c6d2de'; ctx.beginPath(); ctx.moveTo(0, rect.height / 2); ctx.lineTo(rect.width, rect.height / 2); ctx.stroke();
  if (!audio) return;
  ctx.strokeStyle = '#214fe0'; ctx.lineWidth = 1.5;
  for (let x = 0; x < rect.width; x += 3) {
    const start = Math.floor(x / rect.width * audio.length), end = Math.max(start + 1, Math.floor((x + 3) / rect.width * audio.length));
    let peak = 0; for (let i = start; i < Math.min(end, audio.length); i++) peak = Math.max(peak, Math.abs(audio[i]));
    const height = Math.min(1, peak * 2) * rect.height * 0.44;
    ctx.beginPath(); ctx.moveTo(x, rect.height / 2 - height); ctx.lineTo(x, rect.height / 2 + height); ctx.stroke();
  }
}
function failure(message) { $('#status').textContent = message; $('#status').dataset.error = 'true'; $('#generate').disabled = !ready; $('#generate').textContent = ready ? 'Generate speech' : 'Model unavailable'; }
worker.onerror = event => failure(event.message || 'The speech worker could not start. Reload to try again.');
worker.onmessage = ({ data }) => {
  if (data.type === 'ready') { ready = true; $('#generate').disabled = false; $('#generate').textContent = 'Generate speech'; $('#status').textContent = 'Ready on this device.'; return; }
  if (data.type === 'error') { failure(data.message); return; }
  if (data.type !== 'audio') return;
  lastAudio = data.audio;
  if (url) URL.revokeObjectURL(url);
  url = URL.createObjectURL(new Blob([wavBytes(lastAudio)], { type: 'audio/wav' }));
  $('#player').src = url; $('#player').hidden = false; $('#download').href = url; $('#download').hidden = false;
  $('#download').download = `xiaotts-${submittedLanguage}.wav`;
  $('#empty-audio').hidden = true; $('#duration').textContent = `${data.metrics.audioSeconds.toFixed(1)} s`;
  $('#audio-language').textContent = names[submittedLanguage]; draw(lastAudio);
  $('#status').textContent = data.metrics.clippedFraction > 0 ? 'Audio generated; some samples clipped. This checkpoint needs more training.' : 'Audio generated on your CPU.';
  $('#performance').textContent = `${(data.metrics.synthesisMs / 1000).toFixed(2)} s to generate ${data.metrics.audioSeconds.toFixed(1)} s of audio · ${data.metrics.realTimeFactor.toFixed(2)}× real-time factor`;
  $('#generate').disabled = false; $('#generate').textContent = 'Generate speech';
  window.__lastSynthesis = { metrics: data.metrics, input: data.input, language: submittedLanguage };
};
$('#speech-form').addEventListener('submit', event => {
  event.preventDefault(); submittedLanguage = language;
  $('#generate').disabled = true; $('#generate').textContent = 'Generating…';
  $('#status').textContent = 'Turning your sentence into audio…'; delete $('#status').dataset.error;
  worker.postMessage({ type: 'generate', base, language, text: $('#sentence').value });
});
$('#languages').addEventListener('change', event => { language = event.target.value; $('#sentence').lang = language; $('#sentence').value = examples[language]; count(); });
$('#example').addEventListener('click', () => { $('#sentence').value = examples[language]; count(); $('#sentence').focus(); });
$('#sentence').addEventListener('input', count);
window.addEventListener('resize', () => draw(lastAudio));
count(); draw(); worker.postMessage({ type: 'load', base });

if (import.meta.env.PROD && 'serviceWorker' in navigator) {
  navigator.serviceWorker.register(new URL('sw.js', base), { scope: './', type: 'classic' }).then(async () => {
    await navigator.serviceWorker.ready;
    $('#offline-status').textContent = 'All five languages saved for offline use.';
    document.documentElement.dataset.offlineReady = 'true';
  }).catch(() => { $('#offline-status').textContent = 'Offline saving failed. Keep this page online to load assets.'; });
} else $('#offline-status').textContent = 'Offline caching is available in the production build.';
fetch(new URL('asset-report.json', base)).then(response => response.ok ? response.json() : null).then(report => {
  if (report) $('#assets').textContent = `${(report.neuralGraphBytes / 1e6).toFixed(2)} MB of neural graphs. ${(report.uncompressedBytes / 1e6).toFixed(2)} MB for all uncompressed app assets, including pronunciation tables and the CPU runtime.`;
}).catch(() => {});
