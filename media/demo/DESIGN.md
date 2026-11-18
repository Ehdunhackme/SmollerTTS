# SmollerTTS demo video

White background, black logo and type, and a waveform drawn from the actual
SmollerTTS audio. The only motion is the waveform scrolling with playback and
a brief crossfade between sentence captions. No app controls appear in the export.

- Canvas: 1920 × 1080, 30 fps.
- Palette: white `#ffffff`, ink `#111111`, unplayed waveform `#cdcdcd`.
- Type: the same Geist variable font used by the website. Brand 64 px and
  captions 58 px. All content is left aligned on a 144 px inset. No specs line.
- Header stays fixed. A full-width waveform sits in the middle. Captions have
  two reserved lines below it, so the layout does not jump when text changes.
- Bars sample a 6.4-second window of the real concatenated WAV, with played
  samples black and future samples light gray. The audio scrolls through a fixed
  playback position, with soft edge fades. No synthetic waveform or fake activity.

```text
  [mark] SmollerTTS


  ||||| |||||||||||| ||||||||||| ||||||||||| |||||||


  A short caption synchronized
  with the model's speech.
```

The supplied Tiny Voca video establishes the scrolling-waveform reference.
SmollerTTS uses its own logo, Geist type, white canvas and monochrome palette.
The model's voice is the focus. No specs strip, speaker badge or decorative footer.

Edit `script.json`, run the website locally, then run:

```sh
python scripts/render-demo.py --url http://127.0.0.1:5175/projects/smollertts/
node scripts/preview-demo.mjs
```

Requires FFmpeg on PATH, Playwright with Chromium, Pillow, NumPy and fontTools
with WOFF support (`pip install playwright pillow numpy "fonttools[woff]"`).
Outputs are in `output/demo/`: MP4, WAV, SRT, poster, self-contained preview
page, and render/audio provenance. The script uses the actual browser engine
at Natural settings and pronunciation overrides for the brand name and “megabytes.” It does
not change model weights or process the voice with EQ, denoising or pitch effects.
The preview server supports byte-range requests so the exported video can be scrubbed.

To reuse captured audio without changing its samples, run:

```sh
python scripts/render-demo.py --reuse-audio
```

`scripts/export-social.py` also requires Matplotlib. It exports the existing
UTMOS and WER snapshot and the final architecture figure to `output/x-post/`,
plus a ZIP of those three PNGs. It generates the logo-only link preview and
portfolio thumbnail in `web/`. No new benchmark scores are calculated.
