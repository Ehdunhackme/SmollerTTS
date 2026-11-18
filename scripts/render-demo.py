"""Render a monochrome demo using the released browser voice and real waveform."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_wav(path):
    with wave.open(str(path)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 22050)
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").copy()


def capture_voice(spec, out, url):
    speech = out / "speech"
    speech.mkdir(exist_ok=True)
    records = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        page.wait_for_load_state("networkidle")
        page.wait_for_function("window.ready || window.failure", timeout=180000)
        assert page.evaluate("window.ready"), page.evaluate("window.failure")
        page.locator("#settings-toggle").click()
        page.locator("#stream").uncheck()
        page.locator("#style").select_option("natural")
        page.locator("#pronunciations").fill(spec["pronunciations"])
        model = page.evaluate("async () => (await fetch('./model.json')).json()")
        for index, clip in enumerate(spec["clips"]):
            page.locator("#text").fill(clip["text"])
            page.locator("#generate").click()
            page.wait_for_function("window.lastResult || document.querySelector('#status').dataset.error === 'true'", timeout=180000)
            result = page.evaluate("window.lastResult")
            assert result, page.locator("#status").inner_text()
            with page.expect_download() as download:
                page.locator("#download").click()
            filename = speech / f"{index + 1:02d}.wav"
            download.value.save_as(filename)
            records.append({"text": clip["text"], "file": filename.relative_to(out).as_posix(), "sha256": sha(filename), "inference": result})
            print(f"Voice {index + 1}/{len(spec['clips'])}: {result['samples'] / 22050:.2f}s", flush=True)
        browser.close()
    save_json(out / "voice.json", {"model": model, "pronunciations": spec["pronunciations"], "records": records})


def font(out, weight, size):
    target = out / f"geist-{weight}.ttf"
    if not target.exists():
        original = TTFont(ROOT / "web/fonts/geist-latin-wght-normal.woff2")
        instance = instantiateVariableFont(original, {"wght": weight}, inplace=False)
        instance.flavor = None
        instance.save(target)
    return ImageFont.truetype(str(target), size)


def mark(size):
    # Same geometry as web/mark.svg, rasterized at 4x for smooth edges.
    scale = size * 4 / 40
    im = Image.new("RGBA", (size * 4, size * 4), (255, 255, 255, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, size * 4 - 1, size * 4 - 1), radius=13 * scale, fill="#111111")
    for x, start, end in [(11, 17, 23), (17, 11, 29), (23, 14, 26), (29, 18, 22)]:
        d.rounded_rectangle(((x - 1.5) * scale, (start - 1.5) * scale, (x + 1.5) * scale, (end + 1.5) * scale), radius=1.5 * scale, fill="white")
    return im.resize((size, size), Image.Resampling.LANCZOS)


def srt_time(t):
    ms = round(t * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def prepare(spec, out):
    voice = json.loads((out / "voice.json").read_text(encoding="utf-8"))
    assert voice["pronunciations"] == spec["pronunciations"], "Recapture the changed pronunciation"
    assert len(voice["records"]) == len(spec["clips"])
    sr = 22050
    pieces = [np.zeros(round(spec["intro_seconds"] * sr), dtype="<i2")]
    offset = len(pieces[0])
    clips = []
    for index, (clip, recording) in enumerate(zip(spec["clips"], voice["records"])):
        path = out / recording["file"]
        assert recording["text"] == clip["text"] and sha(path) == recording["sha256"], "Voice changed; recapture"
        pcm = read_wav(path)
        start, end = offset / sr, (offset + len(pcm)) / sr
        clips.append({**clip, "start": start, "end": end, "pcm": pcm})
        pieces.append(pcm)
        gap = spec["outro_seconds"] if index == len(spec["clips"]) - 1 else spec["gap_seconds"]
        pieces.append(np.zeros(round(gap * sr), dtype="<i2"))
        offset += len(pcm) + len(pieces[-1])
    frames = math.ceil(offset / sr * spec["fps"])
    audio = np.concatenate(pieces)
    audio = np.pad(audio, (0, max(0, round(frames / spec["fps"] * sr) - len(audio))))
    with wave.open(str(out / "smollertts-demo.wav"), "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(sr)
        wav.writeframes(audio.astype("<i2").tobytes())
    captions = []
    for index, clip in enumerate(clips):
        captions.append(f"{index + 1}\n{srt_time(clip['start'])} --> {srt_time(clip['end'])}\n{clip['caption']}\n")
    (out / "smollertts-demo.srt").write_text("\n".join(captions), encoding="utf-8")
    return clips, frames


def renderer(spec, out, clips):
    width, height = spec["width"], spec["height"]
    assert (width, height) == (1920, 1080), "Layout is composed for 1080p"
    base = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(base)
    logo = mark(72)
    base.paste(logo, (144, 122), logo)
    d.text((240, 123), "SmollerTTS", font=font(out, 550, 64), fill="#111111", anchor="lt", stroke_width=0)
    caption_font = font(out, 430, 58)
    x, y, wave_width, wave_height = 144, 500, 1632, 240
    n = 116
    # A continuously scrolling envelope of the unchanged, concatenated audio.
    # Interpolation moves measured audio across the screen; no synthetic motion.
    pcm = read_wav(out / "smollertts-demo.wav").astype(float)
    hop = 441
    envelope = np.array([np.max(np.abs(pcm[i:i + hop])) for i in range(0, len(pcm), hop)])
    envelope = np.convolve(envelope, [.15, .7, .15], mode="same")
    envelope = (envelope / max(1., envelope.max())) ** .65
    times = (np.arange(len(envelope)) + .5) * hop / 22050
    positions = (np.arange(n) + .5) / n
    offsets = (positions - spec["wave_playhead"]) * spec["wave_window_seconds"]
    edge_alpha = np.minimum(1, np.minimum(positions, 1 - positions) / .065)
    layers = []
    for clip in clips:
        caption = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        cd = ImageDraw.Draw(caption)
        for line, text in enumerate(clip["caption"].splitlines()):
            assert cd.textlength(text, font=caption_font) <= wave_width, f"Caption too wide: {text}"
            cd.text((144, 752 + line * 78), text, font=caption_font, fill="#111111", anchor="lt")
        layers.append(caption)

    def frame(t):
        index = max([i for i, clip in enumerate(clips) if t >= clip["start"] - 0.14] or [0])
        clip = clips[index]
        caption = layers[index]
        im = base.copy()
        wave_draw = ImageDraw.Draw(im)
        values = np.interp(t + offsets, times, envelope, left=0, right=0)
        for i, value in enumerate(values):
            bar_height = max(6, float(value) * wave_height)
            left = x + positions[i] * wave_width - 3.5
            shade = 17 if positions[i] <= spec["wave_playhead"] else 205
            shade = round(255 - (255 - shade) * edge_alpha[i])
            wave_draw.rounded_rectangle((left, y - bar_height / 2, left + 7, y + bar_height / 2), radius=3.5, fill=(shade, shade, shade))
        alpha = min(1, max(0, (t - clip["start"] + 0.14) / 0.12))
        if index == 0:
            alpha = min(1, max(0, (t - 0.16) / 0.20))
        if alpha < 1:
            mask = caption.getchannel("A").point(lambda value: round(value * alpha))
            im.paste(caption, (0, 0), mask)
        else:
            im.paste(caption, (0, 0), caption)
        return im
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5175/projects/smollertts/")
    parser.add_argument("--script", type=Path, default=ROOT / "media/demo/script.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output/demo")
    parser.add_argument("--reuse-audio", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.script.read_text(encoding="utf-8"))
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if not args.reuse_audio:
        capture_voice(spec, out, args.url)
    clips, frames = prepare(spec, out)
    draw = renderer(spec, out, clips)
    poster_time = clips[1]["start"] + (clips[1]["end"] - clips[1]["start"]) * 0.48
    draw(poster_time).save(out / "smollertts-poster.png")
    contact = Image.new("RGB", (1280, 1080), "white")
    for index, clip in enumerate(clips):
        im = draw((clip["start"] + clip["end"]) / 2).resize((640, 360), Image.Resampling.LANCZOS)
        contact.paste(im, ((index % 2) * 640, (index // 2) * 360))
    contact.save(out / "contact-sheet.jpg", quality=94)
    if args.preview_only:
        print(f"Preview ready: {out}")
        return
    command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1920x1080", "-r", str(spec["fps"]), "-i", "pipe:0",
               "-i", str(out / "smollertts-demo.wav"), "-c:v", "libx264", "-preset", "medium", "-crf", "18",
               "-pix_fmt", "yuv420p", "-threads", "4", "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
               "-movflags", "+faststart", "-map_metadata", "-1", str(out / "smollertts-demo.mp4")]
    with (out / "ffmpeg.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for number in range(frames):
                process.stdin.write(draw(number / spec["fps"]).tobytes())
                if number % (spec["fps"] * 5) == 0:
                    print(f"Rendering {number / spec['fps']:.0f} / {frames / spec['fps']:.1f} seconds", flush=True)
            process.stdin.close()
            assert process.wait() == 0, "See ffmpeg.log"
        finally:
            if process.poll() is None:
                process.kill()
    save_json(out / "render.json", {"script": spec, "model": json.loads((out / "voice.json").read_text(encoding="utf-8"))["model"],
              "width": 1920, "height": 1080, "fps": spec["fps"], "duration": frames / spec["fps"],
              "audio_processing": "Original PCM16 browser WAVs concatenated with silence. No EQ, denoising, time stretch or pitch changes. MP4 audio is AAC encoded; the companion WAV retains original PCM.",
              "timeline": [{k: v for k, v in clip.items() if k != "pcm"} for clip in clips],
              "mp4_sha256": sha(out / "smollertts-demo.mp4"), "wav_sha256": sha(out / "smollertts-demo.wav")})
    (out / "index.html").write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SmollerTTS demo</title><style>html{background:#fff;color:#111;font-family:Arial,sans-serif}body{margin:0;padding:32px;display:grid;min-height:90vh;place-content:center}video{width:min(100%,1280px);display:block}nav{display:flex;gap:24px;margin:24px 0;font-size:14px}a{color:inherit;text-underline-offset:4px}@media(max-width:600px){body{padding:12px}}</style><video controls playsinline preload="metadata" poster="smollertts-poster.png"><source src="smollertts-demo.mp4" type="video/mp4"></video><nav><a href="smollertts-demo.mp4" download>Download MP4</a><a href="smollertts-demo.srt" download>Captions</a><a href="smollertts-demo.wav" download>Original audio</a></nav></html>''', encoding="utf-8")
    print(f"Finished: {out / 'smollertts-demo.mp4'}", flush=True)


if __name__ == "__main__":
    main()
