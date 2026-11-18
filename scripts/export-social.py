"""Export logo previews and three shareable figures from the published snapshot.

Requires Pillow, fontTools[woff], numpy, matplotlib and the render-demo helpers.
The figures reuse recorded results; this script does not run a new benchmark.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator, PercentFormatter
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/x-post"
OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("demo", ROOT / "scripts/render-demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)

INK = "#171717"
MUTED = "#676767"
COLORS = {"SmollerTTS": "#c95645", "Under 10M": "#527baf", "10–100M": "#a88735", "100M+": "#8c75b4"}


def brand(canvas, x, y, size=48):
    logo = demo.mark(size)
    canvas.paste(logo, (x, y), logo)
    ImageDraw.Draw(canvas).text((x + size + 18, y + size // 2), "SmollerTTS",
                               font=demo.font(OUT, 550, round(size * .7)), fill=INK, anchor="lm")


def previews():
    logo = Image.new("RGB", (400, 400), "white")
    mark = demo.mark(256)
    logo.paste(mark, (72, 72), mark)
    logo.save(ROOT / "web/logo.png", optimize=True)
    social = Image.new("RGB", (1200, 630), "white")
    size = 112
    font = demo.font(OUT, 550, 80)
    text_width = ImageDraw.Draw(social).textlength("SmollerTTS", font=font)
    x = round((1200 - size - 32 - text_width) / 2)
    mark = demo.mark(size)
    social.paste(mark, (x, 259), mark)
    ImageDraw.Draw(social).text((x + size + 32, 315), "SmollerTTS", font=font, fill=INK, anchor="lm")
    social.save(ROOT / "web/social.png", optimize=True)


def group(row):
    if row.get("local"):
        return "SmollerTTS"
    return "Under 10M" if row["parameters_m"] < 10 else "10–100M" if row["parameters_m"] < 100 else "100M+"


def chart(data, metric, filename):
    rows = [row for row in data["models"] if isinstance(row.get(metric), (int, float))]
    own = next(row for row in rows if row.get("local"))
    fig = plt.figure(figsize=(24, 16), dpi=100, facecolor="white")
    ax = fig.add_axes([.092, .225, .85, .52])
    ax.set_xscale("log")
    ax.set_xlim(.07, max(row["parameters_m"] for row in rows) * 1.5)
    ax.set_xticks([.1, 1, 10, 100, 1000, 10000], ["100K", "1M", "10M", "100M", "1B", "10B"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Parameters · log scale", fontsize=18, color=MUTED, labelpad=24)
    ax.tick_params(axis="both", which="both", length=0, pad=14, labelsize=18, colors=MUTED)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e9e9e9", linewidth=1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("#d4d4d4")
    if metric == "utmos":
        title, direction, unit = "Speech quality vs. model size (rough)", "Higher is better", "UTMOS ↑"
        ax.set_ylim(2, 5)
        ax.set_yticks([2, 2.5, 3, 3.5, 4, 4.5, 5])
        value, note = f"{own[metric]:.2f} UTMOS", "UTMOS predicts audio quality; it is not a human rating."
    else:
        title, direction, unit = "Word errors vs. model size (rough)", "Lower is better", "WER ↓"
        ax.set_ylim(0, .7)
        ax.set_yticks([0, .1, .2, .3, .4, .5, .6, .7])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        value, note = f"{own[metric] * 100:.2f}% WER", "WER is averaged across prompts using Whisper large-v3 transcripts."
    fig.text(.05, .861, title, fontsize=38, color=INK, weight=500)
    fig.text(.05, .811, f"{unit}   ·   {direction}", fontsize=18, color=MUTED)

    # Descriptive frontier: the best observed score at or below each size.
    frontier_x, frontier_y, best = [], [], None
    for row in sorted(rows, key=lambda r: r["parameters_m"]):
        score = row[metric]
        if best is None or (score > best if metric == "utmos" else score < best):
            best = score
            if frontier_x and frontier_x[-1] == row["parameters_m"]:
                frontier_y[-1] = score
            else:
                frontier_x.append(row["parameters_m"])
                frontier_y.append(score)
    ax.step(frontier_x + [ax.get_xlim()[1]], frontier_y + [best], where="post", color="#c9c9c9", linewidth=1.6, zorder=1)
    for category in list(COLORS)[1:]:
        selected = [row for row in rows if group(row) == category]
        ax.scatter([r["parameters_m"] for r in selected], [r[metric] for r in selected],
                   color=COLORS[category], alpha=.76, s=90, edgecolors="white", linewidths=.8, zorder=3)
    ax.scatter([own["parameters_m"]], [own[metric]], s=520, color=COLORS["SmollerTTS"], alpha=.12, linewidths=0, zorder=4)
    ax.scatter([own["parameters_m"]], [own[metric]], s=230, color=COLORS["SmollerTTS"], edgecolors="white", linewidths=2, zorder=5)
    target = (own["parameters_m"], own[metric])
    label_at = (1.9, 2.82) if metric == "utmos" else (.65, .26)
    ax.annotate(f"SmollerTTS\n2.81M · {value}", xy=target, xytext=label_at,
                fontsize=21, color=COLORS["SmollerTTS"], linespacing=1.5, ha="left", va="center",
                arrowprops={"arrowstyle": "-", "color": "#c9564580", "lw": 1.25,
                            "shrinkA": 10, "shrinkB": 14, "connectionstyle": "angle,angleA=0,angleB=90,rad=0"}, zorder=6)
    offsets = ({"auroratts": (0, -28), "sanotts_amy": (-14, 24), "tinyvoca": (14, 28), "kokoro": (12, 24)}
               if metric == "utmos" else {"auroratts": (8, 23), "tinyvoca": (8, 32)})
    for model_id, offset in offsets.items():
        row = next((r for r in rows if r["id"] == model_id), None)
        if row:
            ax.annotate(row["name"], (row["parameters_m"], row[metric]), xytext=offset,
                        textcoords="offset points", fontsize=16, color=MUTED,
                        ha="right" if offset[0] < 0 else "left", va="center")
    handles = [Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor=color,
                      markeredgecolor="none", label=name) for name, color in COLORS.items()]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(.95, .823), ncol=4,
               frameon=False, fontsize=17, handletextpad=.3, columnspacing=1.5, labelcolor=MUTED)
    fig.text(.05, .12, note, fontsize=16, color=MUTED)
    fig.text(.05, .091, "Four prompts; voices, runtimes and scoring revisions differ. Gray line: best observed score at each size.", fontsize=16, color=MUTED)
    fig.text(.05, .05, f"Source: tts-bench ({data['source_revision'][:7]}) + SmollerTTS browser output", fontsize=15, color=MUTED)
    fig.text(.95, .05, "dunlichan.com/projects/smollertts/", fontsize=15, ha="right", color=INK)
    target = OUT / filename
    fig.savefig(target, dpi=100, facecolor="white", metadata={"Description": note + " " + data["note"]})
    plt.close(fig)
    image = Image.open(target).convert("RGB")
    brand(image, 120, 70, 48)
    image.save(target, optimize=True)


def main():
    for weight in [400, 500, 550]:
        demo.font(OUT, weight, 32)
        font_manager.fontManager.addfont(str(OUT / f"geist-{weight}.ttf"))
    plt.rcParams.update({"font.family": "Geist", "font.weight": 400, "axes.unicode_minus": False})
    previews()
    data = json.loads((ROOT / "benchmarks/overview.json").read_text(encoding="utf-8"))
    chart(data, "utmos", "01-utmos.png")
    chart(data, "wer", "02-wer.png")
    architecture = Image.open(ROOT / "docs/assets/architecture.png").convert("RGB")
    canvas = Image.new("RGB", (architecture.width, architecture.height + 160), "white")
    canvas.paste(architecture, (0, 160))
    brand(canvas, 90, 55, 48)
    ImageDraw.Draw(canvas).text((architecture.width - 90, 79), "Architecture", anchor="rm",
                                font=demo.font(OUT, 400, 30), fill=MUTED)
    canvas.save(OUT / "03-architecture.png", optimize=True)
    files = ["01-utmos.png", "02-wer.png", "03-architecture.png"]
    with zipfile.ZipFile(OUT / "smollertts-images.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(OUT / name, name)
    manifest = {"benchmark_snapshot": data["source_revision"], "files": {}}
    for name in files:
        raw = (OUT / name).read_bytes()
        manifest["files"][name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "dimensions": Image.open(OUT / name).size}
    (OUT / "export.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
