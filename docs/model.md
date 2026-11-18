# Model details

SmollerTTS has one English voice. Its acoustic network is custom and trained from random initialization. The vocoder is the published HiFi-GAN LJ_V2 generator with weight normalization folded into the convolution weights.

## Acoustic network

| Stage | Configuration |
| :--- | :--- |
| Input | 132 phoneme IDs; three stress levels; sinusoidal positions |
| Embeddings | Width 160; phoneme and stress embeddings summed |
| Text encoder | Four blocks: LayerNorm, four-head self-attention, residual, depthwise convolution (kernel 7), LayerNorm, feed-forward 160 → 480 → 160, learned residual scale |
| Duration head | Two kernel-3 convolutions with LayerNorm/ReLU, then a linear scalar prediction of log(1 + frames) |
| Phone prosody | LayerNorm, linear 160 → 160, GELU, linear 160 → 3 |
| Length regulator | Repeats encoded phones according to duration; adds a learned projection of within-phone position, sine and cosine |
| Frame decoder | Six depthwise-convolution blocks, dilations 1, 2, 4, 8, 1, 2; feed-forward width 320 |
| Prosody conditioning | After the first two frame blocks, predict a residual over phone-level pitch, voicing and energy; project three channels to width 160 |
| Output | Linear 160 → 80, then per-band mel denormalization |

The acoustic model has **1,882,807 trainable parameters**. Mel normalization values are buffers, not trainable parameters. There is no speaker embedding.

During acoustic training the decoder receives recorded prosody targets. During ordinary inference it uses predicted prosody. Pitch is represented as `(ln(F0) − ln(180)) / 0.5`, voicing as a probability, and energy as `(ln(RMS) + 3) / 2`.

## Waveform generator

The frozen HiFi-GAN generator maps 80 mel bands to 128 channels, then upsamples by **8 × 8 × 2 × 2 = 256**. Channel widths decrease to 64, 32, 16 and 8. Each stage combines residual branches with kernel sizes 3, 7 and 11 and dilations 1, 3 and 5. A final convolution and tanh produce mono audio.

After folding weight normalization, the generator has **925,985 parameters**. Training discriminators are not included in inference. This project did not train these vocoder weights.

Output is 22,050 Hz; each mel frame spans a hop of 256 waveform samples. The frontend configuration uses an FFT/window of 1,024, 80 mel bands and a 0–8,000 Hz mel range. Full settings are in [vocoder.json](../model/vocoder.json).

## Browser execution

- A Web Worker runs three FP32 ONNX graphs on a single WebAssembly CPU thread.
- Text normalization and phonemization run locally. Only tested compounds are split automatically; manual pronunciations apply to explicitly named words.
- Length regulation and conservative punctuation/delivery adjustments run in JavaScript between the encoder and frame decoder.
- Playback starts when a complete speech chunk is ready. This is chunk streaming, not token-by-token neural streaming.
- A bounded in-memory audio cache avoids repeated synthesis. Model downloads use a separate persistent cache where available.
- ONNX graph optimizations remain disabled to preserve the verified arithmetic. Changing graph fusion or numerical precision requires new waveform validation.

There is no fixed editor character limit. The frontend divides text into bounded chunks; individual words, malformed notation and generated frame counts still have validation limits. Very long inputs can use substantial memory because the full WAV is assembled for playback and download.

## Files and counts

| File | Bytes | Purpose |
| :--- | ---: | :--- |
| `encoder.onnx` | 5,002,113 | Phoneme encoding, duration and phone prosody |
| `frames.onnx` | 2,592,343 | Frame features and mel prediction |
| `vocoder.onnx` | 3,743,259 | Mel-to-waveform synthesis |
| **Total** | **11,337,715** | **11.34 decimal MB** |

The **2,808,792** parameter count includes the acoustic model and waveform generator. It excludes non-neural dictionaries, runtime code and training-only state. The generated `model-size.json` lists the actual file sizes for each build; download size and runtime memory are separate quantities.

[config.json](../model/config.json) records component counts and checkpoint hashes. [assets.lock.json](../model/assets.lock.json) pins the distributed graphs, pronunciation data and runtime to an immutable Hugging Face revision.
