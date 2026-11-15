# Browser demo design

The subject is a small, multilingual speech instrument. Its primary action is to
type a sentence, choose its language, and hear the model run on this device.

Palette: mist #edf2f7, white #ffffff, ink #243b53, cobalt #214fe0, slate #c6d2de,
and signal amber #eab74f for the early-checkpoint notice.
Type: system Aptos/Segoe UI with native multilingual fallbacks; large quiet display
type, readable 17px editor text, no externally fetched fonts.

Layout: a left-aligned editor and language controls beside an audio monitor, which
becomes a single stack on narrow screens.

```
XiaoTTS                                     About / Source
One small model. Five languages.
English   中文   Español   Français   Deutsch
[ text editor                     ] [ actual waveform ]
[ example             Generate    ] [ play / save      ]
Early-checkpoint note and measured runtime details
```

Review against the brief: a generic hero and metric cards would compete with the
demo. Keep the actual editor as the focal point, with one waveform drawn only from
generated audio. No gradients, stock imagery, decorative dashboard cards or fake
performance numbers. The current poor speech quality must be visible in the copy.
