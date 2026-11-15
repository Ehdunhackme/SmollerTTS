"""Shared MFA IPA symbols with explicit Mandarin tone and lexical stress labels."""
import re
import subprocess
import unicodedata
from functools import lru_cache

# MFA uses low and mid realizations for neutral syllables, depending on context.
TONES = {"˥": 1, "˧˥": 2, "˨˩˦": 3, "˨˩": 3, "˥˩": 4, "˧": 5, "˩": 5}
VOWELS = set("aeiouyɑɐɒæɛɜɞəɚɝɘɤɨɪɔœøɵʉʊʌɯɶɷ")
VOICES = {"en": "en-us", "zh": "cmn", "es": "es", "fr": "fr-fr", "de": "de"}


def split_phone(phone):
    phone = unicodedata.normalize("NFC", phone)
    if phone in ("", "sil", "sp", "<eps>"):
        return "<sil>", 0, 0
    stress = 1 if "ˈ" in phone else 2 if "ˌ" in phone else 0
    contour = "".join(char for char in phone if char in "˥˦˧˨˩")
    if contour and contour not in TONES:
        raise ValueError(f"Unmapped Mandarin tone contour: {phone}")
    base = re.sub(r"[ˈˌ˥˦˧˨˩]", "", phone)
    return base, TONES.get(contour, 0), stress


def is_vowel(phone):
    return any(char in VOWELS for char in phone) or "\u0329" in phone


@lru_cache(maxsize=100000)
def word_stress(word, language):
    if language == "zh":
        return ()
    result = subprocess.run(["espeak-ng", "-q", "--ipa=1", "-v", VOICES[language], word],
                            check=True, capture_output=True, text=True, encoding="utf-8")
    # --ipa=1 separates phonemes with underscores. Stress remains on the vowel.
    syllables, pending = [], 0
    for phone in re.split(r"[_\s]+", result.stdout.strip()):
        if "ˈ" in phone:
            pending = 1
        elif "ˌ" in phone:
            pending = 2
        if is_vowel(phone):
            syllables.append(pending)
            pending = 0
    return tuple(syllables)


def label_word(phones, word, language):
    labels = [split_phone(phone) for phone in phones]
    vowels = [index for index, (phone, _, _) in enumerate(labels) if is_vowel(phone)]
    stress = word_stress(word, language)
    # Ambiguous syllable mappings remain explicitly unmarked, rather than assigning
    # an accent to an arbitrary phoneme. Coverage is reported by cache preparation.
    matched = len(vowels) == len(stress)
    if language != "zh" and matched:
        for index, value in zip(vowels, stress):
            base, tone, old = labels[index]
            labels[index] = (base, tone, value or old)
    return labels, matched
