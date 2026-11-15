"""Dictionary pronunciation frontend shared with the alignment phone inventory.

Unknown words fail explicitly. Arbitrary-word browser G2P and number normalization
remain deployment work; this initial path supports unseen sentences of known words.
"""
from pathlib import Path
import re
import unicodedata
import torch
from xiaotts import LANGUAGES
from xiaotts.alignment import dictionary_path, read_dictionary, words
from xiaotts.phones import label_word


class Frontend:
    def __init__(self, data_root, language, vocabulary):
        if language not in LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")
        self.language, self.vocabulary = language, vocabulary
        self.lexicon = read_dictionary(dictionary_path(Path(data_root), language))

    def __call__(self, text):
        text = unicodedata.normalize("NFKC", text)
        if re.search(r"\d", text):
            raise ValueError("Number normalization is not implemented yet; spell numbers out")
        labels, pronunciations = [("<sil>", 0, 0)], []
        unknown = []
        for chunk in re.split(r"([,.;:!?，。！？；：])", text):
            if not chunk.strip():
                continue
            if re.fullmatch(r"[,.;:!?，。！？；：]", chunk):
                if labels[-1][0] != "<sil>":
                    labels.append(("<sil>", 0, 0))
                continue
            for word in words(chunk, self.language, self.lexicon):
                if word not in self.lexicon:
                    unknown.append(word)
                    continue
                phones = self.lexicon[word]
                marked, _ = label_word(phones, word, self.language)
                labels.extend(marked)
                pronunciations.append((word, phones))
        if unknown:
            raise ValueError(f"Pronunciation missing; no words were silently dropped: {sorted(set(unknown))}")
        if len(labels) == 1:
            raise ValueError("Text contains no pronounceable words")
        if labels[-1][0] != "<sil>":
            labels.append(("<sil>", 0, 0))
        return dict(tokens=torch.tensor([[self.vocabulary[phone] for phone, _, _ in labels]]),
                    tones=torch.tensor([[tone for _, tone, _ in labels]]),
                    stress=torch.tensor([[stress for _, _, stress in labels]]),
                    language=torch.tensor([LANGUAGES.index(self.language)])), pronunciations
