"""Export fixed pronunciation tables and native frontend parity cases for the demo.

Uses train/dev pronunciation dictionaries only. It never reads final-test text.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import shutil
import subprocess
from xiaotts import LANGUAGES
from xiaotts.alignment import dictionary_path, read_dictionary
from xiaotts.phones import split_phone, is_vowel, label_word
from xiaotts.frontend import Frontend
from xiaotts.fetch import atomic_json

EXAMPLES = {
    'en': 'The little bird is singing in the garden.',
    'zh': '今天天气很好，我们一起去公园。',
    'es': 'El sol brilla y los pájaros cantan.',
    'fr': 'Le petit chat regarde par la fenêtre.',
    'de': 'Die Sonne scheint und die Vögel singen.',
}

_VOCABULARY = None


def initialize_vocabulary(vocabulary):
    global _VOCABULARY
    _VOCABULARY = vocabulary


def annotate_word(job):
    language, word, phones = job
    try:
        # Native tokenization cannot produce a leading hyphen. Retain unreachable
        # dictionary keys without passing them as executable options to eSpeak.
        if word.startswith('-'):
            labels = [split_phone(phone) for phone in phones]
        else:
            labels, _ = label_word(phones, word, language)
        return word, [[_VOCABULARY[phone], tone, stress] for phone, tone, stress in labels], None
    except (ValueError, KeyError, subprocess.CalledProcessError) as error:
        return word, None, str(error)


def binary_lexicon(lexicon):
    records, offsets = [], [0]
    for word in sorted(lexicon, key=lambda word: word.encode('utf-8')):
        encoded, labels = word.encode('utf-8'), lexicon[word]
        if len(encoded) > 65535 or len(labels) > 65535:
            raise ValueError('Oversized dictionary entry')
        codes = []
        for phone, tone, stress in labels:
            if not (0 <= phone < 512 and 0 <= tone < 8 and 0 <= stress < 4):
                raise ValueError('Pronunciation label exceeds packed format')
            codes.append(phone | tone << 9 | stress << 12)
        record = struct.pack('<HH', len(encoded), len(labels)) + encoded + struct.pack('<'+'H'*len(codes), *codes)
        records.append(record)
        offsets.append(offsets[-1] + len(record))
    return b'XTL1' + struct.pack('<I', len(records)) + struct.pack('<'+'I'*len(offsets), *offsets) + b''.join(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data'))
    parser.add_argument('--vocabulary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--format', choices=('binary', 'json'), default='binary')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    vocab = json.loads(args.vocabulary.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    report, fixtures = {}, []
    if args.workers < 1:
        raise ValueError('At least one annotation worker is required')
    executable = Path(shutil.which('espeak-ng') or '')
    if not executable.is_file():
        raise ValueError('Native eSpeak NG is required to compile matching stress annotations')
    phonemizer = dict(version=subprocess.run(['espeak-ng', '--version'], check=True, capture_output=True, text=True).stdout.strip(),
                     binary_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                     phones_source_sha256=hashlib.sha256(Path('xiaotts/phones.py').read_bytes()).hexdigest())
    cache_root = args.root/'browser-native-lexicons'
    cache_root.mkdir(parents=True, exist_ok=True)
    for language in LANGUAGES:
        identity = dict(language=language, phonemizer=phonemizer, vocabulary_sha256=hashlib.sha256(args.vocabulary.read_bytes()).hexdigest(),
                        dictionary_sha256=hashlib.sha256(dictionary_path(args.root, language).read_bytes()).hexdigest())
        cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        cache_path = cache_root/(language+'-'+cache_key+'.json')
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            if cached['identity'] != identity:
                raise ValueError('Cached pronunciation identity mismatch')
            lexicon, rejected = cached['lexicon'], cached['rejected']
        else:
            lexicon, rejected = {}, []
            dictionary = read_dictionary(dictionary_path(args.root, language))
            jobs = ((language, word, phones) for word, phones in dictionary.items())
            with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_vocabulary, initargs=(vocab,)) as pool:
                for number, (word, labels, failure) in enumerate(pool.map(annotate_word, jobs, chunksize=128), 1):
                    if failure:
                        rejected.append([word, failure])
                    else:
                        lexicon[word] = labels
                    if number % 10000 == 0:
                        print(f'Native stress {language}: {number}/{len(dictionary)}', flush=True)
            atomic_json(cache_path, dict(identity=identity, lexicon=lexicon, rejected=rejected))
        path = args.output / f'{language}.lex' if args.format == 'binary' else args.output / f'{language}.json'
        if args.format == 'binary':
            path.write_bytes(binary_lexicon(lexicon))
        else:
            path.write_text(json.dumps(lexicon, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
        report[language] = dict(words=len(lexicon), bytes=path.stat().st_size, rejected=rejected, identity=identity)
        frontend = Frontend(args.root, language, vocab)
        inputs, pronunciations = frontend(EXAMPLES[language])
        fixtures.append(dict(language=language, text=EXAMPLES[language],
                             **{key: value.tolist() for key, value in inputs.items() if key != 'language'}))
    atomic_json(args.output/'frontend.json', dict(vocabulary=vocab,
                                               stress_mode='precomputed-native', phonemizer=phonemizer,
                                               vocabulary_sha256=hashlib.sha256(args.vocabulary.read_bytes()).hexdigest(),
                                               lexicon_sha256={language:hashlib.sha256(dictionary_path(args.root, language).read_bytes()).hexdigest() for language in LANGUAGES},
                                               vowels=[vocab[p] for p in vocab if is_vowel(p)],
                                               languages=list(LANGUAGES), examples=EXAMPLES))
    atomic_json(args.output/'frontend-parity.json', fixtures)
    atomic_json(args.output/'lexicon-report.json', report)
    print(json.dumps({language: {k:v for k,v in item.items() if k != 'rejected'} for language,item in report.items()}, indent=2))


if __name__ == '__main__':
    main()
