"""Export fixed pronunciation tables and native frontend parity cases for the demo.

Uses train/dev pronunciation dictionaries only. It never reads final-test text.
"""
import argparse
import json
from pathlib import Path
from xiaotts import LANGUAGES
from xiaotts.alignment import dictionary_path, read_dictionary
from xiaotts.phones import split_phone, is_vowel
from xiaotts.frontend import Frontend
from xiaotts.fetch import atomic_json

EXAMPLES = {
    'en': 'The little bird is singing in the garden.',
    'zh': '今天天气很好，我们一起去公园。',
    'es': 'El sol brilla y los pájaros cantan.',
    'fr': 'Le petit chat regarde par la fenêtre.',
    'de': 'Die Sonne scheint und die Vögel singen.',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data'))
    parser.add_argument('--vocabulary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    vocab = json.loads(args.vocabulary.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    report, fixtures = {}, []
    for language in LANGUAGES:
        lexicon, rejected = {}, []
        for word, phones in read_dictionary(dictionary_path(args.root, language)).items():
            try:
                labels = [split_phone(phone) for phone in phones]
                lexicon[word] = [[vocab[phone], tone, stress] for phone, tone, stress in labels]
            except (ValueError, KeyError) as error:
                rejected.append([word, str(error)])
        path = args.output / f'{language}.json'
        path.write_text(json.dumps(lexicon, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
        report[language] = dict(words=len(lexicon), bytes=path.stat().st_size, rejected=rejected)
        frontend = Frontend(args.root, language, vocab)
        inputs, pronunciations = frontend(EXAMPLES[language])
        fixtures.append(dict(language=language, text=EXAMPLES[language],
                             **{key: value.tolist() for key, value in inputs.items() if key != 'language'}))
    atomic_json(args.output/'frontend.json', dict(vocabulary=vocab, vowels=[vocab[p] for p in vocab if is_vowel(p)],
                                               languages=list(LANGUAGES), examples=EXAMPLES))
    atomic_json(args.output/'frontend-parity.json', fixtures)
    atomic_json(args.output/'lexicon-report.json', report)
    print(json.dumps({language: {k:v for k,v in item.items() if k != 'rejected'} for language,item in report.items()}, indent=2))


if __name__ == '__main__':
    main()
