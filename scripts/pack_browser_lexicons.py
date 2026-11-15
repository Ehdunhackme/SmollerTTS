"""Losslessly convert already-exported JSON pronunciation tables to compact records."""
import json
from pathlib import Path
from scripts.browser_lexicon import binary_lexicon
from xiaotts import LANGUAGES

for language in LANGUAGES:
    path = Path('web/public/model')/f'{language}.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    destination = path.with_suffix('.lex')
    destination.write_bytes(binary_lexicon(data))
    print(language, path.stat().st_size, destination.stat().st_size)
