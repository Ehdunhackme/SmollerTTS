from xiaotts.phones import word_stress, VOICES
import json
import subprocess
for language, text in {
    'en': 'the is in a of an and to little garden',
    'fr': 'le la les de un une du des chat',
    'de': 'die der das und ist ein eine',
    'es': 'el la los de un una y es',
}.items():
    for word in text.split():
        ipa = subprocess.run(['espeak-ng', '-q', '--ipa=1', '-v', VOICES[language], word], capture_output=True, text=True).stdout.strip()
        print(json.dumps(dict(language=language, word=word, stress=word_stress(word, language), ipa=ipa), ensure_ascii=False))
