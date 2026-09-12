from asr.bodo_asr import BodoASR
from translation.bodo_english import BodoEnglishTranslator
from translation.english_bodo import EnglishBodoTranslator


class BodoVaani:
    def __init__(self):
        print("Initializing BodoVaani...")

        # Bodo speech recognition
        self.asr = BodoASR()

        # Bodo → English
        self.bodo_english = BodoEnglishTranslator()

        # English → Bodo
        self.english_bodo = EnglishBodoTranslator()

        print("BodoVaani ready")

    def process_bodo_to_english(self, audio_path):
        print("\n[1/2] Bodo Speech → Bodo Text")
        bodo_text = self.asr.transcribe(audio_path)

        print("[2/2] Bodo Text → English")
        english_text = self.bodo_english.translate(bodo_text)

        return {
            "bodo": bodo_text,
            "english": english_text,
        }

    def process_english_to_bodo(self, text):
        print("\n[1/1] English Text → Bodo")
        bodo_text = self.english_bodo.translate(text)

        return {
            "english": text,
            "bodo": bodo_text,
        }

    def unload(self):
        self.bodo_english.unload()
        self.english_bodo.unload()
