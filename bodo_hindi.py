import gc
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit import IndicProcessor


MODEL_NAME = "ai4bharat/indictrans2-indic-indic-dist-320M"


class BodoHindiTranslator:
    def __init__(self):
        print("Loading Bodo → Hindi model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to("cuda")

        self.model.eval()
        self.processor = IndicProcessor(inference=True)

        print("Bodo → Hindi ready")

    def translate(self, text):
        batch = self.processor.preprocess_batch(
            [text],
            src_lang="brx_Deva",
            tgt_lang="hin_Deva",
        )

        inputs = self.tokenizer(
            batch,
            padding="longest",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                use_cache=True,
                max_length=256,
                num_beams=5,
                num_return_sequences=1,
            )

        decoded = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        result = self.processor.postprocess_batch(
            decoded,
            lang="hin_Deva",
        )[0]

        return result

    def unload(self):
        del self.model
        del self.tokenizer
        del self.processor

        gc.collect()
        torch.cuda.empty_cache()
