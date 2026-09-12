from pathlib import Path

import torch
from nemo.collections.asr.models import ASRModel


MODEL_PATH = (
    Path.home()
    / ".cache/torch/NeMo/NeMo_1.23.0rc0/hf_hub_cache/"
    / "ai4bharat/indicconformer_stt_brx_hybrid_ctc_rnnt_large/"
    / "a668fdfb6f459adde887f90191a5c994/"
    / "indicconformer_stt_brx_hybrid_rnnt_large.nemo"
)


class BodoASR:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not available")

        print("Loading Bodo ASR model...")

        self.model = ASRModel.restore_from(
            str(MODEL_PATH),
            map_location="cuda",
        )

        self.model.eval()

        print("Bodo ASR ready")
        print("GPU:", torch.cuda.get_device_name(0))

    def transcribe(self, audio_path):
        result = self.model.transcribe(
            [str(audio_path)],
            language_id="brx",
            verbose=False,
        )

        # NeMo returns a tuple for this model.
        if isinstance(result, tuple):
            result = result[0]

        return result[0].strip()


if __name__ == "__main__":
    asr = BodoASR()

    result = asr.transcribe("audio/bodo_test.wav")

    print("\nBodo transcription:")
    print(result)
