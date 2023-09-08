from transformers import pipeline
import base64
import io
import soundfile as sf


class TatAsr:
    '''Интерфейс взаимодействия с моделями Silero'''

    def __init__(
            self, sample_rate=48000, device='cuda'
    ):
        self.sample_rate = sample_rate
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model="dhcppc0/soyle_29_08",
            device=device)

    def base64_to_wav(self, base64_data: str):
        decode_string = base64.b64decode(base64_data)
        temp_wav_file = io.BytesIO()
        sf.write(temp_wav_file, decode_string, self.sample_rate, format='WAV', subtype='PCM_16')
        temp_wav_file.seek(0)
        return temp_wav_file

    async def predict(self, filepath: str) -> dict:
        text = self.pipe(filepath)['text']
        return text