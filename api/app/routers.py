from fastapi import APIRouter

from .asr import TatAsr
from .models import Audio

router = APIRouter()

tat_asr = TatAsr()

@router.get('/listening/', response_model=Audio)
async def listening(base64_wav: str) -> dict:
    '''Аудирование - изучение татарских слов'''
    file_path = tat_asr.base64_to_wav(base64_wav)
    text = await tat_asr.predict(file_path)

    return {
        'text': text
    }