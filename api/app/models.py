from pydantic import BaseModel


class Audio(BaseModel):
    '''wav'''
    wav_base64: str
    sample_rate: int