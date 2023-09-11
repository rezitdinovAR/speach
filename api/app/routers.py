from fastapi import APIRouter,  File, UploadFile

from asr import TatAsr

router = APIRouter()

tat_asr = TatAsr()

@router.post('/listening/')
async def listening(file: UploadFile = File(...)) -> dict:
    '''Аудирование - изучение татарских слов'''
    try:
        contents = file.file.read()
        with open(file.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    text = await tat_asr.predict(file.filename)

    return {
        'text': text
    }
