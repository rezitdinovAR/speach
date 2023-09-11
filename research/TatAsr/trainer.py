import gc
import os

import comet_ml
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import Trainer, TrainingArguments

from whisper_dataset import WhisperDataset

def get_last_model(): 
    checkpoint_path = max(os.listdir('../whisper-dataset-2'), key=lambda x: int(x.split('-')[-1]) if 'checkpoint-' in x else 0)
    checkpoint_path = os.path.join('../whisper-dataset-2', checkpoint_path)
    return AutoModelForSpeechSeq2Seq.from_pretrained(checkpoint_path)


os.environ["COMET_LOG_ASSETS"] = "True"
WHISPER_MODEL = 'openai/whisper-small'
DATASET_DIR_1 = '../../tatar_asr_1/'
DATASET_DIR_2 = '../../tatar_asr_2/'
PROJECT_NAME = 'TatAsr-whisper'
EXPERIMENT_NAME = 'TatAsr-whisper-dataset-all'
MODEL_PATH = '../whisper-dataset-all'

accelerator = Accelerator(mixed_precision='fp16')
processor = AutoProcessor.from_pretrained(WHISPER_MODEL)


model = get_last_model() # AutoModelForSpeechSeq2Seq.from_pretrained(WHISPER_MODEL)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="tatar", task="transcribe")

train_dataset_1 = WhisperDataset(DATASET_DIR_1 + 'train/', processor, model.config.max_length)
train_dataset_2 = WhisperDataset(DATASET_DIR_2 + 'train/', processor, model.config.max_length)
train_dataset = torch.utils.data.ConcatDataset([train_dataset_1, train_dataset_2])

valid_dataset = WhisperDataset(DATASET_DIR_2 + 'valid/', processor, model.config.max_length)
test_dataset = WhisperDataset(DATASET_DIR_1 + 'valid/', processor, model.config.max_length)

torch.cuda.empty_cache()
gc.collect()

comet_ml.init(project_name=PROJECT_NAME, experiment_name=EXPERIMENT_NAME)

training_args = TrainingArguments(
    output_dir=MODEL_PATH,
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=7,
    save_steps=500,
    save_total_limit=2,
    do_train=True,
)

set_seed(42)
torch.manual_seed(7)

trainer = accelerator.prepare(Trainer(
    model,
    training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
))

trainer.train()
