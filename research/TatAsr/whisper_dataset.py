import os

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from utils import clean_text


class WhisperDataset(Dataset):
    def __init__(self, audio_dir: str, processor, max_length, only_char=True):
        self.audio_dir = audio_dir
        df = pd.read_csv(audio_dir[:-1] + '.csv', index_col='id')
        self.data = {}
        counter = 0
        for row in df.itertuples():
            if not os.path.exists(audio_dir + str(row[0]) + '.txt'):
                print('Отсутствует файл', str(row[0]) + '.txt')
                continue
            if not os.path.exists(audio_dir + str(row[0]) + '.wav'):
                print(f'Отсутствует файл', str(row[0]) + '.wav')
                continue
            self.data[counter] = {
                'text': str(row[0]) + '.txt',
                'audio': str(row[0]) + '.wav'
            }
            counter += 1
        self.len = counter - 1
        self.only_char = only_char
        del counter
        del df
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return self.len

    def _get_audio_sample_path(self, index):
        return self.audio_dir + self.data[index]['audio']

    def _get_audio_sample_label(self, index):
        label_path = self.audio_dir + self.data[index]['text']
        with open(label_path, 'r') as f:
            label = clean_text(f.read()) if self.only_char else f.read()  # Не учитывает ошибки в заполнение .txt
        return label

    def __getitem__(self, idx):
        filepath = self._get_audio_sample_path(idx)
        text = self._get_audio_sample_label(idx)

        audio, sample_rate = torchaudio.load(filepath)
        audio = torch.reshape(audio, (-1,))

        tokenized = self.processor.tokenizer(
            text, return_tensors='pt', padding='max_length', return_attention_mask=True,
            max_length=self.max_length
        )

        labels, attention_mask = tokenized['input_ids'][0], tokenized['attention_mask'][0]

        input_features = self.processor(audio, return_tensors="pt", sampling_rate=sample_rate).input_features[0]

        return {
            'input_features': input_features,
            'labels': labels,
            'attention_mask': attention_mask
        }

