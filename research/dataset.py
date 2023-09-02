import os
import re


import torchaudio
from torch.utils.data import Dataset

class IPS1ASRDataset(Dataset):
    def __init__(self, audio_dir: str):
        self.audio_dir = audio_dir
        self.data = {}
        names = set()
        counter = 0
        for filename in os.listdir(self.audio_dir):
            name = os.path.splitext(filename)[0]
            if name not in names:
                self.data[counter] = {
                    'text': name + '.txt',
                    'audio': name + '.wav'
                }
                counter += 1
                names.add(name)
        self.len = counter
        del names
        del counter

    def __len__(self):
        return self.len

    def _get_audio_sample_path(self, index):
        return self.audio_dir + self.data[index]['audio']

    def _get_audio_sample_label(self, index):
        label_path = self.audio_dir + self.data[index]['text']
        with open(label_path, 'r') as f:
            label = f.read() # Не учитывает ошибки в заполнение .txt
        return label

    def __getitem__(self, index: int):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sample_rate = torchaudio.load(audio_sample_path)

        return (signal, sample_rate, label, 0, 0, 0)

    def get_metadata(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sample_rate = torchaudio.load(audio_sample_path)
        return (audio_sample_path, sample_rate, label, 0, 0, 0)


if __name__ == '__main__':
    ips_dataset = IPS1ASRDataset('../tatar_tts/')
    print('ips_dataset[0] =', ips_dataset[0])
    print('len ips_dataset =', len(ips_dataset))
