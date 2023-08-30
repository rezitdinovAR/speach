from transformers import pipeline
from dataset import IPS1ASRDataset
from utils import *
from tqdm import trange
import warnings
warnings.filterwarnings("ignore")

pipe = pipeline("automatic-speech-recognition", model="dhcppc0/soyle_29_08", device='cuda')
ips_dataset = IPS1ASRDataset('../tatar_tts/')

assert len(ips_dataset[0]) == 2
assert ips_dataset[0][1] == 'караса аны җен шулай чакыра ди'
assert len(ips_dataset) == 87438

test_cer, test_wer = [], []
n = int(input('Введите размер датасета: '))


for index in trange(n):
    path, label = ips_dataset[index]
    pred_label = pipe(path)['text']
    words = min(len(pred_label), len(label))
    test_cer.append(cer(pred_label, label))
    test_wer.append(wer(pred_label, label))

avg_cer = sum(test_cer)/len(test_cer) * 100
avg_wer = sum(test_wer)/len(test_wer) * 100

print('Average CER: {:2f}% Average WER: {:.2f}%\n'.format(avg_cer, avg_wer))
