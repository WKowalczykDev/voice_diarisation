import torch
import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition

model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa"
)

TARGET_SR = 16000

def get_embedding(path):
    waveform, sr = torchaudio.load(path)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    return model.encode_batch(waveform)

def compare(emb1, emb2):
    return float(torch.nn.functional.cosine_similarity(emb1, emb2, dim=-1).mean())