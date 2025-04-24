import json
from pathlib import Path

import librosa
import torch
import torchaudio
from tqdm import tqdm

from ..hifigan.data import mel_spectrogram
from .data import LibriTTS_R
from .utils.textless import load_encoder


def preprocess(config):
    resample(config)
    tokenize(config)
    extract_features(config)


def resample(config):
    wav_dir_orig = Path(config.dataset.wav_dir_orig)
    wav_dir = Path(config.dataset.wav_dir)
    wav_paths = list(wav_dir_orig.glob("**/*" + config.dataset.ext_audio))

    for wav_path in tqdm(wav_paths):
        wav_name = wav_path.relative_to(wav_dir_orig)
        wav_path = str(wav_path)

        wav, sr = torchaudio.load(wav_path)
        wav = torchaudio.functional.resample(wav, sr, 16000)

        if config.dataset.vad:
            wav = wav.numpy()
            wav, _ = librosa.effects.trim(wav, top_db=20)
            wav = torch.from_numpy(wav)

        wav_path = wav_dir / wav_name
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        wav_path = str(wav_path)  # for sox backend
        torchaudio.save(wav_path, wav, 16000)


def tokenize(config):
    trainset = LibriTTS_R(config.dataset.wav_dir, split="train-*")
    devset = LibriTTS_R(config.dataset.wav_dir, config.dataset.wav_dir_orig, split="dev-clean")
    testset = LibriTTS_R(config.dataset.wav_dir, config.dataset.wav_dir_orig, split="test-clean")

    train_loader = torch.utils.data.DataLoader(trainset)
    dev_loader = torch.utils.data.DataLoader(devset)
    test_loader = torch.utils.data.DataLoader(testset)

    encoder = load_encoder(
        config.flow_matching.dense_model_name,
        config.flow_matching.quantizer_model_name,
        config.flow_matching.vocab_size,
        config.flow_matching.predict_duration,
    )

    _tokenize(encoder, config.dataset.train_file, train_loader)
    _tokenize(encoder, config.dataset.dev_file, dev_loader)
    _tokenize(encoder, config.dataset.test_file, test_loader)


def _tokenize(encoder, file, dataloader: torch.utils.data.DataLoader):
    dataset = dict()

    for item in tqdm(dataloader):
        outputs = encoder(item["input_values"].cuda())
        units = outputs["units"].tolist()
        durations = outputs["durations"].tolist()

        dataset[item["name"][0]] = {"units": units, "durations": durations, "transcript": item["transcript"][0]}

    Path(file).parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w") as f:
        json.dump(dataset, f)


def extract_features(config):
    wav_dir = Path(config.dataset.wav_dir)
    spectrogram_dir = Path(config.dataset.spectrogram_dir)
    wav_paths = list(wav_dir.glob("**/*" + config.dataset.ext_audio))

    for wav_path in tqdm(wav_paths):
        wav_name = wav_path.relative_to(wav_dir).with_suffix("")
        spectrogram_path = spectrogram_dir / wav_name.with_suffix(".pt")
        if spectrogram_path.is_file():
            continue
        spectrogram_path.parent.mkdir(parents=True, exist_ok=True)

        wav_path = str(wav_path)
        wav, sr = torchaudio.load(wav_path)
        wav = wav.cuda()
        wav = wav / wav.abs().max() * 0.95

        spectrogram_labels = mel_spectrogram(wav)  # (1, 80, len)
        spectrogram_labels = spectrogram_labels.transpose(1, 2)  # (1, len, 80)
        spectrogram_labels = spectrogram_labels.cpu()

        torch.save(spectrogram_labels, spectrogram_path)


def tokenize_librilight(config):
    wav_dir = Path(config.dataset.wav_dir)
    wav_dir_orig = Path(config.dataset.wav_dir_orig)

    paths = wav_dir_orig.glob("**/*" + config.dataset.ext_audio)
    paths = list(paths)

    encoder = load_encoder(
        config.flow_matching.dense_model_name,
        config.flow_matching.quantizer_model_name,
        config.flow_matching.vocab_size,
        config.flow_matching.predict_duration,
    )

    dataset = dict()

    for path in tqdm(paths):
        wav, sr = torchaudio.load(path)
        wavs = torch.split(wav, 400080, dim=1)
        name = path.relative_to(wav_dir_orig).with_suffix("")

        for idx, wav in enumerate(wavs):
            path = wav_dir / f"{name}_{idx:04}{config.dataset.ext_audio}"
            path.parent.mkdir(exist_ok=True, parents=True)
            path = str(path)
            torchaudio.save(path, wav, sr)

            try:
                outputs = encoder(wav.cuda())
                units = outputs["units"].tolist()

                dataset[path] = units
            except RuntimeError:
                pass

    with open(config.dataset.train_file, "w") as f:
        json.dump(dataset, f)
