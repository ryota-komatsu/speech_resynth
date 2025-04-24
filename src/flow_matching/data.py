import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        wav_dir: str,
        txt_dir: Optional[str] = None,
        split: str = "train-*",
        ext_audio: str = ".flac",
        ext_txt: Optional[str] = None,
    ):
        self.wav_dir = Path(wav_dir)
        self.txt_dir = Path(txt_dir) if txt_dir is not None else self.wav_dir
        self.wav_paths = sorted(self.wav_dir.glob(f"{split}/**/*" + ext_audio))

        self.ext_audio = ext_audio
        self.ext_txt = ext_txt

    def __len__(self) -> int:
        return len(self.wav_paths)

    def __getitem__(self, n: int) -> Dict[str, Any]:
        wav_path = self.wav_paths[n]
        wav_name = wav_path.relative_to(self.wav_dir)
        wav_name = wav_name.with_suffix("")
        wav_name = str(wav_name)
        wav_path = str(wav_path)

        input_values, sr = torchaudio.load(wav_path)
        input_values = torchaudio.functional.resample(input_values, sr, 16000)
        input_values = input_values.squeeze(0)

        return {"input_values": input_values, "name": wav_name}

    @staticmethod
    def collate_fn(batch):
        input_values = [item["input_values"] for item in batch]
        attention_mask = [torch.ones_like(item["input_values"], dtype=torch.long) for item in batch]
        names = [item["name"] for item in batch]

        input_values = pad_sequence(input_values, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        wavs_len = torch.tensor([len(item["input_values"]) for item in batch])

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "wavs_len": wavs_len,
            "padding_mask": ~attention_mask.bool(),
            "names": names,
        }


class LibriTTS_R(SpeechDataset):
    def __init__(
        self,
        wav_dir,
        txt_dir=None,
        split: str = "train-*",
        ext_audio: str = ".wav",
        ext_txt: Optional[str] = ".normalized.txt",
    ):
        super().__init__(wav_dir, txt_dir, split, ext_audio, ext_txt)

    def __getitem__(self, n: int) -> Dict[str, Any]:
        item = super().__getitem__(n)

        txt_path = self.txt_dir / item["name"]
        txt_path = txt_path.with_suffix(".normalized.txt")

        transcript = ""
        if txt_path.is_file():
            with open(txt_path) as g:
                transcript = g.read().rstrip()

        item["transcript"] = transcript

        return item


class LibriSpeech(SpeechDataset):
    def __getitem__(self, n: int) -> Dict[str, Any]:
        item = super().__getitem__(n)

        # transcript
        split, speaker_id, chap_id, utterance_id = item["name"].split("/")
        file = self.txt_dir / split / speaker_id / chap_id / f"{speaker_id}-{chap_id}.trans.txt"

        with open(file) as f:
            for line in f:
                id, transcript = line.rstrip().split(" ", maxsplit=1)
                if id == utterance_id:
                    break

        item["transcript"] = transcript

        return item


class UnitDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file,
        wav_dir=None,
        spectrogram_dir=None,
        frames_per_seg: Optional[int] = None,
        ext_audio: str = ".wav",
    ):
        self.input_ids = []
        self.spectrogram_labels = []
        self.duration_labels = []
        self.transcripts = []
        self.names = []
        self.input_values = []

        with open(file) as f:
            dataset = json.load(f)

        for name, value in tqdm(dataset.items()):
            units, duration_labels, transcript = value["units"], value["durations"], value["transcript"]

            input_ids = torch.tensor(units) + 1  # 0: pad
            duration_labels = torch.tensor(duration_labels)

            if spectrogram_dir is not None:
                spectrogram_path = os.path.join(spectrogram_dir, name + ".pt")
                spectrogram_labels = torch.load(spectrogram_path, "cpu", weights_only=True)
                spectrogram_labels = spectrogram_labels.squeeze(0)  # (len, 80)
            else:
                spectrogram_labels = torch.zeros(1, 80)  # dummy

            if wav_dir is not None:
                wav_path = os.path.join(wav_dir, name + ext_audio)
                input_values, sr = torchaudio.load(wav_path)
                input_values = input_values.squeeze(0)  # (len,)
            else:
                input_values = torch.zeros(1)  # dummy

            self.input_ids.append(input_ids)
            self.spectrogram_labels.append(spectrogram_labels)
            self.duration_labels.append(duration_labels)
            self.transcripts.append(transcript)
            self.names.append(name)
            self.input_values.append(input_values)

        self.frames_per_seg = frames_per_seg

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, n: int) -> Dict[str, Any]:
        input_ids = self.input_ids[n]
        spectrogram_labels = self.spectrogram_labels[n]
        duration_labels = self.duration_labels[n]
        transcripts = self.transcripts[n]
        names = self.names[n]
        input_values = self.input_values[n]

        if self.frames_per_seg is not None:
            diff = len(input_ids) - self.frames_per_seg

            if diff > 0:
                start = random.randrange(diff)
                input_ids = input_ids[start : start + self.frames_per_seg]
                spectrogram_labels = spectrogram_labels[start : start + self.frames_per_seg]
                duration_labels = duration_labels[start : start + self.frames_per_seg]
            else:
                input_ids = torch.nn.functional.pad(input_ids, (0, -diff))
                spectrogram_labels = torch.nn.functional.pad(spectrogram_labels, (0, 0, 0, -diff), value=-100)
                duration_labels = torch.nn.functional.pad(duration_labels, (0, -diff))

        return {
            "input_ids": input_ids,
            "spectrogram_labels": spectrogram_labels,
            "duration_labels": duration_labels,
            "transcripts": transcripts,
            "names": names,
            "input_values": input_values,
        }

    @staticmethod
    def collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        spectrogram_labels = [item["spectrogram_labels"] for item in batch]
        duration_labels = [item["duration_labels"] for item in batch]
        transcripts = [item["transcripts"] for item in batch]
        names = [item["names"] for item in batch]
        input_values = [item["input_values"].unsqueeze(0) for item in batch]

        input_ids = pad_sequence(input_ids, batch_first=True)
        spectrogram_labels = pad_sequence(spectrogram_labels, batch_first=True, padding_value=-100)
        duration_labels = pad_sequence(duration_labels, batch_first=True)

        return {
            "input_ids": input_ids,
            "spectrogram_labels": spectrogram_labels,
            "duration_labels": duration_labels,
            "transcripts": transcripts,
            "names": names,
            "input_values": input_values,
        }


class LibriLight(torch.utils.data.Dataset):
    def __init__(
        self,
        file,
        frames_per_seg: Optional[int] = None,
    ):
        self.input_ids = []
        self.paths = []

        with open(file) as f:
            dataset = json.load(f)

        for path, units in tqdm(dataset.items()):
            input_ids = torch.tensor(units) + 1  # 0: pad

            self.input_ids.append(input_ids)
            self.paths.append(path)

        self.frames_per_seg = frames_per_seg
        self.segment_size = (frames_per_seg - 1) * 320 + 400

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, n: int) -> Dict[str, torch.Tensor]:
        input_ids = self.input_ids[n]
        path = self.paths[n]

        input_values, sr = torchaudio.load(path)
        input_values = input_values.squeeze(0)  # (len,)

        diff = len(input_ids) - self.frames_per_seg
        if diff < 0:
            input_ids = torch.nn.functional.pad(input_ids, (0, -diff))

        diff = len(input_values) - self.segment_size
        if diff < 0:
            input_values = torch.nn.functional.pad(input_values, (0, -diff))

        return {
            "input_ids": input_ids,
            "input_values": input_values,
        }
