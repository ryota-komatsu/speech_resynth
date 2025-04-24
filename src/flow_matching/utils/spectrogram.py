# from https://github.com/jik876/hifi-gan/blob/master/meldataset.py

# MIT License
#
# Copyright (c) 2020 Jungil Kong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression_torch(x: torch.Tensor, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def mel_spectrogram(
    y: torch.FloatTensor,
    n_fft: int = 400,
    num_mels: int = 80,
    sampling_rate: int = 16000,
    hop_size: int = 320,
    fmin=0,
    fmax=8000,
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel_basis = torch.from_numpy(mel).float().to(y.device)
    hann_window = torch.hann_window(n_fft).to(y.device)

    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        window=hann_window,
        center=False,
        onesided=True,
        return_complex=True,
    )

    spec = spec.abs()
    spec = torch.matmul(mel_basis, spec)
    spec = dynamic_range_compression_torch(spec)

    return spec
