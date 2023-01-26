import os
from functools import lru_cache
from typing import Union

import ffmpeg
import numpy as np
import torch
import torch.nn.functional as F

from .utils import exact_div

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000: number of samples in a chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000: number of frames in a mel spectrogram input


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(dim=axis, index=torch.arange(length, device=array.device))

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    with np.load(os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(audio: Union[str, np.ndarray, torch.Tensor], n_mels: int = N_MELS):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    # print(f"{audio.shape=} {audio[100:120]=}")
    # raise Exception
    # audio = audio.to('cuda')
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)

    dft_mat = torch.fft.fft(torch.eye(N_FFT, dtype=torch.float64), dim=-1).to(dtype=torch.cfloat, device=audio.device)
    dft_mat_real = dft_mat.real
    dft_mat_imag = dft_mat.imag

    def spectify(input):
        signal_dim = input.dim()
        extended_shape = [1] * (3 - signal_dim) + list(input.size())
        pad = int(N_FFT // 2)
        print(f"{input.shape=} {signal_dim=} {extended_shape=} {pad=}")
        input = torch.nn.functional.pad(audio.view(extended_shape), [pad, pad], 'reflect')
        input = input.view(input.shape[-signal_dim:])

        windowed = input.unfold(dimension=0, size=N_FFT, step=HOP_LENGTH) * window
        ans = torch.complex(windowed, torch.zeros_like(windowed)) @ dft_mat
        return ans[:, :201].T

    # print(f"{(stft == stft_).all()=} {torch.allclose(stft, stft_)=} {max_diff=}")
    # assert torch.allclose(stft, stft_)
    # print(f"{audio.shape} {stft.shape=} {dft_mat.shape=} {windowed.shape=} {window.shape=} {N_FFT=}, {HOP_LENGTH=}")
    # raise Exception

    filters = torch.tensor(mel_filters(audio.device, n_mels))

    def melify(stft):
        magnitudes = stft[:, :-1].abs() ** 2

        # filters = mel_filters(audio.device, n_mels)
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def unfold_input(input):
        signal_dim = input.dim()
        extended_shape = [1] * (3 - signal_dim) + list(input.size())
        pad = int(N_FFT // 2)
        print(f"{input.shape=} {signal_dim=} {extended_shape=} {pad=} {N_FFT=} {HOP_LENGTH=}")
        input = torch.nn.functional.pad(audio.view(extended_shape), [pad, pad], 'reflect')
        print(f"{input.shape[-signal_dim:]=} {signal_dim=} {input.shape=}")
        input = input.view(input.shape[-signal_dim:])
        result = input.unfold(dimension=0, size=N_FFT, step=HOP_LENGTH)
        print(f"{result.shape=} {input.shape}")
        return result
    
    def mel_spectify(input):

        windowed = input * window
        # ans = torch.complex(windowed, torch.zeros_like(windowed)) @ dft_mat
        ans_r = windowed @ dft_mat_real
        ans_i = windowed @ dft_mat_imag
        # maxdif_r = (ans.real - ans_r).abs().max()
        # maxdif_i = (ans.imag - ans_i).abs().max()
        # print(f"{ans.real.shape=} {maxdif_r=} {maxdif_i=} {ans_r.shape=}")
        # assert maxdif_r < 1e-5 # torch.allclose(ans.real, ans_r)
        # assert  maxdif_i < 1e-5 # torch.allclose(ans.imag, ans_i)
        # raise Exception
        magnitudes2 = torch.transpose(ans_r[:-1, :201], -1, -2)**2 + torch.transpose(ans_i[:-1, :201], -1, -2)**2
        magnitudes = magnitudes2
        # stft = ans[:, :201].T
        # magnitudes = stft[:, :-1].abs() ** 2
        # maxdif = (magnitudes - magnitudes2).abs().max()
        # print(f"{maxdif=}")
        # assert maxdif < 2e-5 # torch.equal(magnitudes, magnitudes2)
        # raise Exception

        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, torch.max(log_spec) - 8.0)

        # dif = (log_spec-log_spec_).abs().max()
        # print(f"{dif=} {log_spec.max()=} {log_spec.min()=} {dif.device=}")
        # assert torch.equal(log_spec, log_spec_)
        # raise Exception

        log_spec = (log_spec + 4.0) / 4.0
        return log_spec.half()

    # log_spec_ = melify(spectify(audio)).half()
    log_spec__ = mel_spectify(unfold_input(audio))
    # return log_spec__
    # maxdif = (log_spec_ - log_spec__).abs().max()
    # print(f"{maxdif=}")
    # assert maxdif < 1e-3
    # raise Exception

    # max_diff = (log_spec - log_spec_).abs().max()
    # print(f"{(log_spec == log_spec_).all()=} {torch.allclose(log_spec, log_spec_)=} {max_diff=}")
    # assert max_diff.item() < 4e-5

    # print(f"{audio.shape=} {log_spec.shape=}")
    # raise Exception

    # class wrapper_melspec(torch.nn.Module):
    #     def forward(self, a):
    #         return mel_spectify(a)

    # torch.onnx.export(
    #     wrapper_melspec(),
    #     unfold_input(audio),
    #     "melspec13.onnx",
    #     verbose=False,
    #     opset_version=13,
    #     input_names=["audio"],
    #     output_names=["mel_spectrogram"],
    #     dynamic_axes={
    #         "audio": [0],
    #         "mel_spectrogram": [1],
    #     }
    # )
    # raise Exception

    log_spec = melify(stft)
    print(f"{log_spec.shape=} {stft.shape=}")
    # raise Exception

    return log_spec__
    # return log_spec
