"""Make a noisy copy of clean 16-bit WAV recordings (stdlib only).

    python scripts/add_noise.py --snr 5 eval/audio          # writes <id>__noisy.wav next to each <id>.wav
Convert other formats first:  ffmpeg -i in.m4a -ac 1 -ar 16000 out.wav
SNR (dB) is signal RMS over noise RMS; 20 = barely audible hiss, 5 = hard, 0 = noise as loud as speech.
White noise is a crude stand-in for a real cafe; record a real noisy set if you can.
"""
import argparse
import array
import math
import random
import wave
from pathlib import Path


def add_noise(src: Path, dst: Path, snr_db: float, seed: int = 0) -> None:
    with wave.open(str(src), "rb") as w:
        params, frames = w.getparams(), w.readframes(w.getnframes())
    if params.sampwidth != 2:
        raise SystemExit(f"{src}: only 16-bit PCM WAV is supported")
    x = array.array("h")
    x.frombytes(frames)
    rms = math.sqrt(sum(v * v for v in x) / max(len(x), 1)) or 1.0
    sigma = rms / (10 ** (snr_db / 20))
    rng = random.Random(seed)
    y = array.array("h", (max(-32768, min(32767, int(v + rng.gauss(0, sigma)))) for v in x))
    with wave.open(str(dst), "wb") as w:
        w.setparams(params)
        w.writeframes(y.tobytes())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--snr", type=float, default=5.0)
    a = ap.parse_args()
    for p in sorted(Path(a.folder).glob("*.wav")):
        if p.stem.endswith("__noisy"):
            continue
        add_noise(p, p.with_name(p.stem + "__noisy.wav"), a.snr)
        print("wrote", p.stem + "__noisy.wav")
