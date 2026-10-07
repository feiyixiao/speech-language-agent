# Recordings for the speech eval (stay local, not in git)

1. Open `READING_SHEET.md`. Read each sentence EXACTLY as written, mistakes included. Do not self-correct.
2. Save `g000.wav`, `g001.wav`, ... in this folder (m4a/mp3/flac/ogg/webm also work with Groq).
   Phone voice memo, then convert if needed: `ffmpeg -i in.m4a -ac 1 -ar 16000 g000.wav`.
3. Noisy condition: `python scripts/add_noise.py --snr 5 eval/audio` (needs .wav), or record again as `g000__noisy.wav`.
4. `python -m eval.speech_eval collect --condition clean` (then `--condition noisy`).
Recordings contain your voice, so `.gitignore` keeps everything here out of the repository except these two text files.
