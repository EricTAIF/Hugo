# Separation + Transcription Toolkit

This adds **stem separation** + **MIDI transcription** to your earlier pipeline.

## Install (venv recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements-separation.txt
```

> Demucs may require a Torch build for your platform/GPU. If pip doesn't pull it automatically, install one from https://pytorch.org/get-started/locally/ then re-run `pip install demucs`.

Install FFmpeg if you haven't:
- Ubuntu/Debian: `sudo apt-get install -y ffmpeg`
- macOS (Homebrew): `brew install ffmpeg`
- Windows (Chocolatey): `choco install ffmpeg`

## Separate a song and transcribe stems to MIDI
```bash
# Demucs (default)
python separate_song.py "path/to/song.mp3" --out outputs --engine demucs --model htdemucs --device cpu

# Demucs 6-stem (has guitar/piano stems)
python separate_song.py "song.mp3" --out outputs --engine demucs --model htdemucs_6s --device cpu

# Spleeter (5 stems: vocals, drums, bass, piano, other)
python separate_song.py "song.mp3" --out outputs --engine spleeter --stems 5
```

Outputs:
- Separated audio: `outputs/<engine or model>/<song-stem>/...wav/mp3`
- Transcribed MIDIs: `outputs/midi/<song-stem>/*_basic_pitch.mid`

You can then feed `*_basic_pitch.mid` files into your arrangement/UI.
