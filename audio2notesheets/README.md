# audio2notesheets

Turn any MP3 into **separated stems** and **printable sheet music (MusicXML)** with one command.

**Pipeline**  
1) Source separation → [Demucs htdemucs] splits into vocals, drums, bass, other.  
2) Transcription → [Spotify Basic Pitch] turns *pitched* stems (vocals, bass, other) into MIDI.  
3) Score assembly → [music21] merges MIDI tracks and writes **MusicXML** (open in MuseScore/Sibelius/Dorico/Finale).

> ⚠️ Quality depends a lot on the audio and genre. Polyphonic accompaniments may be imperfect; vocals with heavy vibrato can create dense notation. You can always edit the generated MusicXML in a notation editor.

## Quick start

### 0) System deps
- Python 3.9–3.11 recommended.
- **ffmpeg** + **libsndfile** (audio I/O)
- **MuseScore** (optional) to view/print the MusicXML or export PDF.

On macOS (Homebrew):
```bash
brew install ffmpeg libsndfile
```

On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1
```

### 1) Create & activate a virtualenv (recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2) Install Python packages
**Install a matching PyTorch first** (CPU is fine). Choose the command for your OS from https://pytorch.org/get-started/locally/ . Example (CPU only):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
Then:
```bash
pip install -r requirements.txt
```

> If Basic Pitch asks for TensorFlow on your setup, install it with:  
> `pip install basic-pitch[tf]`

### 3) Run
```bash
python main.py /path/to/song.mp3 -o out_dir
```
Outputs:
- `separated/…/*.wav` stems from Demucs
- `midi/…/*.mid` per-stem MIDI from Basic Pitch
- `score/combined.musicxml` multi-staff printable score
- `score/combined.mid` merged MIDI

Options:
```bash
python main.py INPUT_AUDIO [-o OUT_DIR] [--device cpu|cuda] [--model htdemucs|htdemucs_ft|hdemucs_mmi|htdemucs_6s]
                           [--include-other yes|no] [--tempo BPM] [--time-signature TS]
                           [--skip-separation] [--skip-transcription]
```
Examples:
```bash
# CPU, exclude the "other" stem from notation to reduce clutter
python main.py song.mp3 --device cpu --include-other no

# Use the fine-tuned Demucs model (slower, a bit better)
python main.py song.mp3 --model htdemucs_ft
```

### 4) View/print
Open `score/combined.musicxml` in MuseScore and export to PDF.

---

## How it works (libs)
- **Demucs** (MIT) separates songs into vocals/drums/bass/other and more (v4 “Hybrid Transformer Demucs”).  
- **Basic Pitch** (Apache-2.0) transcribes audio (mono/stereo, polyphonic) to MIDI.  
- **music21** writes **MusicXML** and merges tracks into a printable score.

## Legal note (not legal advice)
If you use copyrighted songs, make sure you have the rights or local exceptions (like private study) in your jurisdiction.

## Troubleshooting
- If Demucs complains about CUDA memory, run with `--device cpu`.
- If Basic Pitch output is too detailed (vibrato), edit the MusicXML in your notation app or run Basic Pitch on simpler stems (e.g., vocals only).
- If MuseScore is installed but `music21` can’t find it, run:
```python
from music21 import environment
us = environment.UserSettings()
us['musicxmlPath'] = '/path/to/MuseScore4'  # MuseScore executable
```
