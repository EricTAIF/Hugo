# Audio to Piano Toolkit with Separation

A comprehensive toolkit for separating audio stems and transcribing them to piano-playable MIDI files. This project combines stem separation using Demucs or Spleeter with advanced MIDI transcription and post-processing.

## Features

- **Audio Stem Separation**: Extract individual instruments (vocals, bass, drums, guitar, piano, etc.) from mixed audio tracks
- **MIDI Transcription**: Convert separated stems to MIDI using Basic Pitch with multiple backends
- **Piano Post-Processing**: Quantize, clean, and optimize MIDI for piano performance
- **Hand Splitting**: Automatically split piano parts into left and right hand voices
- **Multiple Engines**: Support for both Demucs and Spleeter separation engines
- **Flexible Output**: Raw MIDI, piano-optimized MIDI, and hand-split versions

## Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio processing)

#### Install FFmpeg:
- **Ubuntu/Debian**: `sudo apt-get install -y ffmpeg`
- **macOS (Homebrew)**: `brew install ffmpeg`
- **Windows (Chocolatey)**: `choco install ffmpeg`

### Setup

1. **Create and activate a virtual environment** (recommended):
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\Activate.ps1
```

2. **Install dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements-separation.txt
```

> **Note**: Demucs may require a specific PyTorch build for your platform/GPU. If pip doesn't install it automatically, get the appropriate version from [PyTorch's official site](https://pytorch.org/get-started/locally/) then re-run `pip install demucs`.

## Usage

### Basic Stem Separation and Transcription

Use the main CLI script to separate a song and transcribe all stems to MIDI:

> **🎤 New: Enhanced Vocal Processing!** Vocals now get special treatment with multiple processing modes to handle pitch bends and complex harmonies better on piano.

```bash
# Default Demucs separation (4 stems: vocals, bass, drums, other)
python separate_song.py "path/to/song.mp3" --out outputs --engine demucs --model htdemucs --device cpu

# Demucs 6-stem model (includes guitar and piano stems)
python separate_song.py "song.mp3" --out outputs --engine demucs --model htdemucs_6s --device cpu

# Spleeter separation (5 stems: vocals, drums, bass, piano, other)
python separate_song.py "song.mp3" --out outputs --engine spleeter --stems 5
```

### Individual Stem Transcription

To transcribe a single audio file to piano MIDI:

```bash
# Auto-detect instrument type
python audio_to_piano.py "stem.mp3" --out outputs

# Specify instrument for optimized processing
python audio_to_piano.py "guitar_stem.mp3" --out outputs --instrument guitar

# Piano-specific transcription
python audio_to_piano.py "piano_stem.mp3" --out outputs --instrument piano
```

### Enhanced Vocal Processing 🎤

For better vocal transcription, use the dedicated vocal enhancement tool:

```bash
# Process vocals with all modes (recommended)
python enhance_vocals.py "vocals.mp3" --out vocal_output

# Focus on main melody only
python enhance_vocals.py "vocals.mp3" --mode lead --out vocal_output

# Preserve complex harmonies (Jacob Collier style)
python enhance_vocals.py "complex_vocals.mp3" --mode harmony --max-voices 6

# Compare different vocal processing modes
python compare_vocal_modes.py vocal_output/
```

#### Vocal Processing Modes:
- **Clean**: Removes pitch bends, focuses on main melody, piano-friendly
- **Harmony**: Preserves multiple voices for complex arrangements 
- **Lead**: Extracts only the main melody line
- **Original**: Keeps pitch bends for comparison with raw transcription

### Available Instruments

- `auto` - General polyphonic transcription (default)
- `guitar` - Optimized for guitar stems
- `piano` - Piano-specific transcription with pedal detection
- `monophonic` - Single-note instruments
- `vocals` - Voice transcription
- `flute`, `violin`, `sax` - Monophonic wind/string instruments

## Output Structure

After processing, you'll find:

```
outputs/
├── htdemucs/                    # Demucs separated stems
│   └── song_name/
│       ├── vocals.mp3
│       ├── bass.mp3
│       ├── drums.mp3
│       └── other.mp3
├── htdemucs_6s/                 # 6-stem Demucs output
│   └── song_name/
│       ├── vocals.mp3
│       ├── bass.mp3
│       ├── drums.mp3
│       ├── guitar.mp3
│       ├── piano.mp3
│       └── other.mp3
└── midi/                        # Transcribed MIDI files
    └── song_name/
        ├── vocals.vocals.clean.mid      # 🎤 Clean vocals (no pitch bends)
        ├── vocals.vocals.harmony.mid    # 🎤 Harmony vocals (multi-voice)
        ├── vocals.vocals.lead.mid       # 🎤 Lead vocals (melody only)
        ├── vocals.vocals.original.mid   # 🎤 Original vocals (with pitch bends)
        ├── vocals_basic_pitch.mid       # Raw Basic Pitch output
        ├── guitar.piano.mid             # Piano-optimized version
        ├── guitar.piano.LR.mid          # Left/Right hand split
        └── ...                          # Same for each non-vocal stem
```

## Command Line Options

### separate_song.py
- `--out`: Output directory (default: `outputs`)
- `--engine`: Separation engine (`demucs`, `spleeter`, or model name)
- `--device`: Processing device (`cpu`, `cuda`)
- `--model`: Demucs model (`htdemucs`, `htdemucs_6s`)
- `--stems`: Number of Spleeter stems (2, 4, or 5)

### audio_to_piano.py
- `--out`: Output directory (default: `out`)
- `--instrument`: Instrument type for optimized processing
- `--grid`: Beat subdivisions (2=eighths, 4=sixteenths)
- `--max-rh-voices`: Limit right hand voices per beat
- `--min-note-ms`: Minimum note duration in milliseconds
- `--nosplit`: Skip left/right hand splitting

## Technical Details

### Separation Engines

- **Demucs**: State-of-the-art neural source separation
  - `htdemucs`: 4-stem model (vocals, bass, drums, other)
  - `htdemucs_6s`: 6-stem model (adds guitar and piano)
- **Spleeter**: Deezer's open-source separation (requires separate environment)

### Transcription Pipeline

1. **Basic Pitch**: Neural polyphonic transcription
2. **Beat Detection**: Librosa-based tempo and beat tracking
3. **Quantization**: Snap notes to musical grid
4. **Piano Optimization**: Remove pitch bends, set piano program
5. **Hand Splitting**: Intelligent left/right hand voice separation

### Post-Processing Features

- Note duration filtering
- Pitch range clamping (A0-C8)
- Duplicate note merging
- Sustain pedal simulation for left hand
- Voice limiting to prevent dense chords

## Troubleshooting

### Common Issues

1. **Import errors with typing-extensions**: Demucs requires newer versions
   ```bash
   pip install -U typing-extensions==4.12.2
   ```

2. **Spleeter conflicts**: Use separate virtual environment for Spleeter
   ```bash
   python -m venv spleeter_env
   source spleeter_env/bin/activate
   pip install spleeter tensorflow==2.12.1
   ```

3. **CUDA/GPU issues**: Use `--device cpu` for CPU-only processing

4. **FFmpeg not found**: Ensure FFmpeg is installed and in your PATH

## Dependencies

Core libraries:
- `demucs>=4.0.0` - Neural source separation
- `basic-pitch>=0.2.0` - Polyphonic transcription
- `pretty_midi>=0.2.10` - MIDI manipulation
- `librosa>=0.10.1` - Audio analysis
- `numpy>=1.23` - Numerical computing

Optional:
- `spleeter>=2.4.0` - Alternative separation engine
- `music21>=9.0.0` - Score generation

## License

This project uses open-source libraries. Please check individual package licenses for commercial use.

## Contributing

Feel free to submit issues and pull requests for improvements and bug fixes.