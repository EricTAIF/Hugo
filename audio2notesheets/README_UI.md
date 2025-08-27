# 🎹 Intelligent Piano Arrangement Tuner

This tool provides both CLI and web UI interfaces for creating intelligent piano arrangements with customizable parameters.

## Features

- **Web UI**: Interactive interface with sliders and real-time parameter adjustment
- **CLI Integration**: Command-line arguments for batch processing and automation
- **Presets**: Quick presets for minimal, balanced, rich, and full arrangements
- **Real-time Generation**: Generate arrangements with custom parameters instantly

## Usage

### 1. Web UI (Recommended for Parameter Tuning)

Start the web server:
```bash
python piano_ui_server.py
```

Then open your browser to: http://localhost:5000

The web UI provides:
- 🎛️ **Quick Presets**: Minimal, Balanced, Rich, Full arrangements
- ⏱️ **Phrase Detection**: Control phrase duration and overlap
- 🎵 **Note Density**: Set min/max notes per second
- 🎯 **Note Selection**: Adjust ratios for different phrase densities
- ⏰ **Duration Filtering**: Penalize very short notes
- 🎼 **Part Weights**: Control importance of vocals, piano, guitar, bass

### 2. CLI (For Automation and Batch Processing)

Basic usage with intelligent piano arrangement:
```bash
python main.py "song.mp3" --intelligent-piano
```

With custom parameters:
```bash
python main.py "song.mp3" --intelligent-piano \
  --phrase-duration 10 \
  --min-notes-per-sec 3 \
  --max-notes-per-sec 6 \
  --dense-ratio 0.4 \
  --medium-ratio 0.6
```

Using a parameter file:
```bash
python main.py "song.mp3" --intelligent-piano --piano-params example_piano_params.json
```

### 3. CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--intelligent-piano` | - | Enable intelligent piano arrangement |
| `--piano-params` | - | JSON file with parameters |
| `--phrase-duration` | 12 | Phrase duration in seconds |
| `--phrase-overlap` | 2 | Phrase overlap in seconds |
| `--min-notes-per-sec` | 4.0 | Minimum notes per second |
| `--max-notes-per-sec` | 8.0 | Maximum notes per second |
| `--dense-ratio` | 0.35 | Note ratio for dense phrases |
| `--medium-ratio` | 0.50 | Note ratio for medium phrases |

## Parameter Examples

### Minimal Arrangement (200-400 notes)
```bash
python main.py "song.mp3" --intelligent-piano \
  --phrase-duration 15 \
  --min-notes-per-sec 2 \
  --max-notes-per-sec 4 \
  --dense-ratio 0.2 \
  --medium-ratio 0.3
```

### Rich Arrangement (800-1200 notes)
```bash
python main.py "song.mp3" --intelligent-piano \
  --phrase-duration 10 \
  --min-notes-per-sec 6 \
  --max-notes-per-sec 12 \
  --dense-ratio 0.55 \
  --medium-ratio 0.7
```

## Output Files

The tool generates intelligent piano arrangements with filenames like:
- CLI: `{song_name}_intelligent_piano.mid`
- UI: `intelligent_piano_ui_{hash}.mid`

## Requirements

- Python 3.7+
- Flask (for web UI): `pip install flask`
- All dependencies from main audio2notesheets project

## Tips for Best Results

1. **Start with Balanced preset** in the web UI to get familiar with parameters
2. **Use CLI for batch processing** multiple songs with the same parameters
3. **Adjust duration filtering** to eliminate unwanted short notes
4. **Tune phrase detection** based on song structure (shorter phrases for complex songs)
5. **Balance note density** - aim for 4-8 notes per second for playable arrangements