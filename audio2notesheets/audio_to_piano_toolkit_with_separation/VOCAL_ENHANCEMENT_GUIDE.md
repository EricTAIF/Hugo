# 🎤 Enhanced Vocal Processing Guide

## Overview

The original vocal transcriptions had issues with pitch bends that don't translate well to piano. This enhancement adds multiple vocal processing modes specifically designed to create better piano-playable MIDI from vocal stems.

## The Problem with Original Vocals

- **Pitch bends**: Vocals naturally use continuous pitch (vibrato, slides) that creates MIDI pitch bend data
- **Piano limitation**: Pianos can't reproduce pitch bends smoothly, making vocal MIDI sound strange
- **Complex harmonies**: Raw transcription often captures too much detail, making it hard to play

## New Vocal Processing Modes

### 1. **Clean Mode** 🧽
- **Best for**: General piano performance
- **Features**: 
  - Removes all pitch bends
  - Filters to comfortable vocal range (C3-C6)
  - Reduces polyphony to max 2 voices
  - Minimum note duration: 100ms
- **Use case**: Playing vocal melodies on piano

### 2. **Harmony Mode** 🎵
- **Best for**: Complex vocal arrangements (Jacob Collier style)
- **Features**:
  - Preserves multiple voices (up to 6 configurable)
  - Wider vocal range (A2-C6) 
  - Intelligent voice grouping by time windows
  - Harmonic importance scoring
- **Use case**: Jazz arrangements, choir parts, complex harmonies

### 3. **Lead Mode** 🎯
- **Best for**: Simple melody extraction
- **Features**:
  - Extracts only the main melody line
  - Takes highest pitch in each time window
  - Main vocal range (C3-C6)
  - Minimum note duration: 120ms
- **Use case**: Lead sheets, simple melodies

### 4. **Original Mode** 📝
- **Best for**: Comparison and reference
- **Features**:
  - Keeps original pitch bends
  - Quantized to musical grid
  - All original notes preserved
- **Use case**: Comparing against processed versions

## Test Results

### Snow White - Laufey (Complex vocal style)
- **Original**: 491 notes, D1-A#5, has pitch bends
- **Clean**: 438 notes, E3-A#5, no pitch bends ✅
- **Harmony**: 464 notes, D2-A#5, preserves voices ✅  
- **Lead**: 349 notes, E3-A#5, melody only ✅

### Blackbird - The Beatles (Simpler vocal style)
- **Original**: 211 notes, D#1-C#6, has pitch bends
- **Clean**: 186 notes, D3-G4, no pitch bends ✅
- **Harmony**: 206 notes, A2-C#6, preserves harmonies ✅
- **Lead**: 185 notes, D3-G4, clean melody ✅

## Usage Examples

```bash
# Process any vocal stem with all modes
python enhance_vocals.py "vocals.mp3" --out vocal_output

# For Jacob Collier-style complex harmonies
python enhance_vocals.py "complex_vocals.mp3" --mode harmony --max-voices 6

# For simple melody extraction
python enhance_vocals.py "vocals.mp3" --mode lead

# Compare the different modes
python compare_vocal_modes.py vocal_output/
```

## Integration with Main Pipeline

When using `separate_song.py`, vocals are automatically processed with enhanced modes:
- Regular stems get standard processing
- Vocal stems get 4 different MIDI versions automatically
- All modes are saved for comparison

## Recommendations

| Vocal Style | Recommended Mode | Max Voices |
|-------------|------------------|------------|
| Pop/Rock vocals | Clean | 2 |
| Jazz standards | Harmony | 4 |
| Complex arrangements | Harmony | 6+ |
| Simple melodies | Lead | 1 |
| Background vocals | Harmony | 3-4 |

## Technical Notes

- **Time windows**: Harmony mode uses 25ms windows for precise voice separation
- **Harmonic scoring**: Notes are ranked by pitch, velocity, and harmonic importance
- **Range filtering**: Each mode has optimized pitch ranges for playability
- **Duration filtering**: Removes very short notes that don't work well on piano

The enhanced vocal processing makes vocal MIDI much more piano-friendly while preserving the musical intent of the original performance!