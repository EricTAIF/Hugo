#!/usr/bin/env python3

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append('.')

from ui_functions import create_intelligent_piano_arrangement_with_ui_params

# Set up paths
midi_dir = Path("/home/hp/Documents/hugo/audio2notesheets/outputs/midi/Snow White - Laufey (Lyrics)")
output_midi = midi_dir / "intelligent_piano_v7_smooth.mid"

# Get existing MIDI files
midi_files = {}
for stem in ["vocals", "bass", "guitar", "piano", "other"]:
    midi_path = midi_dir / f"{stem}_basic_pitch.mid"
    if midi_path.exists():
        midi_files[stem] = midi_path
        print(f"Found {stem}: {midi_path}")

print(f"Creating intelligent arrangement v7 with smooth legato connections...")

# Parameters with enhanced smoothness
params = {
    'phraseDuration': 12000,
    'phraseOverlap': 2000,
    'minNotesPerSec': 4.0,
    'maxNotesPerSec': 8.0,
    'veryDenseRatio': 0.25,
    'denseRatio': 0.35,
    'mediumRatio': 0.50,
    'sparseRatio': 0.70,
    'veryShortPenalty': 20,
    'shortPenalty': 10,
    'minDuration': 150,
    'goodDuration': 500,
    'legatoEnabled': True,
    'legatoGapThreshold': 400,  # More generous legato connections
    'minNoteDuration': 250,     # Longer minimum durations
    'sustainedHarmony': True,
    'vocalsWeight': 1.0,
    'pianoWeight': 0.8,
    'guitarWeight': 0.6,
    'bassWeight': 0.7
}

result = create_intelligent_piano_arrangement_with_ui_params(midi_files, output_midi, params)
print(f"Done! Generated {result['noteCount']} notes with smooth legato")
print(f"Saved to: {output_midi}")