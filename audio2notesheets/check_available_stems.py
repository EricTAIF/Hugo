#!/usr/bin/env python3
"""
Check which stems are available for each song
"""

from pathlib import Path
import os

def check_available_stems():
    """Check which stems are available for each song"""
    
    outputs_dir = Path("outputs/midi")
    if not outputs_dir.exists():
        print("No outputs/midi directory found")
        return
    
    print("🎵 Available Stems by Song:")
    print("=" * 60)
    
    # Get all song directories
    song_dirs = [d for d in outputs_dir.iterdir() if d.is_dir()]
    
    if not song_dirs:
        print("No song directories found")
        return
    
    stem_types = ["vocals", "bass", "guitar", "piano", "other", "drums"]
    
    for song_dir in sorted(song_dirs):
        song_name = song_dir.name
        print(f"\n📁 {song_name}")
        
        available_stems = []
        for stem in stem_types:
            midi_file = song_dir / f"{stem}_basic_pitch.mid"
            if midi_file.exists():
                available_stems.append(stem)
        
        if available_stems:
            print(f"   ✅ Available stems: {', '.join(available_stems)}")
        else:
            print(f"   ❌ No MIDI stems found")
    
    print("\n" + "=" * 60)
    print("💡 To use single stem mode, select a song that has the stem you want!")

if __name__ == "__main__":
    check_available_stems()