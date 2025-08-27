#!/usr/bin/env python3
"""
Compare different vocal processing modes to see the differences.
This helps understand which mode works best for different vocal styles.

Usage:
  python compare_vocal_modes.py vocal_enhanced_test/
  python compare_vocal_modes.py blackbird_vocals/
"""

import argparse
from pathlib import Path
import pretty_midi as pm

def analyze_midi_file(midi_path: Path):
    """Analyze a MIDI file and return statistics."""
    try:
        midi = pm.PrettyMIDI(str(midi_path))
        
        total_notes = 0
        pitch_range = []
        duration_stats = []
        has_pitch_bends = False
        
        for inst in midi.instruments:
            total_notes += len(inst.notes)
            has_pitch_bends = has_pitch_bends or len(inst.pitch_bends) > 0
            
            for note in inst.notes:
                pitch_range.append(note.pitch)
                duration_stats.append((note.end - note.start) * 1000)  # in ms
        
        if pitch_range:
            min_pitch = min(pitch_range)
            max_pitch = max(pitch_range)
            avg_duration = sum(duration_stats) / len(duration_stats)
            min_duration = min(duration_stats)
            max_duration = max(duration_stats)
        else:
            min_pitch = max_pitch = avg_duration = min_duration = max_duration = 0
        
        return {
            'total_notes': total_notes,
            'pitch_range': (min_pitch, max_pitch),
            'avg_duration_ms': avg_duration,
            'min_duration_ms': min_duration,
            'max_duration_ms': max_duration,
            'has_pitch_bends': has_pitch_bends,
            'total_duration': midi.get_end_time()
        }
    except Exception as e:
        return {'error': str(e)}

def compare_vocal_modes(output_dir: Path):
    """Compare all vocal processing modes in a directory."""
    output_dir = Path(output_dir).resolve()
    
    # Find vocal MIDI files
    vocal_files = {
        'Raw Basic Pitch': list(output_dir.glob("*_basic_pitch.mid")),
        'Clean (No Pitch Bends)': list(output_dir.glob("*.vocals.clean.mid")),
        'Harmony (Multi-Voice)': list(output_dir.glob("*.vocals.harmony.mid")),
        'Lead (Melody Only)': list(output_dir.glob("*.vocals.lead.mid")),
        'Original (With Pitch Bends)': list(output_dir.glob("*.vocals.original.mid"))
    }
    
    print(f"\\n=== Vocal Mode Comparison: {output_dir.name} ===\\n")
    
    for mode_name, files in vocal_files.items():
        if not files:
            print(f"❌ {mode_name}: No files found")
            continue
            
        print(f"🎵 {mode_name}:")
        for file in files:
            stats = analyze_midi_file(file)
            
            if 'error' in stats:
                print(f"   ❌ {file.name}: Error - {stats['error']}")
                continue
            
            # Convert MIDI note numbers to note names for readability
            def midi_to_note(midi_num):
                notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                octave = midi_num // 12 - 1
                note = notes[midi_num % 12]
                return f"{note}{octave}"
            
            min_note = midi_to_note(stats['pitch_range'][0]) if stats['pitch_range'][0] > 0 else "N/A"
            max_note = midi_to_note(stats['pitch_range'][1]) if stats['pitch_range'][1] > 0 else "N/A"
            
            print(f"   📁 {file.name}")
            print(f"      Notes: {stats['total_notes']}")
            print(f"      Range: {min_note} - {max_note}")
            print(f"      Avg Duration: {stats['avg_duration_ms']:.1f}ms")
            print(f"      Duration Range: {stats['min_duration_ms']:.1f}ms - {stats['max_duration_ms']:.1f}ms")
            print(f"      Total Length: {stats['total_duration']:.1f}s")
            print(f"      Has Pitch Bends: {'Yes' if stats['has_pitch_bends'] else 'No'}")
            print()
    
    # Recommendations
    print("\\n🎯 Recommendations:")
    print("   • Clean Mode: Best for piano performance, removes pitch bends")
    print("   • Harmony Mode: Best for complex vocal arrangements (Jacob Collier style)")
    print("   • Lead Mode: Best for simple melody extraction")
    print("   • Original Mode: Keep for comparison, shows raw vocal nuances")
    print()

def main():
    ap = argparse.ArgumentParser(description="Compare vocal processing modes")
    ap.add_argument("output_dir", help="Directory containing vocal MIDI files")
    args = ap.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Directory not found: {output_dir}")
        return
    
    compare_vocal_modes(output_dir)

if __name__ == "__main__":
    main()