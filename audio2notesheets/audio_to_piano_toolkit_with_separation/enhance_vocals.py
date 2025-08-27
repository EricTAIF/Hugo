#!/usr/bin/env python3
"""
Enhanced vocal transcription tool specifically designed for better vocal MIDI processing.

This script provides multiple vocal processing modes:
- clean: Removes pitch bends, focuses on main melody, piano-friendly
- harmony: Preserves multiple voices for complex arrangements (Jacob Collier style)
- lead: Extracts only the main melody line
- original: Keeps pitch bends for comparison

Examples:
  # Process existing vocal stems with all modes
  python enhance_vocals.py "outputs/htdemucs_6s/Snow White - Laufey/vocals.mp3"
  
  # Process with specific mode only
  python enhance_vocals.py "vocals.mp3" --mode clean --out vocal_outputs
  
  # Process for complex harmonies (Jacob Collier style)
  python enhance_vocals.py "complex_harmonies.mp3" --mode harmony --max-voices 6
"""

import argparse
from pathlib import Path
from main import transcribe_vocals_enhanced, load_audio_for_beats, make_grid
import pretty_midi as pm

def process_existing_vocal_midi(midi_path: Path, out_dir: Path, mode: str = "all", max_voices: int = 4):
    """Process an existing vocal MIDI file with enhanced vocal algorithms."""
    from main import process_vocals_advanced, quantize_pm, to_piano_program, remove_pitch_bends
    
    midi_path = Path(midi_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[enhance] Processing existing MIDI: {midi_path}")
    
    # Load the MIDI
    try:
        m = pm.PrettyMIDI(str(midi_path))
    except Exception as e:
        print(f"[enhance] Failed to load MIDI: {e}")
        return
    
    stem_name = midi_path.stem
    
    if mode == "all":
        modes = ["clean", "harmony", "lead", "original"]
    else:
        modes = [mode]
    
    for proc_mode in modes:
        try:
            # Create a copy for processing
            m_copy = pm.PrettyMIDI(str(midi_path))
            
            if proc_mode != "original":
                # Apply vocal processing
                m_processed = process_vocals_advanced(m_copy, mode=proc_mode)
                if proc_mode == "harmony":
                    # Allow more voices for harmony mode
                    for inst in m_processed.instruments:
                        inst.notes = _preserve_vocal_harmonies_enhanced(inst.notes, max_voices=max_voices)
            else:
                # Keep original but ensure it's piano-ready
                m_processed = m_copy
                to_piano_program(m_processed)
            
            # Save processed version
            output_path = out_dir / f"{stem_name}.vocals.{proc_mode}.mid"
            m_processed.write(str(output_path))
            print(f"[enhance] {proc_mode.capitalize()} mode: {output_path}")
            
        except Exception as e:
            print(f"[enhance] Failed to process {proc_mode} mode: {e}")

def _preserve_vocal_harmonies_enhanced(notes, max_voices=6):
    """Enhanced harmony preservation for complex vocal arrangements like Jacob Collier."""
    if not notes:
        return notes
    
    # Use smaller time windows for complex harmonies (25ms)
    time_groups = {}
    for note in notes:
        time_key = round(note.start * 40)  # 25ms buckets
        if time_key not in time_groups:
            time_groups[time_key] = []
        time_groups[time_key].append(note)
    
    result = []
    for group in time_groups.values():
        if len(group) <= max_voices:
            result.extend(group)
        else:
            # For complex harmonies, prefer notes that form good intervals
            group_sorted = _sort_by_harmonic_importance(group)
            result.extend(group_sorted[:max_voices])
    
    return result

def _sort_by_harmonic_importance(notes):
    """Sort notes by harmonic importance for complex vocal arrangements."""
    # Basic harmonic importance: bass notes and octaves/fifths get priority
    def harmonic_score(note):
        score = note.velocity  # Start with velocity
        
        # Boost bass notes (important for harmony foundation)
        if note.pitch < 60:  # Below middle C
            score += 20
        
        # Boost notes that might be chord tones (rough estimation)
        pitch_class = note.pitch % 12
        if pitch_class in [0, 4, 7]:  # C, E, G (major triad)
            score += 10
        elif pitch_class in [2, 5, 9]:  # D, F, A (common extensions)
            score += 5
            
        return score
    
    return sorted(notes, key=harmonic_score, reverse=True)

def main():
    ap = argparse.ArgumentParser(description="Enhanced vocal MIDI processing")
    ap.add_argument("input", help="Audio file (.mp3/.wav) or existing MIDI file (.mid)")
    ap.add_argument("--out", default="vocal_enhanced", help="Output directory")
    ap.add_argument("--mode", choices=["all", "clean", "harmony", "lead", "original"], 
                    default="all", help="Processing mode")
    ap.add_argument("--max-voices", type=int, default=4, 
                    help="Maximum voices in harmony mode (use 6+ for Jacob Collier style)")
    ap.add_argument("--midi-only", action="store_true", 
                    help="Input is already a MIDI file to reprocess")
    
    args = ap.parse_args()
    
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    
    if not input_path.exists():
        print(f"[enhance] Input file not found: {input_path}")
        return
    
    if args.midi_only or input_path.suffix.lower() == ".mid":
        # Process existing MIDI
        process_existing_vocal_midi(input_path, out_dir, args.mode, args.max_voices)
    else:
        # Transcribe audio then process
        try:
            print(f"[enhance] Transcribing audio: {input_path}")
            raw_mid, clean_mid, harmony_mid, lead_mid, original_mid = transcribe_vocals_enhanced(
                input_path, out_dir, "vocals"
            )
            print(f"[enhance] Complete! Check {out_dir} for all vocal modes.")
            
            # If harmony mode requested with custom max voices, reprocess harmony version
            if args.max_voices != 4:
                print(f"[enhance] Reprocessing harmony mode with {args.max_voices} max voices...")
                process_existing_vocal_midi(harmony_mid, out_dir, "harmony", args.max_voices)
                
        except Exception as e:
            print(f"[enhance] Failed to process audio: {e}")

if __name__ == "__main__":
    main()