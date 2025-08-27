#!/usr/bin/env python3
"""
Enhanced piano transcription using Spotify Basic Pitch with piano-friendly post-processing
Based on state-of-the-art recommendations for MP3 → piano-playable MIDI
"""

import os
import pathlib
import math
import numpy as np
import pretty_midi as pm
from basic_pitch.inference import predict_and_save, ICASSP_2022_MODEL_PATH
import librosa
import soundfile as sf

PIANO_MIN, PIANO_MAX = 21, 108        # A0..C8
DEFAULT_GRID = 16                      # 1/16-note grid
SPLIT_PITCH = 60                       # Middle C for L/R split

def _quantize_time(t, tempo_bpm, grid=DEFAULT_GRID):
    """Quantize time to musical grid"""
    spb = 60.0 / max(tempo_bpm, 1e-6)
    step = spb / grid
    return round(t / step) * step

def _remove_pitch_bends(instr: pm.Instrument):
    """Remove pitch bends for consistent piano playback"""
    instr.pitch_bends = []

def _clamp_range(instr: pm.Instrument, lo=PIANO_MIN, hi=PIANO_MAX):
    """Clamp notes to piano range A0-C8"""
    keep = []
    for n in instr.notes:
        if lo <= n.pitch <= hi:
            keep.append(n)
        elif n.pitch < lo:
            # Transpose up by octaves
            new_pitch = n.pitch
            while new_pitch < lo and new_pitch + 12 <= hi:
                new_pitch += 12
            if lo <= new_pitch <= hi:
                n.pitch = new_pitch
                keep.append(n)
        elif n.pitch > hi:
            # Transpose down by octaves
            new_pitch = n.pitch
            while new_pitch > hi and new_pitch - 12 >= lo:
                new_pitch -= 12
            if lo <= new_pitch <= hi:
                n.pitch = new_pitch
                keep.append(n)
    instr.notes = keep

def _normalize_velocities(instr: pm.Instrument, lo=35, hi=105):
    """Normalize velocities to playable range"""
    if not instr.notes:
        return
    
    v = np.array([n.velocity for n in instr.notes], dtype=float)
    vmin, vmax = v.min(), v.max()
    
    for n in instr.notes:
        if vmax > vmin:
            n.velocity = int(lo + (n.velocity - vmin) * (hi - lo) / (vmax - vmin))
        else:
            n.velocity = int((lo + hi) / 2)

def _quantize_notes(pmidi: pm.PrettyMIDI, grid=DEFAULT_GRID):
    """Quantize note timing to musical grid"""
    tempo = pmidi.estimate_tempo() or 120.0
    
    for inst in pmidi.instruments:
        for n in inst.notes:
            n.start = _quantize_time(n.start, tempo, grid)
            duration = n.end - n.start
            quantized_duration = _quantize_time(duration, tempo, grid)
            n.end = max(n.start + quantized_duration, n.start + 0.02)  # Minimum 20ms duration

def _to_piano_program(pmidi: pm.PrettyMIDI):
    """Set all instruments to Acoustic Grand Piano"""
    for inst in pmidi.instruments:
        inst.program = 0   # Acoustic Grand Piano
        inst.is_drum = False

def _split_hands(pmidi: pm.PrettyMIDI, split_pitch=SPLIT_PITCH):
    """Split into left and right hand tracks"""
    lh = pm.Instrument(program=0, name="Left Hand")
    rh = pm.Instrument(program=0, name="Right Hand")
    
    for inst in pmidi.instruments:
        for n in inst.notes:
            if n.pitch < split_pitch:
                lh.notes.append(n)
            else:
                rh.notes.append(n)
    
    pmidi.instruments = [lh, rh]

def _remove_short_notes(instr: pm.Instrument, min_duration=0.05):
    """Remove very short notes that are hard to play"""
    keep = []
    for n in instr.notes:
        if n.end - n.start >= min_duration:
            keep.append(n)
    instr.notes = keep

def _remove_overlapping_notes(instr: pm.Instrument):
    """Remove overlapping notes of the same pitch"""
    if not instr.notes:
        return
    
    # Sort by start time, then pitch
    notes = sorted(instr.notes, key=lambda n: (n.start, n.pitch))
    clean_notes = []
    
    for note in notes:
        # Check for overlap with existing notes of same pitch
        overlap_found = False
        for existing in clean_notes:
            if (existing.pitch == note.pitch and 
                existing.start < note.end and 
                existing.end > note.start):
                # Merge notes if they overlap
                existing.end = max(existing.end, note.end)
                overlap_found = True
                break
        
        if not overlap_found:
            clean_notes.append(note)
    
    instr.notes = clean_notes

def transcribe_with_basic_pitch(audio_path: str, out_dir: str, make_lr_split=True, quantize_grid=DEFAULT_GRID):
    """
    Enhanced transcription using Spotify Basic Pitch with piano-friendly post-processing
    
    Args:
        audio_path: Path to input audio file (MP3, WAV, etc.)
        out_dir: Output directory for MIDI files
        make_lr_split: Whether to create left/right hand split
        quantize_grid: Quantization grid (16 = 1/16 notes)
    
    Returns:
        tuple: (piano_midi_path, split_midi_path or None)
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Basic Pitch] Transcribing {audio_path}...")
    
    # 1) Basic Pitch inference (saves raw .mid)
    predict_and_save(
        audio_path_list=[audio_path],
        output_directory=out_dir,
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=False,
        model_or_model_path=ICASSP_2022_MODEL_PATH  # Use default TFLite model
    )
    
    # Find the created MIDI file (Basic Pitch adds _basic_pitch suffix)
    base = pathlib.Path(audio_path).stem
    # Simple approach: just look for any MIDI file that ends with _basic_pitch.mid
    midi_files = list(out_dir.glob("*_basic_pitch.mid"))
    if not midi_files:
        # Fallback: look for any MIDI file 
        midi_files = list(out_dir.glob("*.mid"))
    if not midi_files:
        raise Exception(f"No MIDI file created for {audio_path}")
    
    # Find the most recent one (in case multiple files)
    midi_path = max(midi_files, key=lambda p: p.stat().st_mtime)
    print(f"[Basic Pitch] Raw MIDI created: {midi_path}")

    # 2) Post-process to piano-friendly format
    print("[Post-process] Converting to piano-friendly MIDI...")
    pmidi = pm.PrettyMIDI(str(midi_path))
    
    # Process each instrument
    for inst in pmidi.instruments:
        _remove_pitch_bends(inst)
        _remove_short_notes(inst, min_duration=0.05)
        _clamp_range(inst)
        _remove_overlapping_notes(inst)
        _normalize_velocities(inst)

    # Apply global transformations
    _quantize_notes(pmidi, grid=quantize_grid)
    _to_piano_program(pmidi)

    # Save piano-optimized version
    piano_mid = out_dir / f"{base}.piano.mid"
    pmidi.write(str(piano_mid))
    print(f"[Post-process] Piano MIDI saved: {piano_mid}")

    # 3) Optional: Create left/right hand split
    split_mid = None
    if make_lr_split:
        print("[Post-process] Creating left/right hand split...")
        pmidi_split = pm.PrettyMIDI(initial_tempo=pmidi.estimate_tempo() or 120.0)
        
        # Load the piano MIDI and split hands
        merged = pm.PrettyMIDI(str(piano_mid))
        _split_hands(merged)
        pmidi_split.instruments = merged.instruments
        
        split_mid = out_dir / f"{base}.piano.LR.mid"
        pmidi_split.write(str(split_mid))
        print(f"[Post-process] L/R split MIDI saved: {split_mid}")

    return str(piano_mid), str(split_mid) if split_mid else None

def analyze_transcription_quality(midi_path: str):
    """Analyze the quality of transcribed MIDI"""
    try:
        pmidi = pm.PrettyMIDI(midi_path)
        
        total_notes = sum(len(inst.notes) for inst in pmidi.instruments)
        duration = pmidi.get_end_time()
        
        if total_notes == 0:
            return {
                'total_notes': 0,
                'duration': duration,
                'notes_per_second': 0,
                'pitch_range': (0, 0),
                'tempo': 0
            }
        
        all_pitches = []
        for inst in pmidi.instruments:
            all_pitches.extend([n.pitch for n in inst.notes])
        
        return {
            'total_notes': total_notes,
            'duration': duration,
            'notes_per_second': total_notes / max(duration, 1e-6),
            'pitch_range': (min(all_pitches), max(all_pitches)),
            'tempo': pmidi.estimate_tempo() or 0
        }
    except Exception as e:
        print(f"Error analyzing MIDI: {e}")
        return None

if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(description="Enhanced piano transcription using Basic Pitch")
    ap.add_argument("mp3_path", help="Path to input audio file")
    ap.add_argument("--out", default="transcriptions", help="Output directory")
    ap.add_argument("--nosplit", action="store_true", help="Do not split into L/R hands")
    ap.add_argument("--grid", type=int, default=16, help="Quantization grid (16=1/16 notes)")
    
    args = ap.parse_args()
    
    try:
        piano_mid, split_mid = transcribe_with_basic_pitch(
            args.mp3_path, 
            args.out, 
            make_lr_split=not args.nosplit,
            quantize_grid=args.grid
        )
        
        print("\n=== Transcription Results ===")
        print(f"Piano MIDI: {piano_mid}")
        if split_mid:
            print(f"L/R split MIDI: {split_mid}")
        
        # Analyze quality
        quality = analyze_transcription_quality(piano_mid)
        if quality:
            print(f"\n=== Quality Analysis ===")
            print(f"Total notes: {quality['total_notes']}")
            print(f"Duration: {quality['duration']:.1f}s")
            print(f"Notes per second: {quality['notes_per_second']:.1f}")
            print(f"Pitch range: {quality['pitch_range'][0]}-{quality['pitch_range'][1]} (MIDI)")
            print(f"Estimated tempo: {quality['tempo']:.1f} BPM")
            
    except Exception as e:
        print(f"Error: {e}")
        exit(1)