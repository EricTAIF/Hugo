#!/usr/bin/env python3
"""
Compare transcription quality between different approaches:
1. Current system (stem separation + Basic Pitch per stem)
2. Enhanced Basic Pitch (direct full-song transcription with piano post-processing)
3. Piano Transcription Inference (when available)
"""

import requests
import json
import time
import pathlib
import pretty_midi as pm
import numpy as np

def analyze_midi_quality(midi_path):
    """Analyze MIDI transcription quality"""
    try:
        pmidi = pm.PrettyMIDI(midi_path)
        
        total_notes = sum(len(inst.notes) for inst in pmidi.instruments)
        duration = pmidi.get_end_time()
        
        if total_notes == 0:
            return None
        
        all_pitches = []
        note_durations = []
        velocities = []
        
        for inst in pmidi.instruments:
            for note in inst.notes:
                all_pitches.append(note.pitch)
                note_durations.append(note.end - note.start)
                velocities.append(note.velocity)
        
        # Analyze melody content (E4 notes for Everglow test)
        e4_notes = [p for p in all_pitches if p == 64]
        melody_range_notes = [p for p in all_pitches if 60 <= p <= 84]
        
        return {
            'total_notes': total_notes,
            'duration': duration,
            'notes_per_second': total_notes / max(duration, 1e-6),
            'pitch_range': (min(all_pitches), max(all_pitches)),
            'pitch_span': max(all_pitches) - min(all_pitches),
            'avg_duration': np.mean(note_durations),
            'avg_velocity': np.mean(velocities),
            'tempo': pmidi.estimate_tempo() or 0,
            'e4_count': len(e4_notes),
            'melody_notes': len(melody_range_notes),
            'melody_ratio': len(melody_range_notes) / total_notes
        }
    except Exception as e:
        print(f"Error analyzing {midi_path}: {e}")
        return None

def test_current_system():
    """Test current arrangement system"""
    print("🎵 Testing Current System (Stem + Arrangement)")
    
    base_url = "http://localhost:8000"
    test_song = "Coldplay - Everglow [Single Version] - (Official Video)"
    
    params = {
        "song": test_song,
        "arrangementMode": "combined",
        "aiEnabled": False,
        "legatoEnabled": True
    }
    
    start_time = time.time()
    resp = requests.post(f"{base_url}/api/generate", json=params)
    process_time = time.time() - start_time
    
    if not resp.json()['success']:
        print(f"❌ Current system failed: {resp.json().get('error', 'Unknown')}")
        return None
    
    data = resp.json()
    
    # Find the generated MIDI file 
    midi_files = list(pathlib.Path("outputs/midi").glob(f"*{test_song}*/intelligent_piano*.mid"))
    if not midi_files:
        print("❌ No MIDI file found for current system")
        return None
    
    midi_path = max(midi_files, key=lambda p: p.stat().st_mtime)
    quality = analyze_midi_quality(str(midi_path))
    
    return {
        'name': 'Current System',
        'process_time': process_time,
        'midi_path': str(midi_path),
        'api_note_count': data['noteCount'],
        'quality': quality
    }

def test_basic_pitch_enhanced():
    """Test enhanced Basic Pitch approach"""
    print("🎵 Testing Enhanced Basic Pitch")
    
    # Use the already created piano-processed version
    midi_path = "enhanced_transcriptions/everglow.basic_pitch.piano.mid"
    
    if not pathlib.Path(midi_path).exists():
        print(f"❌ Enhanced Basic Pitch MIDI not found: {midi_path}")
        return None
    
    quality = analyze_midi_quality(midi_path)
    
    return {
        'name': 'Enhanced Basic Pitch',
        'process_time': None,  # Already processed
        'midi_path': midi_path,
        'quality': quality
    }

def compare_transcriptions():
    """Compare all transcription approaches"""
    print("🔍 Transcription Quality Comparison")
    print("=" * 50)
    
    results = []
    
    # Test current system
    current_result = test_current_system()
    if current_result:
        results.append(current_result)
    
    # Test enhanced Basic Pitch
    basic_pitch_result = test_basic_pitch_enhanced()
    if basic_pitch_result:
        results.append(basic_pitch_result)
    
    # Print comparison
    print("\n📊 COMPARISON RESULTS")
    print("=" * 60)
    
    for result in results:
        print(f"\n🎹 {result['name']}")
        print("-" * 30)
        
        if result['process_time']:
            print(f"Processing time: {result['process_time']:.1f}s")
        
        if result['quality']:
            q = result['quality']
            print(f"Total notes: {q['total_notes']}")
            print(f"Duration: {q['duration']:.1f}s")
            print(f"Notes/second: {q['notes_per_second']:.1f}")
            print(f"Pitch range: {q['pitch_range'][0]}-{q['pitch_range'][1]} (span: {q['pitch_span']})")
            print(f"Avg note duration: {q['avg_duration']:.3f}s")
            print(f"Avg velocity: {q['avg_velocity']:.1f}")
            print(f"Estimated tempo: {q['tempo']:.1f} BPM")
            print(f"E4 notes (melody): {q['e4_count']}")
            print(f"Melody range notes: {q['melody_notes']} ({q['melody_ratio']:.1%})")
        
        print(f"MIDI file: {result['midi_path']}")
    
    # Analysis
    print("\n🎯 ANALYSIS")
    print("=" * 30)
    
    if len(results) >= 2:
        current = results[0]['quality']
        basic_pitch = results[1]['quality']
        
        if current and basic_pitch:
            print(f"Note count: Current={current['total_notes']}, Basic Pitch={basic_pitch['total_notes']}")
            print(f"E4 melody notes: Current={current['e4_count']}, Basic Pitch={basic_pitch['e4_count']}")
            print(f"Melody ratio: Current={current['melody_ratio']:.1%}, Basic Pitch={basic_pitch['melody_ratio']:.1%}")
            print(f"Pitch span: Current={current['pitch_span']}, Basic Pitch={basic_pitch['pitch_span']}")
            
            # Which is better for piano tutorials?
            print("\n💡 RECOMMENDATIONS:")
            
            if basic_pitch['melody_notes'] > current['melody_notes']:
                print("✅ Enhanced Basic Pitch captures more melody content")
            else:
                print("✅ Current system has good melody capture")
            
            if basic_pitch['total_notes'] > current['total_notes'] * 1.2:
                print("⚠️ Enhanced Basic Pitch may be too dense for tutorials")
            elif basic_pitch['total_notes'] < current['total_notes'] * 0.8:
                print("⚠️ Enhanced Basic Pitch may be too sparse")
            else:
                print("✅ Enhanced Basic Pitch has good note density")
            
            if abs(basic_pitch['tempo'] - 144) < abs(current['tempo'] - 144):  # Everglow is ~144 BPM
                print("✅ Enhanced Basic Pitch has better tempo detection")
            else:
                print("✅ Current system has better tempo detection")

if __name__ == "__main__":
    try:
        compare_transcriptions()
        print("\n🎊 Comparison completed successfully!")
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()