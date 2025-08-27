#!/usr/bin/env python3
"""
Enhanced Audio2NoteSheets Testing Script (2025)
Test and evaluate the improvements in BPM detection and musical intelligence.
"""

import sys
import os
import json
from pathlib import Path
import subprocess
import time
import mido
import librosa
import numpy as np

# Add current directory to path
sys.path.append('.')

try:
    from enhanced_bpm_detection import enhanced_bpm_detection, enhanced_midi_bpm_detection
    from enhanced_note_extraction import enhance_note_events_with_harmonic_analysis
    ENHANCED_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"[test] Enhanced modules not available: {e}")
    ENHANCED_MODULES_AVAILABLE = False

def run_bpm_comparison_test(audio_path: Path):
    """Compare BPM detection methods and show improvements"""
    print(f"\n{'='*60}")
    print(f"BPM DETECTION COMPARISON TEST")
    print(f"Testing audio: {audio_path.name}")
    print(f"{'='*60}")
    
    if not ENHANCED_MODULES_AVAILABLE:
        print("[test] Enhanced modules not available for testing")
        return
    
    # Test enhanced audio BPM detection
    print("\n[1] Testing Enhanced Audio BPM Detection...")
    try:
        result = enhanced_bpm_detection(audio_path)
        print(f"    Enhanced BPM: {result['bpm']:.1f}")
        print(f"    Confidence: {result['confidence']:.2f}")
        print(f"    Method: {result['method']}")
        if 'all_estimates' in result:
            print(f"    All estimates:")
            for bpm, conf, method in result['all_estimates']:
                print(f"      - {method}: {bpm:.1f} BPM (conf: {conf:.2f})")
    except Exception as e:
        print(f"    Error: {e}")
    
    # Test librosa fallback for comparison
    print("\n[2] Testing Librosa Fallback...")
    try:
        y, sr = librosa.load(str(audio_path), sr=22050)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        print(f"    Librosa BPM: {tempo:.1f}")
        print(f"    Beat count: {len(beats)}")
    except Exception as e:
        print(f"    Error: {e}")

def run_midi_analysis_test(midi_dir: Path):
    """Test MIDI file analysis and BPM detection"""
    print(f"\n{'='*60}")
    print(f"MIDI ANALYSIS TEST")
    print(f"Testing directory: {midi_dir}")
    print(f"{'='*60}")
    
    if not midi_dir.exists():
        print(f"[test] MIDI directory not found: {midi_dir}")
        return
    
    # Find MIDI files
    midi_files = list(midi_dir.glob("*_basic_pitch.mid"))
    if not midi_files:
        midi_files = list(midi_dir.glob("*.mid"))
    
    if not midi_files:
        print("[test] No MIDI files found")
        return
    
    print(f"\n[1] Found {len(midi_files)} MIDI files:")
    for midi_file in midi_files:
        print(f"    - {midi_file.name}")
    
    # Test enhanced MIDI BPM detection on each file
    if ENHANCED_MODULES_AVAILABLE:
        print(f"\n[2] Testing Enhanced MIDI BPM Detection...")
        for midi_file in midi_files[:3]:  # Test first 3 files
            try:
                result = enhanced_midi_bpm_detection(midi_file)
                print(f"    {midi_file.name}:")
                print(f"      BPM: {result['bpm']:.1f}")
                print(f"      Confidence: {result['confidence']:.2f}")
                print(f"      Method: {result['method']}")
            except Exception as e:
                print(f"    {midi_file.name}: Error - {e}")
    
    # Analyze note density and characteristics
    print(f"\n[3] Analyzing Note Characteristics...")
    for midi_file in midi_files[:2]:  # Analyze first 2 files
        try:
            analyze_midi_file_characteristics(midi_file)
        except Exception as e:
            print(f"    Error analyzing {midi_file.name}: {e}")

def analyze_midi_file_characteristics(midi_path: Path):
    """Analyze musical characteristics of a MIDI file"""
    print(f"    Analyzing: {midi_path.stem}")
    
    try:
        midi_file = mido.MidiFile(str(midi_path))
        
        # Extract basic statistics
        total_notes = 0
        note_pitches = []
        note_durations = []
        note_velocities = []
        
        for track in midi_file.tracks:
            current_time = 0
            active_notes = {}
            
            for msg in track:
                current_time += msg.time
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = (current_time, msg.velocity)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        start_time, velocity = active_notes[msg.note]
                        duration = current_time - start_time
                        
                        total_notes += 1
                        note_pitches.append(msg.note)
                        note_durations.append(duration)
                        note_velocities.append(velocity)
                        
                        del active_notes[msg.note]
        
        if total_notes > 0:
            avg_pitch = np.mean(note_pitches)
            avg_duration = np.mean(note_durations)
            avg_velocity = np.mean(note_velocities)
            pitch_range = max(note_pitches) - min(note_pitches)
            
            print(f"      Notes: {total_notes}")
            print(f"      Avg Pitch: {avg_pitch:.1f} (MIDI)")
            print(f"      Pitch Range: {pitch_range} semitones")
            print(f"      Avg Duration: {avg_duration:.0f} ticks")
            print(f"      Avg Velocity: {avg_velocity:.0f}")
        else:
            print(f"      No notes found")
            
    except Exception as e:
        print(f"      Error: {e}")

def run_arrangement_quality_test(song_name: str):
    """Test and compare arrangement quality"""
    print(f"\n{'='*60}")
    print(f"ARRANGEMENT QUALITY TEST")
    print(f"Testing song: {song_name}")
    print(f"{'='*60}")
    
    midi_dir = Path(f"outputs/midi/{song_name}")
    if not midi_dir.exists():
        print(f"[test] MIDI directory not found: {midi_dir}")
        return
    
    # Find different arrangement versions
    arrangements = {
        'basic': list(midi_dir.glob("*_basic_pitch.mid")),
        'intelligent': list(midi_dir.glob("*intelligent*.mid")),
        'clean': list(midi_dir.glob("*clean*.mid"))
    }
    
    print(f"\n[1] Found arrangements:")
    for arr_type, files in arrangements.items():
        print(f"    {arr_type}: {len(files)} files")
    
    # Compare arrangements
    print(f"\n[2] Arrangement Comparison:")
    for arr_type, files in arrangements.items():
        if files:
            file_to_analyze = files[0]  # Analyze first file of each type
            print(f"    {arr_type.upper()} - {file_to_analyze.name}")
            analyze_arrangement_complexity(file_to_analyze)

def analyze_arrangement_complexity(midi_path: Path):
    """Analyze the complexity and playability of an arrangement"""
    try:
        midi_file = mido.MidiFile(str(midi_path))
        
        # Extract notes with timing
        notes = []
        for track in midi_file.tracks:
            current_time = 0
            active_notes = {}
            
            for msg in track:
                current_time += msg.time
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = current_time
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        start_time = active_notes[msg.note]
                        duration = current_time - start_time
                        
                        notes.append({
                            'note': msg.note,
                            'start': start_time,
                            'duration': duration,
                            'velocity': msg.velocity if hasattr(msg, 'velocity') else 64
                        })
                        del active_notes[msg.note]
        
        if not notes:
            print(f"      No notes found")
            return
        
        # Calculate metrics
        total_duration = max(n['start'] + n['duration'] for n in notes)
        note_density = len(notes) / (total_duration / 1000)  # Notes per second
        
        # Polyphony analysis (max simultaneous notes)
        max_polyphony = 0
        time_points = set(n['start'] for n in notes)
        time_points.update(n['start'] + n['duration'] for n in notes)
        
        for time_point in sorted(time_points):
            active_count = sum(1 for n in notes 
                             if n['start'] <= time_point < n['start'] + n['duration'])
            max_polyphony = max(max_polyphony, active_count)
        
        # Piano playability score (rough estimate)
        playability_score = 10.0  # Start with perfect score
        if max_polyphony > 10:
            playability_score -= (max_polyphony - 10) * 0.5
        if note_density > 8:
            playability_score -= (note_density - 8) * 0.5
        playability_score = max(0, playability_score)
        
        print(f"      Total Notes: {len(notes)}")
        print(f"      Duration: {total_duration/1000:.1f}s")
        print(f"      Note Density: {note_density:.1f} notes/sec")
        print(f"      Max Polyphony: {max_polyphony} simultaneous notes")
        print(f"      Playability Score: {playability_score:.1f}/10")
        
    except Exception as e:
        print(f"      Analysis error: {e}")

def run_comprehensive_test():
    """Run comprehensive tests of the enhanced system"""
    print("="*80)
    print("ENHANCED AUDIO2NOTESHEETS COMPREHENSIVE TEST SUITE (2025)")
    print("="*80)
    
    # Check for test data
    test_audio_dir = Path("../../data")  # Adjust path as needed
    if not test_audio_dir.exists():
        test_audio_dir = Path("../data")
    if not test_audio_dir.exists():
        print("[test] No test audio directory found. Please place audio files in '../data' or '../../data'")
        return
    
    # Find audio files
    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.flac']:
        audio_files.extend(test_audio_dir.glob(ext))
    
    if not audio_files:
        print("[test] No audio files found for testing")
        return
    
    print(f"Found {len(audio_files)} audio files for testing:")
    for i, audio_file in enumerate(audio_files[:5], 1):  # Show first 5
        print(f"  {i}. {audio_file.name}")
    
    # Test first audio file
    if audio_files:
        test_audio = audio_files[0]
        song_name = test_audio.stem
        
        # Test BPM detection
        run_bpm_comparison_test(test_audio)
        
        # Test MIDI analysis (if MIDI files exist)
        midi_dir = Path(f"outputs/midi/{song_name}")
        if midi_dir.exists():
            run_midi_analysis_test(midi_dir)
            run_arrangement_quality_test(song_name)
        else:
            print(f"\n[test] No existing MIDI files found for {song_name}")
            print("       Run the main system first to generate MIDI files for testing")

def benchmark_bpm_detection(audio_files: list, num_files: int = 3):
    """Benchmark BPM detection performance"""
    print(f"\n{'='*60}")
    print(f"BPM DETECTION BENCHMARK")
    print(f"{'='*60}")
    
    if not ENHANCED_MODULES_AVAILABLE:
        print("[benchmark] Enhanced modules not available")
        return
    
    results = []
    
    for i, audio_file in enumerate(audio_files[:num_files]):
        print(f"\nProcessing {i+1}/{num_files}: {audio_file.name}")
        
        # Time the enhanced detection
        start_time = time.time()
        try:
            result = enhanced_bpm_detection(audio_file, confidence_threshold=0.5)
            detection_time = time.time() - start_time
            
            results.append({
                'file': audio_file.name,
                'bpm': result['bpm'],
                'confidence': result['confidence'],
                'method': result['method'],
                'time': detection_time,
                'success': True
            })
            
            print(f"  BPM: {result['bpm']:.1f}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Time: {detection_time:.2f}s")
            
        except Exception as e:
            results.append({
                'file': audio_file.name,
                'error': str(e),
                'success': False
            })
            print(f"  Error: {e}")
    
    # Summary
    successful = [r for r in results if r['success']]
    if successful:
        avg_time = np.mean([r['time'] for r in successful])
        avg_confidence = np.mean([r['confidence'] for r in successful])
        print(f"\nBenchmark Summary:")
        print(f"  Successful: {len(successful)}/{len(results)}")
        print(f"  Average Time: {avg_time:.2f}s")
        print(f"  Average Confidence: {avg_confidence:.2f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "bpm":
            # Test BPM detection on specific file
            if len(sys.argv) > 2:
                audio_path = Path(sys.argv[2])
                if audio_path.exists():
                    run_bpm_comparison_test(audio_path)
                else:
                    print(f"Audio file not found: {audio_path}")
            else:
                print("Usage: python test_enhancements.py bpm <audio_file>")
        
        elif command == "midi":
            # Test MIDI analysis on specific directory
            if len(sys.argv) > 2:
                midi_dir = Path(sys.argv[2])
                run_midi_analysis_test(midi_dir)
            else:
                print("Usage: python test_enhancements.py midi <midi_directory>")
        
        elif command == "arrangement":
            # Test arrangement quality
            if len(sys.argv) > 2:
                song_name = sys.argv[2]
                run_arrangement_quality_test(song_name)
            else:
                print("Usage: python test_enhancements.py arrangement <song_name>")
        
        elif command == "benchmark":
            # Run benchmark tests
            test_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("../data")
            audio_files = []
            for ext in ['*.mp3', '*.wav', '*.flac']:
                audio_files.extend(test_dir.glob(ext))
            
            if audio_files:
                benchmark_bpm_detection(audio_files)
            else:
                print(f"No audio files found in {test_dir}")
        
        else:
            print("Unknown command. Available commands: bmp, midi, arrangement, benchmark")
    
    else:
        # Run comprehensive test
        run_comprehensive_test()