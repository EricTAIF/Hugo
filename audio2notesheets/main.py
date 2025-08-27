#!/usr/bin/env python3
import argparse
import os
import sys
import shutil
import subprocess
from pathlib import Path

# Transcription
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH

# Score assembly
from music21 import converter, instrument, stream, clef, tempo, meter
import math

# Enhanced modules for 2025 improvements
try:
    from enhanced_bpm_detection import enhanced_bpm_detection, enhanced_midi_bpm_detection
    from enhanced_note_extraction import (
        enhance_basic_pitch_parameters, enhance_note_events_with_harmonic_analysis,
        create_intelligent_phrase_detection, create_adaptive_arrangement
    )
    ENHANCED_MODULES_AVAILABLE = True
    print("[system] Enhanced 2025 modules loaded successfully")
except ImportError as e:
    print(f"[system] Enhanced modules not available: {e}")
    ENHANCED_MODULES_AVAILABLE = False

# Video generation removed per user request

SUPPORTED_STEMS = ["vocals", "bass", "other", "drums", "guitar", "piano"]  # Enhanced instrument separation

# Video generation functionality removed per user request

def run_spleeter(input_audio: Path, out_root: Path, stems: int = 5) -> Path:
    """
    Run Spleeter separation for 5-stem (vocals/drums/bass/piano/other) separation.
    Returns the directory containing the separated WAVs.
    """
    try:
        import spleeter.separator
        from spleeter.audio.adapter import AudioAdapter
    except ImportError:
        raise ImportError("Spleeter not installed. Run: pip install spleeter")
    
    print(f"[spleeter] Separating with {stems}-stem model...")
    
    # Create output directory
    base = input_audio.stem
    sep_dir = out_root / "spleeter" / base
    sep_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize separator
    separator = spleeter.separator.Separator(f'spleeter:{stems}stems')
    audio_adapter = AudioAdapter.default()
    
    # Load and separate audio
    sample_rate = 44100
    waveform, _ = audio_adapter.load(str(input_audio), sample_rate=sample_rate)
    prediction = separator.separate(waveform)
    
    # Save separated stems
    for instrument, audio_data in prediction.items():
        output_path = sep_dir / f"{instrument}.wav"
        audio_adapter.save(str(output_path), audio_data, sample_rate)
        print(f"[spleeter] Saved: {output_path}")
    
    print(f"[spleeter] Stems at: {sep_dir}")
    return sep_dir

def run_demucs(input_audio: Path, out_root: Path, model: str, device: str) -> Path:
    """
    Run Demucs separation via its Python entrypoint.
    Enhanced with support for 6-stem models (guitar/piano).
    Returns the directory containing the separated WAVs for this track.
    """
    demucs_args = ["--mp3", "-d", device, "-n", model, "-o", str(out_root), str(input_audio)]
    print(f"[demucs] Separating with model={model} device={device} ...")
    
    # Enhanced models support
    if model == "htdemucs_6s":
        print("[demucs] Using 6-stem model for enhanced guitar/piano separation")
    elif model == "htdemucs_ft":
        print("[demucs] Using fine-tuned hybrid transformer model (slower but better quality)")
    elif model == "mdx_extra":
        print("[demucs] Using MDX challenge winner model with extra training data")
    
    import demucs.separate as demucs_separate
    demucs_separate.main(demucs_args)

    base = input_audio.stem
    # Demucs creates outputs directly in model name directory, not under "separated"
    sep_dir = out_root / model / base
    if not sep_dir.exists():
        # Fallback search
        candidates = list(out_root.rglob(base))
        if candidates:
            sep_dir = candidates[0]
        else:
            raise FileNotFoundError(f"Could not find Demucs output directory for {base}")
    print(f"[demucs] Stems at: {sep_dir}")
    return sep_dir

def create_intelligent_piano_arrangement(midi_files: dict, output_midi: Path):
    """
    Create an intelligent piano arrangement using modern AI-inspired approaches:
    1. Harmonic analysis and chord detection
    2. Voice leading optimization
    3. Dynamic density based on musical structure
    4. Phrase-aware note selection
    5. Proper piano hand distribution
    """
    import mido
    
    # Detect BPM from available MIDI files (fallback to 120)
    bpm = detect_bpm_from_midi_files(midi_files)
    print(f"[intelligent-piano] Detected BPM: {bpm}")

    # Create new MIDI file
    new_midi = mido.MidiFile(ticks_per_beat=480)
    new_track = mido.MidiTrack()
    new_midi.tracks.append(new_track)
    
    # Add tempo and time signature
    tempo_us = int(round(60000000 / max(1, bpm)))
    new_track.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
    new_track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    
    print(f"[intelligent-piano] Creating intelligent piano arrangement...")
    
    # Step 1: Extract and analyze all musical content
    all_events = extract_all_musical_events(midi_files)
    if not all_events:
        print("[intelligent-piano] No musical events found")
        return
    
    # Step 2: Detect phrases and musical structure
    phrases = detect_musical_phrases(all_events)
    print(f"[intelligent-piano] Detected {len(phrases)} musical phrases")
    
    # Step 3: Analyze harmony and chords
    harmonic_analysis = analyze_harmony(all_events)
    print(f"[intelligent-piano] Analyzed {len(harmonic_analysis)} harmonic regions")
    
    # Step 4: Create intelligent arrangement
    arrangement = create_layered_arrangement(phrases, harmonic_analysis)
    
    # Step 5: Apply voice leading optimization
    optimized_arrangement = optimize_voice_leading(arrangement)
    
    # Step 6: Ensure piano playability
    playable = ensure_piano_playability(optimized_arrangement)
    
    # Step 7: Apply additional smoothing: quantize/merge and sustain
    final_arrangement = quantize_and_merge(playable, bpm=bpm, grid='1/16', merge_gap_ms=80, min_duration_ms=200)
    pedal_events = generate_sustain_pedal_events(final_arrangement, bpm=bpm, enabled=True, window_ms=None)
    
    print(f"[intelligent-piano] Final arrangement: {len(final_arrangement)} notes")
    
    # Convert to MIDI with proper tempo
    convert_to_midi(final_arrangement, new_track, bpm=bpm, ticks_per_beat=new_midi.ticks_per_beat, control_events=pedal_events)
    new_track.append(mido.MetaMessage('end_of_track', time=0))
    new_midi.save(str(output_midi))
    print(f"[intelligent-piano] Saved intelligent arrangement: {output_midi}")

def extract_all_musical_events(midi_files):
    """Extract and categorize all musical events from all parts"""
    all_events = []
    
    for part_name, midi_path in midi_files.items():
        if not midi_path.exists():
            continue
            
        try:
            import mido
            events = extract_note_events(mido.MidiFile(str(midi_path)))
            for event in events:
                event['part'] = part_name
                event['importance'] = get_part_importance(part_name)
                all_events.append(event)
        except Exception as e:
            print(f"[intelligent-piano] Error processing {part_name}: {e}")
    
    return sorted(all_events, key=lambda x: x['start'])

def get_part_importance(part_name):
    """Assign importance weights to different parts"""
    weights = {
        'vocals': 1.0,    # Highest priority - melody
        'piano': 0.8,     # High priority - harmonic content  
        'guitar': 0.6,    # Medium priority - harmony/rhythm
        'bass': 0.7,      # Medium-high - structural foundation
        'drums': 0.3,     # Low priority for melodic content
        'other': 0.4      # Low priority
    }
    return weights.get(part_name, 0.5)

def detect_musical_phrases(events):
    """Detect musical phrases - create more phrases for better coverage"""
    phrases = []
    if not events:
        return phrases
    
    # Much more aggressive phrase detection to create more manageable chunks
    events_sorted = sorted(events, key=lambda x: x['start'])
    total_duration = max(e['start'] + e['duration'] for e in events_sorted)
    
    # Create phrases every 10-15 seconds instead of waiting for large gaps
    phrase_duration = 12000  # 12 seconds per phrase
    overlap = 2000           # 2 second overlap between phrases
    
    phrase_start = 0
    phrase_number = 1
    
    while phrase_start < total_duration:
        phrase_end = phrase_start + phrase_duration
        
        # Get all events in this time window
        phrase_events = [e for e in events_sorted 
                        if e['start'] >= phrase_start and e['start'] < phrase_end]
        
        if phrase_events:  # Any events found
            actual_start = min(e['start'] for e in phrase_events)
            actual_end = max(e['start'] + e['duration'] for e in phrase_events)
            density = len(phrase_events) / ((actual_end - actual_start) / 1000 + 1)
            
            phrases.append({
                'events': phrase_events,
                'start': actual_start,
                'end': actual_end,
                'density': density,
                'number': phrase_number
            })
            phrase_number += 1
        
        # Move to next phrase with overlap
        phrase_start += phrase_duration - overlap
    
    print(f"[intelligent-piano] Created {len(phrases)} phrases from {total_duration/1000:.1f}s of music")
    return phrases

def analyze_harmony(events):
    """Analyze harmonic content to identify chord progressions"""
    harmonic_regions = []
    window_size = 2000  # 2-second windows for harmonic analysis
    
    if not events:
        return harmonic_regions
    
    total_duration = max(e['start'] + e['duration'] for e in events)
    
    for start_time in range(0, int(total_duration), window_size):
        end_time = start_time + window_size
        
        # Get all notes in this window
        window_events = [e for e in events if 
                        e['start'] < end_time and e['start'] + e['duration'] > start_time]
        
        if window_events:
            # Analyze pitch content
            pitches = [e['note'] % 12 for e in window_events]
            pitch_weights = {}
            
            for event in window_events:
                pitch = event['note'] % 12
                weight = event['importance'] * (event['duration'] / 1000)
                pitch_weights[pitch] = pitch_weights.get(pitch, 0) + weight
            
            # Find most prominent pitches (chord tones)
            sorted_pitches = sorted(pitch_weights.items(), key=lambda x: x[1], reverse=True)
            chord_tones = [p[0] for p in sorted_pitches[:4]]  # Top 4 pitches
            
            harmonic_regions.append({
                'start': start_time,
                'end': end_time,
                'chord_tones': chord_tones,
                'events': window_events
            })
    
    return harmonic_regions

def create_layered_arrangement(phrases, harmonic_analysis):
    """Create arrangement with intelligent layering based on phrase structure"""
    arrangement = []
    
    for phrase in phrases:
        phrase_start = phrase['start']
        phrase_end = phrase['end']
        phrase_density = phrase['density']
        
        # Determine arrangement style - aim for realistic piano arrangement density
        phrase_length_seconds = (phrase_end - phrase_start) / 1000
        
        # Target 4-8 notes per second for a playable arrangement (much more reasonable)
        min_notes_per_phrase = max(15, int(phrase_length_seconds * 4))
        max_notes_per_phrase = int(phrase_length_seconds * 8)
        
        # Use much higher percentages to keep more notes
        available_notes = len(phrase['events'])
        
        if phrase_density > 20:  # Very high density phrase - be more selective
            target_notes = min(int(available_notes * 0.25), max_notes_per_phrase)
            target_notes = max(target_notes, min_notes_per_phrase)
            style = 'very_dense'
        elif phrase_density > 15:  # High density phrase - moderately selective
            target_notes = min(int(available_notes * 0.35), max_notes_per_phrase)
            target_notes = max(target_notes, min_notes_per_phrase)
            style = 'dense'
        elif phrase_density > 8:  # Medium density - keep reasonable amount
            target_notes = min(int(available_notes * 0.5), max_notes_per_phrase) 
            target_notes = max(target_notes, min_notes_per_phrase)
            style = 'medium'
        else:  # Low density - keep most notes but respect limits
            target_notes = min(int(available_notes * 0.7), max_notes_per_phrase)
            target_notes = max(target_notes, min_notes_per_phrase // 2)
            style = 'sparse'
        
        print(f"[intelligent-piano] Phrase {len(arrangement)+1}: {style} style, target {target_notes} notes")
        
        # Select best notes for this phrase
        phrase_arrangement = select_phrase_notes(phrase, target_notes, harmonic_analysis)
        arrangement.extend(phrase_arrangement)
    
    return arrangement

def select_phrase_notes(phrase, target_count, harmonic_analysis):
    """Select the most important notes for a phrase using multiple criteria"""
    events = phrase['events']
    
    # Score each note based on multiple factors
    scored_events = []
    for event in events:
        score = calculate_note_importance(event, phrase, harmonic_analysis)
        scored_events.append((score, event))
    
    # Sort by score and take top notes
    scored_events.sort(key=lambda x: x[0], reverse=True)
    selected_events = [event for score, event in scored_events[:target_count]]
    
    return selected_events

def calculate_note_importance(event, phrase, harmonic_analysis):
    """Calculate importance score for a note based on multiple musical factors"""
    score = 0.0
    duration = event['duration']
    
    # Factor 1: Duration filtering - heavily penalize very short notes
    if duration < 150:  # Less than 150ms - too short for piano
        score -= 20  # Heavy penalty
    elif duration < 300:  # Less than 300ms - quite short
        score -= 10  # Moderate penalty
    elif duration < 500:  # Less than 500ms - acceptable but short
        score += duration / 200  # Small bonus
    else:  # 500ms+ - good duration for piano
        score += min(duration / 200, 10)  # Good bonus, capped at 10
    
    # Factor 2: Part importance
    score += event['importance'] * 8
    
    # Factor 3: Pitch range suitability
    note = event['note']
    if event['part'] == 'vocals' and 60 <= note <= 84:  # Melody range
        score += 8
    elif event['part'] == 'bass' and 24 <= note <= 55:  # Bass range
        score += 6
    elif 48 <= note <= 72:  # Harmony range
        score += 4
    
    # Factor 4: Harmonic relevance
    event_time = event['start']
    relevant_harmony = next((h for h in harmonic_analysis 
                           if h['start'] <= event_time < h['end']), None)
    if relevant_harmony:
        pitch_class = note % 12
        if pitch_class in relevant_harmony['chord_tones'][:3]:  # Top 3 chord tones
            score += 6
        elif pitch_class in relevant_harmony['chord_tones']:
            score += 3
    
    # Factor 5: Avoid overly high/low notes
    if note < 24 or note > 96:
        score -= 5
    
    # Factor 6: Prefer notes that are on strong beats (every 500ms)
    beat_position = event['start'] % 500
    if beat_position < 50:  # Close to beat
        score += 3
    
    return score

def optimize_voice_leading(arrangement):
    """Apply voice leading principles to smooth transitions between chords"""
    if len(arrangement) < 2:
        return arrangement
    
    optimized = [arrangement[0]]  # Keep first note
    
    for i in range(1, len(arrangement)):
        current_note = arrangement[i]
        prev_notes = [n for n in optimized if 
                     abs(n['start'] - current_note['start']) < 500]  # Notes playing around same time
        
        if prev_notes:
            # Check for voice leading issues
            best_octave = find_best_octave_for_voice_leading(current_note, prev_notes)
            if best_octave != current_note['note'] // 12:
                # Transpose to better octave
                pitch_class = current_note['note'] % 12
                new_note = best_octave * 12 + pitch_class
                if 24 <= new_note <= 96:  # Keep in piano range
                    current_note = current_note.copy()
                    current_note['note'] = new_note
        
        optimized.append(current_note)
    
    return optimized

def find_best_octave_for_voice_leading(note, context_notes):
    """Find the best octave for a note to minimize voice leading jumps"""
    if not context_notes:
        return note['note'] // 12
    
    pitch_class = note['note'] % 12
    min_movement = float('inf')
    best_octave = note['note'] // 12
    
    # Try different octaves
    for octave in range(1, 8):  # C1 to C8
        candidate_note = octave * 12 + pitch_class
        if not (24 <= candidate_note <= 96):  # Piano range check
            continue
            
        # Calculate total movement from context notes
        total_movement = sum(abs(candidate_note - ctx['note']) for ctx in context_notes)
        
        if total_movement < min_movement:
            min_movement = total_movement
            best_octave = octave
    
    return best_octave

def ensure_piano_playability(arrangement):
    """Ensure the arrangement is playable on piano (max 10 fingers)"""
    # Group notes by time windows
    time_groups = {}
    for note in arrangement:
        time_slot = note['start'] // 200  # 200ms windows
        if time_slot not in time_groups:
            time_groups[time_slot] = []
        time_groups[time_slot].append(note)
    
    final_arrangement = []
    
    for time_slot, notes in time_groups.items():
        if len(notes) <= 8:  # Playable with 8 fingers (reserve 2 for transitions)
            final_arrangement.extend(notes)
        else:
            # Keep only the most important notes
            scored_notes = [(calculate_note_importance(note, None, []), note) for note in notes]
            scored_notes.sort(key=lambda x: x[0], reverse=True)
            final_arrangement.extend([note for score, note in scored_notes[:8]])
    
    return final_arrangement

def convert_to_midi(arrangement, track, bpm=120, ticks_per_beat=None, control_events=None):
    """Convert arrangement (ms timeline) to MIDI events with proper ticks using bpm.

    - arrangement: list of dicts with 'start' and 'duration' in milliseconds.
    - track: mido.MidiTrack to append events to.
    - bpm: tempo to use for time conversion and optional tempo meta should be inserted by caller.
    - ticks_per_beat: ticks resolution; defaults to parent MidiFile.ticks_per_beat or 480.
    - control_events: optional list of CC/meta events, each as dict:
        {'type': 'cc', 'control': 64, 'value': 64, 'time': <ms>}
    """
    import mido
    if ticks_per_beat is None:
        try:
            ticks_per_beat = track._parent.ticks_per_beat  # type: ignore[attr-defined]
        except Exception:
            ticks_per_beat = 480

    # Convert ms to ticks: ticks = ms * ticks_per_beat * bpm / 60000
    def ms_to_ticks(ms: int) -> int:
        return int(round((ms * ticks_per_beat * (bpm / 60000.0))))

    # Build on/off events in ticks
    midi_events = []
    for note in arrangement:
        start_ms = max(0, int(note['start']))
        dur_ms = max(1, int(note['duration']))
        start_ticks = ms_to_ticks(start_ms)
        end_ticks = start_ticks + ms_to_ticks(dur_ms)
        midi_events.append((start_ticks, 0, mido.Message('note_on', note=note['note'], velocity=note.get('velocity', 70), time=0)))
        midi_events.append((end_ticks, 1, mido.Message('note_off', note=note['note'], velocity=0, time=0)))

    # Add control events (e.g., sustain pedal)
    if control_events:
        for ev in control_events:
            if ev.get('type') == 'cc':
                t = ms_to_ticks(int(ev.get('time', 0)))
                midi_events.append((t, 0, mido.Message('control_change', control=int(ev['control']), value=int(ev['value']), time=0)))

    # Sort by absolute tick time; ensure note_off (1) precedes note_on (0) when equal
    midi_events.sort(key=lambda x: (x[0], x[1]))

    last_ticks = 0
    for abs_ticks, _prio, msg in midi_events:
        delta = max(0, abs_ticks - last_ticks)
        msg.time = delta
        track.append(msg)
        last_ticks = abs_ticks

def quantize_and_merge(arrangement, bpm: int, grid: str | None = '1/16', merge_gap_ms: int = 80, min_duration_ms: int = 200):
    """Quantize note starts/ends to a grid and merge near-adjacent/repeated notes.
    - grid: '1/8', '1/16', '1/12' (triplet), or None to skip quantization.
    """
    if not arrangement:
        return arrangement
    beat_ms = 60000.0 / max(1, bpm)
    if grid is None or grid.lower() == 'none':
        step_ms = None
    else:
        denom = grid.split('/')[-1]
        try:
            denom = int(denom)
        except Exception:
            denom = 16
        step_ms = beat_ms * (4.0/denom)

    notes = sorted([n.copy() for n in arrangement], key=lambda x: (x['start'], x['note']))

    # Quantize starts/ends
    if step_ms:
        for n in notes:
            q_start = int(round(n['start'] / step_ms) * step_ms)
            q_end = int(round((n['start'] + n['duration']) / step_ms) * step_ms)
            if q_end <= q_start:
                q_end = q_start + max(min_duration_ms, int(step_ms/2))
            n['start'] = max(0, q_start)
            n['duration'] = max(min_duration_ms, q_end - q_start)

    # Merge repeated notes with tiny gaps
    merged = []
    for n in notes:
        if merged and merged[-1]['note'] == n['note']:
            prev = merged[-1]
            prev_end = prev['start'] + prev['duration']
            gap = n['start'] - prev_end
            if gap <= merge_gap_ms and gap >= -50:  # allow slight overlap
                # Extend previous note to cover
                new_end = max(prev_end, n['start'] + n['duration'])
                prev['duration'] = new_end - prev['start']
                continue
        # Ensure minimum duration
        if n['duration'] < min_duration_ms:
            n['duration'] = min_duration_ms
        merged.append(n)

    return merged

def generate_sustain_pedal_events(arrangement, bpm: int, enabled: bool = True, window_ms: int | None = None, release_ms: int = 60):
    """Generate simple sustain pedal CC events around chord groups to smooth playback.
    Groups notes starting within window_ms and adds CC64 on at group start and off after last note of group + release.
    """
    if not enabled or not arrangement:
        return []
    beat_ms = int(60000.0 / max(1, bpm))
    if window_ms is None:
        window_ms = int(beat_ms * 0.75)

    notes = sorted(arrangement, key=lambda n: n['start'])
    events = []
    i = 0
    nlen = len(notes)
    while i < nlen:
        group_start = notes[i]['start']
        group_end = notes[i]['start'] + notes[i]['duration']
        j = i + 1
        while j < nlen and notes[j]['start'] - group_start <= window_ms:
            group_end = max(group_end, notes[j]['start'] + notes[j]['duration'])
            j += 1
        # Add sustain on/off
        events.append({'type': 'cc', 'control': 64, 'value': 64, 'time': group_start})
        events.append({'type': 'cc', 'control': 64, 'value': 0, 'time': group_end + release_ms})
        i = j
    return events

def create_clean_piano_arrangement(midi_files: dict, output_midi: Path):
    """
    Create a clean, playable piano arrangement by:
    1. Using vocals as primary melody (right hand)
    2. Using bass as foundation (left hand bass notes)
    3. Adding select chord tones from guitar/piano (sparse accompaniment)
    4. Removing note clusters and ensuring playability
    """
    import mido
    
    # Create new MIDI file
    new_midi = mido.MidiFile(ticks_per_beat=480)
    new_track = mido.MidiTrack()
    new_midi.tracks.append(new_track)
    
    # Add basic tempo and time signature
    new_track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))  # 120 BPM
    new_track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    
    # Process each part separately with strict filtering
    melody_notes = []  # Right hand melody (vocals)
    bass_notes = []    # Left hand bass line
    harmony_notes = [] # Sparse chord accompaniment
    
    print(f"[clean-piano] Creating clean piano arrangement...")
    
    # 1. Extract VOCAL MELODY (right hand, sparse)
    if 'vocals' in midi_files and midi_files['vocals'].exists():
        print(f"[clean-piano] Processing vocals for melody...")
        try:
            midi_file = mido.MidiFile(str(midi_files['vocals']))
            vocal_events = extract_note_events(midi_file)
            
            # Filter vocals: keep only sustained notes in melody range
            for event in vocal_events:
                note, start, duration = event['note'], event['start'], event['duration']
                if (60 <= note <= 84 and  # C4-C6 melody range
                    duration > 200):      # Only sustained notes
                    melody_notes.append({
                        'note': note,
                        'start': start,
                        'duration': min(duration, 2000),  # Cap duration
                        'velocity': 85,
                        'type': 'melody'
                    })
        except Exception as e:
            print(f"[clean-piano] Error processing vocals: {e}")
    
    # 2. Extract BASS LINE (left hand, very sparse)
    if 'bass' in midi_files and midi_files['bass'].exists():
        print(f"[clean-piano] Processing bass for foundation...")
        try:
            midi_file = mido.MidiFile(str(midi_files['bass']))
            bass_events = extract_note_events(midi_file)
            
            # Filter bass: only strong bass notes, widely spaced
            last_bass_time = -1000
            for event in bass_events:
                note, start, duration = event['note'], event['start'], event['duration']
                if (24 <= note <= 55 and              # Bass range C1-G3
                    duration > 300 and                # Sustained notes only
                    start - last_bass_time > 500):    # Space out bass notes
                    
                    bass_notes.append({
                        'note': note,
                        'start': start,
                        'duration': min(duration, 1500),
                        'velocity': 75,
                        'type': 'bass'
                    })
                    last_bass_time = start
        except Exception as e:
            print(f"[clean-piano] Error processing bass: {e}")
    
    # 3. Extract SPARSE HARMONY (middle range, very selective)
    harmony_sources = ['piano', 'guitar']
    for source in harmony_sources:
        if source in midi_files and midi_files[source].exists():
            print(f"[clean-piano] Processing {source} for harmony...")
            try:
                midi_file = mido.MidiFile(str(midi_files[source]))
                harmony_events = extract_note_events(midi_file)
                
                # Very selective harmony: only chord tones, well-spaced
                last_harmony_time = -800
                for event in harmony_events:
                    note, start, duration = event['note'], event['start'], event['duration']
                    if (48 <= note <= 72 and              # Middle range C3-C5
                        duration > 400 and                # Longer notes only
                        start - last_harmony_time > 400 and # Space out chords
                        len([h for h in harmony_notes if abs(h['start'] - start) < 100]) < 2):  # Max 2 simultaneous notes
                        
                        harmony_notes.append({
                            'note': note,
                            'start': start,
                            'duration': min(duration, 1200),
                            'velocity': 65,
                            'type': 'harmony'
                        })
                        last_harmony_time = start
                        
                        # Limit harmony density
                        if len(harmony_notes) > 150:  # Cap total harmony notes
                            break
            except Exception as e:
                print(f"[clean-piano] Error processing {source}: {e}")
    
    # Combine all notes
    all_notes = melody_notes + bass_notes + harmony_notes
    
    # Remove conflicts (same note at same time)
    final_notes = []
    all_notes.sort(key=lambda x: (x['start'], x['note']))
    
    for note in all_notes:
        conflicts = [n for n in final_notes 
                    if n['note'] == note['note'] and 
                    abs(n['start'] - note['start']) < 50]  # Very close timing
        
        if not conflicts:
            final_notes.append(note)
        elif note['type'] == 'melody':  # Melody has priority
            final_notes = [n for n in final_notes if n not in conflicts]
            final_notes.append(note)
    
    print(f"[clean-piano] Selected {len(melody_notes)} melody, {len(bass_notes)} bass, {len(harmony_notes)} harmony notes")
    print(f"[clean-piano] Final arrangement: {len(final_notes)} total notes")
    
    # Convert to MIDI events
    midi_events = []
    for note in final_notes:
        midi_events.append((note['start'], mido.Message('note_on', 
                                                       note=note['note'], 
                                                       velocity=note['velocity'], 
                                                       time=0)))
        midi_events.append((note['start'] + note['duration'], mido.Message('note_off', 
                                                                          note=note['note'], 
                                                                          velocity=0, 
                                                                          time=0)))
    
    # Sort and add to track
    midi_events.sort(key=lambda x: x[0])
    last_time = 0
    for abs_time, msg in midi_events:
        msg.time = abs_time - last_time
        new_track.append(msg)
        last_time = abs_time
    
    new_track.append(mido.MetaMessage('end_of_track', time=0))
    new_midi.save(str(output_midi))
    print(f"[clean-piano] Saved clean arrangement: {output_midi}")

def extract_note_events(midi_file):
    """Extract note events with absolute timing in milliseconds from MIDI file.
    Handles tempo changes by converting delta ticks to seconds with the current tempo.
    """
    import mido
    events = []
    ticks_per_beat = midi_file.ticks_per_beat or 480
    default_tempo = 500000  # 120 BPM if none specified

    for track in midi_file.tracks:
        current_time_sec = 0.0
        current_tempo = default_tempo
        active_notes = {}

        for msg in track:
            # Convert this delta to seconds using the tempo in effect before this message
            if hasattr(msg, 'time'):
                current_time_sec += mido.tick2second(msg.time, ticks_per_beat, current_tempo)

            if msg.type == 'set_tempo':
                # Tempo change takes effect after this message's delta time has been applied
                current_tempo = msg.tempo
                continue

            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = (current_time_sec, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    start_time_sec, velocity = active_notes[msg.note]
                    duration_ms = max(0, int(round((current_time_sec - start_time_sec) * 1000)))
                    events.append({
                        'note': msg.note,
                        'start': int(round(start_time_sec * 1000)),
                        'duration': duration_ms,
                        'velocity': velocity
                    })
                    del active_notes[msg.note]

    # Sort by start time in ms
    events.sort(key=lambda e: e['start'])
    return events

def create_piano_arrangement(midi_files: dict, output_midi: Path):
    """
    Create a comprehensive piano arrangement by intelligently combining
    vocals, guitar, piano, and bass parts into a single playable piano piece.
    """
    import mido
    
    # Create new MIDI file
    new_midi = mido.MidiFile(ticks_per_beat=480)
    new_track = mido.MidiTrack()
    new_midi.tracks.append(new_track)
    
    # Add basic tempo and time signature
    new_track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))  # 120 BPM
    new_track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    
    # Collect all notes from all parts with their timing and source
    all_notes = []
    
    for part_name, midi_path in midi_files.items():
        if not midi_path.exists():
            continue
            
        print(f"[piano-arrange] Processing {part_name} from {midi_path.name}")
        
        try:
            midi_file = mido.MidiFile(str(midi_path))
            
            # Extract notes with absolute timing
            for track in midi_file.tracks:
                current_time = 0
                active_notes = {}  # note -> start_time
                
                for msg in track:
                    current_time += msg.time
                    
                    if msg.type == 'note_on' and msg.velocity > 0:
                        active_notes[msg.note] = current_time
                        
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        if msg.note in active_notes:
                            start_time = active_notes[msg.note]
                            duration = current_time - start_time
                            
                            # Categorize notes by part and apply piano-friendly processing
                            note_info = {
                                'note': msg.note,
                                'start': start_time,
                                'duration': duration,
                                'velocity': 64,  # Default velocity
                                'part': part_name,
                                'priority': 0
                            }
                            
                            # Assign priorities and adjust for piano range
                            if part_name == 'vocals':
                                # Vocals become melody line (right hand, upper range)
                                if 60 <= msg.note <= 84:  # C4-C6
                                    note_info['priority'] = 4
                                    note_info['velocity'] = 80
                                elif msg.note > 84:
                                    note_info['note'] = max(60, msg.note - 12)  # Transpose down
                                    note_info['priority'] = 4
                                    note_info['velocity'] = 80
                                elif msg.note < 60:
                                    note_info['note'] = min(84, msg.note + 12)  # Transpose up
                                    note_info['priority'] = 4
                                    note_info['velocity'] = 80
                                    
                            elif part_name == 'guitar':
                                # Guitar becomes harmony/accompaniment (middle range)
                                if 48 <= msg.note <= 72:  # C3-C5
                                    note_info['priority'] = 3
                                    note_info['velocity'] = 65
                                elif msg.note > 72:
                                    note_info['note'] = max(48, msg.note - 12)
                                    note_info['priority'] = 3
                                    note_info['velocity'] = 65
                                    
                            elif part_name == 'piano':
                                # Original piano part (full range)
                                if 36 <= msg.note <= 84:
                                    note_info['priority'] = 2
                                    note_info['velocity'] = 70
                                    
                            elif part_name == 'bass':
                                # Bass becomes left hand (lower range)
                                if 24 <= msg.note <= 60:  # C1-C4
                                    note_info['priority'] = 1
                                    note_info['velocity'] = 75
                                elif msg.note > 60:
                                    note_info['note'] = max(24, msg.note - 12)  # Transpose down
                                    note_info['priority'] = 1
                                    note_info['velocity'] = 75
                            
                            # Only add notes with reasonable duration and within piano range
                            if (note_info['priority'] > 0 and 
                                duration > 50 and  # Minimum duration
                                24 <= note_info['note'] <= 96):  # Piano range
                                all_notes.append(note_info)
                                
                            del active_notes[msg.note]
                            
        except Exception as e:
            print(f"[piano-arrange] Error processing {part_name}: {e}")
            continue
    
    print(f"[piano-arrange] Collected {len(all_notes)} notes from all parts")
    
    # Sort notes by start time
    all_notes.sort(key=lambda x: x['start'])
    
    # Remove overlapping notes of same pitch (keep highest priority)
    final_notes = []
    for note in all_notes:
        # Check for conflicts with existing notes
        conflicts = [n for n in final_notes 
                    if n['note'] == note['note'] and 
                    not (n['start'] + n['duration'] <= note['start'] or 
                         note['start'] + note['duration'] <= n['start'])]
        
        # If there are conflicts, keep the highest priority note
        if conflicts:
            highest_priority = max(conflicts + [note], key=lambda x: x['priority'])
            if highest_priority == note:
                # Remove conflicting notes and add this one
                final_notes = [n for n in final_notes if n not in conflicts]
                final_notes.append(note)
        else:
            final_notes.append(note)
    
    # Sort final notes by start time
    final_notes.sort(key=lambda x: x['start'])
    
    # Convert to MIDI events
    midi_events = []
    for note in final_notes:
        # Add note on
        midi_events.append((note['start'], mido.Message('note_on', 
                                                       note=note['note'], 
                                                       velocity=note['velocity'], 
                                                       time=0)))
        # Add note off
        midi_events.append((note['start'] + note['duration'], mido.Message('note_off', 
                                                                          note=note['note'], 
                                                                          velocity=0, 
                                                                          time=0)))
    
    # Sort all events by time
    midi_events.sort(key=lambda x: x[0])
    
    # Convert to relative timing and add to track
    last_time = 0
    for abs_time, msg in midi_events:
        msg.time = abs_time - last_time
        new_track.append(msg)
        last_time = abs_time
    
    # Add end of track
    new_track.append(mido.MetaMessage('end_of_track', time=0))
    
    # Save the arrangement
    new_midi.save(str(output_midi))
    print(f"[piano-arrange] Created piano arrangement with {len(final_notes)} notes")
    print(f"[piano-arrange] Saved to: {output_midi}")

def optimize_piano_midi(input_midi: Path, output_midi: Path):
    """
    Optimize MIDI specifically for piano performance by:
    1. Consolidating rapid repeated notes into sustained notes
    2. Adjusting note durations to be more realistic for piano
    3. Removing excessive note density
    4. Quantizing to reasonable rhythmic values
    """
    import mido
    
    midi_file = mido.MidiFile(str(input_midi))
    new_midi = mido.MidiFile(ticks_per_beat=midi_file.ticks_per_beat)
    new_track = mido.MidiTrack()
    new_midi.tracks.append(new_track)
    
    # Copy meta messages
    if midi_file.tracks:
        for msg in midi_file.tracks[0]:
            if msg.type in ['set_tempo', 'time_signature', 'key_signature']:
                new_track.append(msg.copy())
    
    # Collect all note events with absolute timing
    all_events = []
    for track in midi_file.tracks:
        current_time = 0
        for msg in track:
            current_time += msg.time
            if msg.type in ['note_on', 'note_off']:
                all_events.append((current_time, msg.copy()))
    
    all_events.sort(key=lambda x: x[0])
    
    # Process notes to create piano-friendly patterns
    processed_notes = {}  # note -> (start_time, velocity)
    final_events = []
    
    # Parameters for piano optimization
    min_note_duration = 480  # Minimum duration (quarter note at 480 ticks)
    max_note_duration = 1920  # Maximum duration (whole note)
    sustain_gap_threshold = 120  # Gap below which notes are considered sustained
    
    for abs_time, msg in all_events:
        if msg.type == 'note_on' and msg.velocity > 0:
            note = msg.note
            
            # If note is already playing, extend it instead of retriggering
            if note in processed_notes:
                start_time, velocity = processed_notes[note]
                gap = abs_time - start_time
                
                if gap < sustain_gap_threshold:
                    # Extend the existing note
                    continue
                else:
                    # End previous note and start new one
                    duration = min(max_note_duration, max(min_note_duration, gap))
                    final_events.append((start_time, mido.Message('note_on', note=note, velocity=velocity, time=0)))
                    final_events.append((start_time + duration, mido.Message('note_off', note=note, velocity=0, time=0)))
            
            processed_notes[note] = (abs_time, msg.velocity)
            
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            note = msg.note
            if note in processed_notes:
                start_time, velocity = processed_notes[note]
                duration = abs_time - start_time
                
                # Ensure reasonable duration
                duration = min(max_note_duration, max(min_note_duration, duration))
                
                final_events.append((start_time, mido.Message('note_on', note=note, velocity=velocity, time=0)))
                final_events.append((start_time + duration, mido.Message('note_off', note=note, velocity=0, time=0)))
                
                del processed_notes[note]
    
    # Handle any remaining notes
    for note, (start_time, velocity) in processed_notes.items():
        final_events.append((start_time, mido.Message('note_on', note=note, velocity=velocity, time=0)))
        final_events.append((start_time + min_note_duration, mido.Message('note_off', note=note, velocity=0, time=0)))
    
    # Sort and convert to relative timing
    final_events.sort(key=lambda x: x[0])
    
    last_time = 0
    for abs_time, msg in final_events:
        msg.time = abs_time - last_time
        new_track.append(msg)
        last_time = abs_time
    
    new_track.append(mido.MetaMessage('end_of_track', time=0))
    new_midi.save(str(output_midi))
    print(f"[piano-optimize] Created piano-optimized MIDI with {len(final_events)//2} notes")

def clean_midi_file(input_midi: Path, output_midi: Path):
    """
    Clean up Basic Pitch MIDI output by consolidating multiple single-note tracks 
    into a single track with all notes.
    """
    import mido
    
    # Load the original MIDI file
    midi_file = mido.MidiFile(str(input_midi))
    
    # Create a new MIDI file with a single track
    new_midi = mido.MidiFile()
    new_track = mido.MidiTrack()
    new_midi.tracks.append(new_track)
    
    # Copy tempo and other meta messages from first track
    if midi_file.tracks:
        for msg in midi_file.tracks[0]:
            if msg.type in ['set_tempo', 'time_signature', 'key_signature']:
                new_track.append(msg.copy())
    
    # Collect all note events with their absolute timing
    all_events = []
    
    for track in midi_file.tracks:
        current_time = 0
        for msg in track:
            current_time += msg.time
            if msg.type in ['note_on', 'note_off']:
                all_events.append((current_time, msg.copy()))
    
    # Sort events by time
    all_events.sort(key=lambda x: x[0])
    
    # Convert back to relative timing and add to new track
    last_time = 0
    for abs_time, msg in all_events:
        msg.time = abs_time - last_time
        new_track.append(msg)
        last_time = abs_time
    
    # Add end of track message
    new_track.append(mido.MetaMessage('end_of_track', time=0))
    
    # Save the cleaned MIDI file
    new_midi.save(str(output_midi))
    print(f"[midi-clean] Consolidated {len(midi_file.tracks)} tracks into 1 track")

def transcribe_with_basic_pitch(stem_wav: Path, out_dir: Path, stem_name: str = ""):
    """Enhanced transcription with 2025 optimized parameters for smoother MIDI output"""
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[basic-pitch] Transcribing {stem_wav.name} ...")
    
    # Use enhanced parameters if available
    if ENHANCED_MODULES_AVAILABLE:
        params = enhance_basic_pitch_parameters(stem_name)
        onset_thresh = params['onset_threshold']
        frame_thresh = params['frame_threshold'] 
        min_note_len = params['minimum_note_length']
        multiple_pitch_bends = params['multiple_pitch_bends']
        melodia_trick = params['melodia_trick']
        print(f"[enhanced-pitch] Using 2025 parameters for {stem_name}: onset={onset_thresh}, frame={frame_thresh}, min_len={min_note_len}")
    else:
        # Fallback to original parameters
        if "vocal" in stem_name.lower():
            # Vocals: optimized for melodic lines and sustained notes
            onset_thresh, frame_thresh, min_note_len = 0.2, 0.15, 400.0
        elif "bass" in stem_name.lower():
            # Bass: sustained low notes, higher minimum length
            onset_thresh, frame_thresh, min_note_len = 0.4, 0.25, 250.0
        elif "drum" in stem_name.lower():
            # Drums: sharp attacks, shorter notes
            onset_thresh, frame_thresh, min_note_len = 0.6, 0.4, 50.0
        elif "guitar" in stem_name.lower():
            # Guitar: medium sustain, moderate attack sensitivity
            onset_thresh, frame_thresh, min_note_len = 0.35, 0.3, 120.0
        elif "piano" in stem_name.lower():
            # Piano: optimized for sustained notes and chord detection
            onset_thresh, frame_thresh, min_note_len = 0.3, 0.2, 300.0
        else:
            # Other instruments: balanced settings
            onset_thresh, frame_thresh, min_note_len = 0.4, 0.3, 150.0
        
        multiple_pitch_bends = True
        melodia_trick = True
    
    predict_and_save(
        audio_path_list=[str(stem_wav)],
        output_directory=str(out_dir),
        save_midi=True,
        sonify_midi=False,
        save_model_outputs=False,
        save_notes=True,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=onset_thresh,
        frame_threshold=frame_thresh,
        minimum_note_length=min_note_len,
        multiple_pitch_bends=multiple_pitch_bends,
        melodia_trick=melodia_trick
    )
    
    # Clean up the generated MIDI file to consolidate tracks
    original_midi = latest_child_with_suffix(out_dir, stem_wav.stem, ".mid")
    cleaned_midi = original_midi.with_name(original_midi.stem + "_cleaned.mid")
    clean_midi_file(original_midi, cleaned_midi)
    
    # Apply enhanced note processing if available
    if ENHANCED_MODULES_AVAILABLE:
        try:
            # Extract note events and enhance them with harmonic analysis
            import mido
            midi_file = mido.MidiFile(str(cleaned_midi))
            note_events = extract_note_events(midi_file)
            
            if note_events:
                enhanced_events = enhance_note_events_with_harmonic_analysis(note_events, stem_wav)
                print(f"[enhanced-notes] Enhanced {len(note_events)} -> {len(enhanced_events)} notes for {stem_name}")
                
                # Create new MIDI file with enhanced events
                enhanced_midi = cleaned_midi.with_name(cleaned_midi.stem + "_enhanced.mid")
                create_enhanced_midi_file(enhanced_events, enhanced_midi, midi_file.ticks_per_beat)
                
                # Replace cleaned with enhanced
                cleaned_midi.unlink()
                cleaned_midi = enhanced_midi
        except Exception as e:
            print(f"[enhanced-notes] Error enhancing notes for {stem_name}: {e}")
    
    # Apply piano-specific optimization if this is a piano track
    if "piano" in stem_name.lower():
        piano_optimized = original_midi.with_name(original_midi.stem + "_piano_optimized.mid")
        optimize_piano_midi(cleaned_midi, piano_optimized)
        cleaned_midi.unlink()  # Delete intermediate file
        cleaned_midi = piano_optimized
    
    # Replace original with processed version
    original_midi.unlink()  # Delete original
    cleaned_midi.rename(original_midi)  # Rename processed to original

def create_enhanced_midi_file(events: list, output_path: Path, ticks_per_beat: int = 480):
    """Create MIDI file from enhanced note events"""
    import mido
    
    new_midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    new_track = mido.MidiTrack()
    new_midi.tracks.append(new_track)
    
    # Add tempo
    new_track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))  # 120 BPM
    
    # Convert events to MIDI messages
    midi_events = []
    for event in events:
        start_ticks = int(event['start'] * ticks_per_beat / 1000)
        duration_ticks = int(event['duration'] * ticks_per_beat / 1000)
        velocity = event.get('velocity', 64)
        
        midi_events.append((start_ticks, mido.Message('note_on', note=event['note'], velocity=velocity, time=0)))
        midi_events.append((start_ticks + duration_ticks, mido.Message('note_off', note=event['note'], velocity=0, time=0)))
    
    # Sort events by time
    midi_events.sort(key=lambda x: x[0])
    
    # Add events with proper timing
    last_time = 0
    for abs_time, msg in midi_events:
        msg.time = abs_time - last_time
        new_track.append(msg)
        last_time = abs_time
    
    new_track.append(mido.MetaMessage('end_of_track', time=0))
    new_midi.save(str(output_path))

def latest_child_with_suffix(folder: Path, stem_name: str, suffix: str = ".mid") -> Path:
    candidates = sorted(folder.glob(f"*{stem_name}*{suffix}"))
    if not candidates:
        candidates = sorted(folder.rglob(f"*{suffix}"))
    if not candidates:
        raise FileNotFoundError(f"No {suffix} produced in {folder}")
    return candidates[-1]

def _detect_bpm_from_midi_file(midi_path: Path) -> int | None:
    """Try to detect BPM from MIDI meta or inter-onset intervals."""
    try:
        import mido
        mf = mido.MidiFile(str(midi_path))
        # 1) Look for set_tempo meta messages
        tempos = []
        for tr in mf.tracks:
            for msg in tr:
                if msg.type == 'set_tempo':
                    tempos.append(msg.tempo)
        if tempos:
            # Use first tempo or median if multiple
            tempo_us = sorted(tempos)[len(tempos)//2]
            bpm = int(round(60000000.0 / max(1, tempo_us)))
            if 30 <= bpm <= 240:
                return bpm

        # 2) Fallback: estimate from inter-onset intervals
        events = extract_note_events(mf)
        onsets = sorted({e['start'] for e in events})
        if len(onsets) >= 4:
            diffs = [ (onsets[i+1] - onsets[i]) for i in range(len(onsets)-1) ]  # in ms
            # Filter reasonable IOIs (0.25s..2s)
            diffs = [d for d in diffs if 250 <= d <= 2000]
            if diffs:
                # Use median IOI
                diffs.sort()
                median_ioi = diffs[len(diffs)//2]
                bpm_est = int(round(60000.0 / median_ioi))
                # Clamp to common range
                bpm_est = min(240, max(30, bpm_est))
                return bpm_est
    except Exception:
        pass
    return None

def detect_bpm_from_midi_files(midi_files: dict) -> int:
    """Enhanced BPM detection from MIDI files with 2025 improvements."""
    if ENHANCED_MODULES_AVAILABLE:
        # Use enhanced MIDI BPM detection
        best_result = None
        best_confidence = 0.0
        
        # Try enhanced detection on each MIDI file
        preferred_order = ['drums', 'piano', 'vocals', 'guitar', 'bass', 'other']
        for key in preferred_order:
            p = midi_files.get(key)
            if p and Path(p).exists():
                result = enhanced_midi_bpm_detection(Path(p))
                if result['confidence'] > best_confidence:
                    best_result = result
                    best_confidence = result['confidence']
        
        if best_result and best_confidence > 0.3:
            print(f"[enhanced-bpm] Detected BPM: {best_result['bpm']:.1f} (confidence: {best_confidence:.2f}, method: {best_result['method']})")
            return int(best_result['bpm'])
    
    # Fallback to original method
    bpm_candidates = []
    # Prefer drums/piano/vocals as sources
    preferred_order = ['drums', 'piano', 'vocals', 'guitar', 'bass', 'other']
    for key in preferred_order:
        p = midi_files.get(key)
        if p and Path(p).exists():
            bpm = _detect_bpm_from_midi_file(Path(p))
            if bpm:
                bpm_candidates.append(bpm)
    if not bpm_candidates:
        # Try any remaining
        for _, p in midi_files.items():
            bpm = _detect_bpm_from_midi_file(Path(p))
            if bpm:
                bpm_candidates.append(bpm)
    if bpm_candidates:
        # Return the median candidate to reduce outliers
        bpm_candidates.sort()
        return bpm_candidates[len(bpm_candidates)//2]
    return 120

def estimate_bpm_from_audio(audio_path: Path) -> int | None:
    """Estimate BPM from an audio file using librosa."""
    try:
        import librosa
        y, sr = librosa.load(str(audio_path), mono=True)
        # Use beat tracker
        tempo_est, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = int(round(float(tempo_est)))
        if 30 <= bpm <= 240:
            return bpm
    except Exception:
        return None
    return None

def parse_context_file(context_path: Path):
    """Parse a context MIDI/MusicXML file into BPM, key, chord timeline (in beats), and intro notes (in beats).
    Returns dict with keys: bpm (int|None), key (dict|None), chords (list), intro (list), duration_beats (float)
    """
    try:
        from music21 import converter, tempo as m21tempo, key as m21key, chord as m21chord, note as m21note
        s = converter.parse(str(context_path))
        # Tempo
        tempos = s.recurse().getElementsByClass(m21tempo.MetronomeMark)
        bpm = None
        for mm in tempos:
            if mm.number:
                bpm = int(round(float(mm.number)))
                break
        # Key
        try:
            ky = s.analyze('key')
            key_dict = {'tonic': ky.tonic.name, 'mode': ky.mode}
        except Exception:
            key_dict = None
        # Chords via chordify
        cs = s.chordify()
        chords = []
        for ch in cs.recurse().getElementsByClass(m21chord.Chord):
            pcs = sorted({p.pitchClass for p in ch.pitches})
            start_beats = float(ch.offset)
            dur_beats = float(ch.quarterLength)
            if pcs and dur_beats > 0:
                chords.append({'start_beats': start_beats, 'end_beats': start_beats + dur_beats, 'chord_tones': pcs})
        # Intro notes: take first measures up to some beats; leave selection to caller
        intro_notes = []
        for n in s.recurse().notes:
            if isinstance(n, m21note.Note):
                intro_notes.append({'note': int(n.pitch.midi), 'start_beats': float(n.offset), 'dur_beats': float(n.quarterLength), 'velocity': 80})
        total_beats = float(s.highestTime)
        return {'bpm': bpm, 'key': key_dict, 'chords': chords, 'intro': intro_notes, 'duration_beats': total_beats}
    except Exception as e:
        print(f"[context] Failed to parse context {context_path}: {e}")
        return {'bpm': None, 'key': None, 'chords': [], 'intro': [], 'duration_beats': 0.0}

def assemble_score(midi_map: dict, out_musicxml: Path, bpm: int, timesig: str, include_drums: bool = False):
    score = stream.Score()
    if bpm:
        score.insert(0, tempo.MetronomeMark(number=bpm))
    if timesig:
        try:
            score.insert(0, meter.TimeSignature(timesig))
        except Exception:
            print(f"[warn] invalid time signature '{timesig}', ignoring.", file=sys.stderr)

    for name, midi_path in midi_map.items():
        # Skip drums in sheet music unless explicitly requested (too complex for standard notation)
        if name.lower() == "drums" and not include_drums:
            print(f"[music21] Skipping {name} from sheet music (MIDI saved separately)")
            continue
            
        print(f"[music21] Adding part {name} from {midi_path.name}")
        s = converter.parse(str(midi_path))
        part = s.parts[0] if hasattr(s, "parts") and len(s.parts) > 0 else s
        part.id = name
        part.partName = name.capitalize()
        if name.lower() == "bass":
            part.insert(0, instrument.ElectricBass()); part.insert(0, clef.BassClef())
        elif name.lower() == "vocals":
            part.insert(0, instrument.Vocalist()); part.insert(0, clef.TrebleClef())
        elif name.lower() == "drums":
            part.insert(0, instrument.Percussion()); part.insert(0, clef.PercussionClef())
        elif name.lower() == "guitar":
            part.insert(0, instrument.ElectricGuitar()); part.insert(0, clef.TrebleClef())
        elif name.lower() == "piano":
            part.insert(0, instrument.Piano()); part.insert(0, clef.TrebleClef())
        else:
            part.insert(0, instrument.Piano()); part.insert(0, clef.TrebleClef())
        score.insert(len(score.parts), part)

    out_musicxml.parent.mkdir(parents=True, exist_ok=True)
    out_midi = out_musicxml.with_suffix(".mid")
    score.write("musicxml", fp=str(out_musicxml))
    score.write("midi", fp=str(out_midi))
    print(f"[music21] Wrote {out_musicxml} and {out_midi}")

def main():
    ap = argparse.ArgumentParser(description="Separate an MP3 and generate printable sheet music (MusicXML).")
    ap.add_argument("input", type=str, help="Path to input audio (mp3/wav etc.)")
    ap.add_argument("-o", "--out", type=str, default="outputs", help="Output root directory")
    ap.add_argument("--separator", choices=["demucs", "spleeter"], default="demucs", help="Separation method")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device for Demucs")
    ap.add_argument("--model", default="htdemucs", help="Model name (demucs: htdemucs/htdemucs_6s/htdemucs_ft/mdx_extra, spleeter: 5stems)")
    ap.add_argument("--spleeter-stems", type=int, choices=[2, 4, 5], default=5, help="Number of stems for Spleeter (2, 4, or 5)")
    ap.add_argument("--include-other", choices=["yes", "no"], default="yes", help="Include 'other' stem in the notation")
    ap.add_argument("--include-drums", choices=["yes", "no"], default="no", help="Include drums in sheet music (generates MIDI regardless)")
# Video generation argument removed
    ap.add_argument("--tempo", type=int, default=0, help="Optional BPM for the score")
    ap.add_argument("--time-signature", type=str, default="", help="Optional time signature like 4/4, 3/4, 6/8")
    ap.add_argument("--skip-separation", action="store_true", help="Skip Demucs (reuse existing stems)")
    ap.add_argument("--skip-transcription", action="store_true", help="Skip Basic Pitch (reuse existing midis)")
    
    # Piano arrangement parameters
    ap.add_argument("--intelligent-piano", action="store_true", help="Generate intelligent piano arrangement")
    ap.add_argument("--piano-params", type=str, help="JSON file with piano arrangement parameters")
    ap.add_argument("--phrase-duration", type=int, default=12, help="Phrase duration in seconds (default: 12)")
    ap.add_argument("--phrase-overlap", type=int, default=2, help="Phrase overlap in seconds (default: 2)")
    ap.add_argument("--min-notes-per-sec", type=float, default=4.0, help="Min notes per second (default: 4.0)")
    ap.add_argument("--max-notes-per-sec", type=float, default=8.0, help="Max notes per second (default: 8.0)")
    ap.add_argument("--dense-ratio", type=float, default=0.35, help="Note ratio for dense phrases (default: 0.35)")
    ap.add_argument("--medium-ratio", type=float, default=0.50, help="Note ratio for medium phrases (default: 0.50)")
    
    # Smoothness and legato parameters
    ap.add_argument("--legato", action="store_true", default=True, help="Enable legato connections (default: True)")
    ap.add_argument("--no-legato", action="store_false", dest="legato", help="Disable legato connections")
    ap.add_argument("--legato-threshold", type=int, default=300, help="Legato gap threshold in ms (default: 300)")
    ap.add_argument("--min-note-duration", type=int, default=200, help="Minimum note duration in ms (default: 200)")
# Video skip argument removed
    args = ap.parse_args()

    input_audio = Path(args.input).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    if not input_audio.exists():
        print(f"Input does not exist: {input_audio}", file=sys.stderr); sys.exit(2)

    if args.skip_separation:
        # Look for existing separation directory  
        candidates = list(out_root.rglob(input_audio.stem))
        sep_dir = next((c for c in candidates if c.is_dir()), None)
        if not sep_dir:
            print("[error] --skip-separation used but no previous stems found.", file=sys.stderr); sys.exit(3)
    else:
        # Run separation based on chosen method
        if args.separator == "spleeter":
            sep_dir = run_spleeter(input_audio, out_root, args.spleeter_stems)
        else:  # demucs
            sep_dir = run_demucs(input_audio, out_root, args.model, args.device)

    expected = {
        "vocals": ["vocals.mp3", "vocals.wav"], 
        "bass": ["bass.mp3", "bass.wav"], 
        "other": ["other.mp3", "other.wav"], 
        "drums": ["drums.mp3", "drums.wav"],
        "guitar": ["guitar.mp3", "guitar.wav"],
        "piano": ["piano.mp3", "piano.wav"]
    }
    midi_out_dir = out_root / "midi" / input_audio.stem
    midi_out_dir.mkdir(parents=True, exist_ok=True)

    # Generate MIDI for ALL stems
    all_midi_paths = {}
    for stem_name, candidates in expected.items():
        wav = None
        
        # Only check for audio files if we need to do transcription
        if not args.skip_transcription:
            for c in candidates:
                p = sep_dir / c
                if p.exists(): wav = p; break
            if wav is None:
                print(f"[warn] No file found for stem {stem_name} in {sep_dir}, skipping.", file=sys.stderr); continue
            
            transcribe_with_basic_pitch(wav, midi_out_dir, stem_name)

        # Look for existing MIDI files
        midi_file = None
        try:
            midi_file = latest_child_with_suffix(midi_out_dir, stem_name, ".mid")
        except FileNotFoundError:
            if wav:
                try:
                    midi_file = latest_child_with_suffix(midi_out_dir, wav.stem, ".mid")
                except FileNotFoundError:
                    pass
            
            # Try basic pitch naming convention
            if not midi_file:
                basic_pitch_name = f"{stem_name}_basic_pitch.mid"
                potential_midi = midi_out_dir / basic_pitch_name
                if potential_midi.exists():
                    midi_file = potential_midi

        if midi_file and midi_file.exists():
            all_midi_paths[stem_name] = midi_file
        else:
            print(f"[warn] No MIDI file found for {stem_name}, skipping.", file=sys.stderr)

    # Select which stems to include in sheet music
    midi_paths = {}
    for stem_name, midi_path in all_midi_paths.items():
        if stem_name == "other" and args.include_other == "no":
            print(f"[music21] Generated MIDI for {stem_name} but excluding from sheet music")
            continue
        midi_paths[stem_name] = midi_path

    if not midi_paths:
        print("[error] No MIDI produced; nothing to score.", file=sys.stderr); sys.exit(4)

    # Auto-detect BPM if not provided
    auto_bpm = detect_bpm_from_midi_files(all_midi_paths) if all_midi_paths else 120
    if not args.tempo or args.tempo <= 0:
        print(f"[CLI] Auto-detected BPM: {auto_bpm}")
    else:
        print(f"[CLI] Using provided BPM: {args.tempo}")
    chosen_bpm = args.tempo if args.tempo and args.tempo > 0 else auto_bpm

    # Generate intelligent piano arrangement if requested
    if args.intelligent_piano:
        from ui_functions import create_intelligent_piano_arrangement_with_ui_params
        import json
        
        # Build parameters from command line args or JSON file
        if args.piano_params and Path(args.piano_params).exists():
            with open(args.piano_params, 'r') as f:
                params = json.load(f)
        else:
            # Use command line parameters
            params = {
                'phraseDuration': args.phrase_duration * 1000,  # Convert to ms
                'phraseOverlap': args.phrase_overlap * 1000,    # Convert to ms
                'minNotesPerSec': args.min_notes_per_sec,
                'maxNotesPerSec': args.max_notes_per_sec,
                'veryDenseRatio': 0.25,
                'denseRatio': args.dense_ratio,
                'mediumRatio': args.medium_ratio,
                'sparseRatio': 0.70,
                'veryShortPenalty': 20,
                'shortPenalty': 10,
                'minDuration': 150,
                'goodDuration': 500,
                'legatoEnabled': args.legato,
                'legatoGapThreshold': args.legato_threshold,
                'minNoteDuration': args.min_note_duration,
                'sustainedHarmony': True,
                'vocalsWeight': 1.0,
                'pianoWeight': 0.8,
                'guitarWeight': 0.6,
                'bassWeight': 0.7,
                'bpm': chosen_bpm
            }
        
        # Generate intelligent piano arrangement
        piano_output = midi_out_dir / f"{input_audio.stem}_intelligent_piano.mid"
        try:
            result = create_intelligent_piano_arrangement_with_ui_params(all_midi_paths, piano_output, params)
            print(f"[CLI] Generated intelligent piano arrangement with {result['noteCount']} notes")
            print(f"[CLI] Saved to: {piano_output}")
        except Exception as e:
            print(f"[CLI] Error generating intelligent piano arrangement: {e}", file=sys.stderr)

    score_dir = out_root / "score"
    out_musicxml = score_dir / f"{input_audio.stem}_combined.musicxml"
    assemble_score(midi_paths, out_musicxml, bpm=chosen_bpm, timesig=args.time_signature, include_drums=(args.include_drums == "yes"))
    print(out_musicxml)

# Video generation section removed per user request

if __name__ == "__main__":
    main()
