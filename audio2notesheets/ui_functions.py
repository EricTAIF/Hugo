#!/usr/bin/env python3

"""
UI-specific functions that support parameterized piano arrangement generation
"""

import mido
from pathlib import Path
import sys
sys.path.append('.')

from main import (extract_all_musical_events, analyze_harmony, 
                 optimize_voice_leading, ensure_piano_playability, convert_to_midi,
                 get_part_importance, detect_bpm_from_midi_files,
                 quantize_and_merge, generate_sustain_pedal_events,
                 parse_context_file)
from ai_arranger import generate_with_ai

def detect_musical_phrases_with_params(events, params):
    """Detect musical phrases with UI parameters"""
    phrases = []
    if not events:
        return phrases
    
    # Use UI parameters
    phrase_duration = params['phraseDuration']  # Already in ms
    overlap = params['phraseOverlap']           # Already in ms
    
    events_sorted = sorted(events, key=lambda x: x['start'])
    total_duration = max(e['start'] + e['duration'] for e in events_sorted)
    
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
    
    print(f"[UI] Created {len(phrases)} phrases from {total_duration/1000:.1f}s of music")
    return phrases

def create_layered_arrangement_with_params(phrases, harmonic_analysis, params):
    """Create arrangement with UI parameters"""
    arrangement = []
    
    for phrase in phrases:
        phrase_start = phrase['start']
        phrase_end = phrase['end']
        phrase_density = phrase['density']
        
        # Use UI parameters for note density
        phrase_length_seconds = (phrase_end - phrase_start) / 1000
        min_notes_per_phrase = max(15, int(phrase_length_seconds * params['minNotesPerSec']))
        max_notes_per_phrase = int(phrase_length_seconds * params['maxNotesPerSec'])
        
        available_notes = len(phrase['events'])
        
        # Use UI ratios for different density levels
        if phrase_density > 20:  # Very high density phrase
            target_notes = min(int(available_notes * params['veryDenseRatio']), max_notes_per_phrase)
            target_notes = max(target_notes, min_notes_per_phrase)
            style = 'very_dense'
        elif phrase_density > 15:  # High density phrase
            target_notes = min(int(available_notes * params['denseRatio']), max_notes_per_phrase)
            target_notes = max(target_notes, min_notes_per_phrase)
            style = 'dense'
        elif phrase_density > 8:  # Medium density
            target_notes = min(int(available_notes * params['mediumRatio']), max_notes_per_phrase) 
            target_notes = max(target_notes, min_notes_per_phrase)
            style = 'medium'
        else:  # Low density
            target_notes = min(int(available_notes * params['sparseRatio']), max_notes_per_phrase)
            target_notes = max(target_notes, min_notes_per_phrase // 2)
            style = 'sparse'
        
        print(f"[UI] Phrase {len(arrangement)+1}: {style} style, target {target_notes} notes")
        
        # Select best notes for this phrase
        phrase_arrangement = select_phrase_notes_with_params(phrase, target_notes, harmonic_analysis, params)
        arrangement.extend(phrase_arrangement)
    
    return arrangement

def select_phrase_notes_with_params(phrase, target_count, harmonic_analysis, params):
    """Select the most important notes for a phrase using UI parameters"""
    events = phrase['events']
    
    # Score each note based on multiple factors
    scored_events = []
    for event in events:
        score = calculate_note_importance_with_params(event, phrase, harmonic_analysis, params)
        scored_events.append((score, event))
    
    # Sort by score and take top notes
    scored_events.sort(key=lambda x: x[0], reverse=True)
    selected_events = [event for score, event in scored_events[:target_count]]
    
    return selected_events

def calculate_note_importance_with_params(event, phrase, harmonic_analysis, params):
    """Calculate importance score using UI parameters"""
    score = 0.0
    duration = event['duration']
    
    # Factor 1: Duration filtering with UI parameters
    if duration < params['minDuration']:
        score -= params['veryShortPenalty']
    elif duration < params['minDuration'] * 2:  # Up to 2x min duration
        score -= params['shortPenalty']
    elif duration < params['goodDuration']:
        score += duration / 200  # Small bonus
    else:  # Good duration
        score += min(duration / 200, 10)  # Good bonus, capped at 10
    
    # Factor 2: Part importance using UI weights
    part_weights = {
        'vocals': params['vocalsWeight'],
        'piano': params['pianoWeight'], 
        'guitar': params['guitarWeight'],
        'bass': params['bassWeight'],
        'other': 0.4
    }
    part_weight = part_weights.get(event.get('part', 'other'), 0.5)
    score += event.get('importance', 1.0) * part_weight * 8
    
    # Factor 3: Pitch range suitability
    note = event['note']
    if event.get('part') == 'vocals' and 60 <= note <= 84:  # Melody range
        score += 8
    elif event.get('part') == 'bass' and 24 <= note <= 55:  # Bass range
        score += 6
    elif 48 <= note <= 72:  # Harmony range
        score += 4
    
    # Factor 4: Harmonic relevance
    event_time = event['start']
    relevant_harmony = next((h for h in harmonic_analysis 
                           if h['start'] <= event_time < h['end']), None)
    if relevant_harmony:
        pitch_class = note % 12
        if pitch_class in relevant_harmony['chord_tones'][:3]:
            score += 6
        elif pitch_class in relevant_harmony['chord_tones']:
            score += 3
    
    # Factor 5: Avoid overly high/low notes
    if note < 24 or note > 96:
        score -= 5
    
    # Factor 6: Enhanced BPM-aware timing preference
    bpm = params.get('bpm', 120)
    beat_ms = max(1, int(60000 / max(1, bpm)))
    
    # Check alignment with different rhythmic levels
    beat_offset = event['start'] % beat_ms
    beat_distance = min(beat_offset, beat_ms - beat_offset)
    
    # Strong beats (downbeats) - highest priority
    if beat_distance <= 0.1 * beat_ms:  # within 10% of beat
        score += 12  # Strong bonus for beat alignment
    elif beat_distance <= 0.15 * beat_ms:  # within 15% of beat
        score += 8   # Good bonus for near-beat timing
    
    # Check eighth note alignment (subdivisions)
    eighth_ms = beat_ms / 2
    eighth_offset = event['start'] % eighth_ms
    eighth_distance = min(eighth_offset, eighth_ms - eighth_offset)
    if eighth_distance <= 0.1 * eighth_ms:
        score += 6   # Moderate bonus for eighth note timing
    
    # Check sixteenth note alignment (finest subdivision)
    sixteenth_ms = beat_ms / 4
    sixteenth_offset = event['start'] % sixteenth_ms
    sixteenth_distance = min(sixteenth_offset, sixteenth_ms - sixteenth_offset)
    if sixteenth_distance <= 0.1 * sixteenth_ms:
        score += 3   # Small bonus for sixteenth note timing
    
    return score

def create_intelligent_piano_arrangement_with_ui_params(midi_files, output_midi, params):
    """Create intelligent piano arrangement with UI parameters"""
    # Create new MIDI file
    new_midi = mido.MidiFile()
    new_track = mido.MidiTrack()
    new_midi.tracks.append(new_track)
    
    output_lines = []
    output_lines.append(f"[UI] Creating intelligent piano arrangement with custom parameters...")
    
    # Detect BPM from available MIDI files with enhanced detection
    bpm = detect_bpm_from_midi_files(midi_files)
    
    # If enhanced BPM detection is enabled, try to get more accurate BPM from audio
    try:
        from enhanced_bpm_detection import enhanced_bpm_detection
        from pathlib import Path
        
        if params.get('enableEnhancedBPM', True):
            # Look for original audio file to get better BPM
            data_dirs = [Path("../../data"), Path("../data"), Path("data")]
            for data_dir in data_dirs:
                if data_dir.exists():
                    # Try to find the original audio file
                    song_name = list(midi_files.keys())[0] if midi_files else ""
                    if song_name:
                        # Extract song name from MIDI path
                        midi_path = next(iter(midi_files.values()))
                        song_dir_name = midi_path.parent.name
                        
                        for ext in ['.mp3', '.wav', '.flac']:
                            audio_file = data_dir / f"{song_dir_name}{ext}"
                            if audio_file.exists():
                                print(f"[UI] Found audio file for enhanced BPM: {audio_file}")
                                enhanced_result = enhanced_bpm_detection(audio_file, 
                                    confidence_threshold=params.get('bpmConfidenceThreshold', 0.7))
                                
                                if enhanced_result['confidence'] >= params.get('bpmConfidenceThreshold', 0.7):
                                    bpm = enhanced_result['bpm']
                                    print(f"[UI] Using enhanced BPM: {bpm:.1f} (confidence: {enhanced_result['confidence']:.2f}, method: {enhanced_result['method']})")
                                    output_lines.append(f"[UI] Enhanced BPM Detection: {bpm:.1f} BPM (confidence: {enhanced_result['confidence']:.2f})")
                                    break
                                else:
                                    print(f"[UI] Enhanced BPM confidence too low ({enhanced_result['confidence']:.2f}), using MIDI BPM: {bpm}")
                                break
                        else:
                            continue
                        break
    except ImportError:
        print(f"[UI] Enhanced BPM detection not available, using MIDI BPM: {bpm}")
    except Exception as e:
        print(f"[UI] Enhanced BPM detection failed: {e}, using MIDI BPM: {bpm}")
    
    print(f"[UI] Final BPM: {bpm}")
    output_lines.append(f"[UI] Final BPM: {bpm}")
    tempo_us = int(round(60000000 / max(1, bpm)))
    # Add tempo and time signature
    new_track.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
    new_track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    
    # Step 1: Extract and analyze all musical content
    all_events = extract_all_musical_events(midi_files)
    if not all_events:
        raise Exception("No musical events found")
    
    # Enrich params with detected bpm for downstream scoring/placement
    params = dict(params)
    params.setdefault('bpm', bpm)

    # Optional: load context file and apply tempo/harmony guidance
    context = None
    context_path = params.get('__contextPath')
    if context_path:
        try:
            context = parse_context_file(Path(context_path))
            output_lines.append(f"[UI] Context loaded (tempo={context.get('bpm')}, key={context.get('key')})")
            # Tempo from context
            if params.get('useContextTempo') and context.get('bpm'):
                bpm = int(round(context['bpm']))
                params['bpm'] = bpm
                tempo_us = int(round(60000000 / max(1, bpm)))
                # Overwrite first tempo meta
                if new_track and len(new_track) > 0 and getattr(new_track[0], 'type', None) == 'set_tempo':
                    new_track[0].tempo = tempo_us
                else:
                    new_track.insert(0, mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
                output_lines.append(f"[UI] Using context BPM: {bpm}")
        except Exception as e:
            output_lines.append(f"[UI] Context load failed: {e}")

    # Step 2: Detect musical phrases with custom parameters
    phrases = detect_musical_phrases_with_params(all_events, params)
    output_lines.append(f"[UI] Detected {len(phrases)} musical phrases")
    
    # Step 3: Analyze harmonic content (or use context chords)
    use_context_harmony = bool(params.get('useContextHarmony')) and context and context.get('chords')
    if use_context_harmony:
        # Convert context chords in beats to ms
        beat_ms = 60000.0 / max(1, bpm)
        harmonic_analysis = []
        for ch in context['chords']:
            harmonic_analysis.append({
                'start': int(round(ch['start_beats'] * beat_ms)),
                'end': int(round(ch['end_beats'] * beat_ms)),
                'chord_tones': ch['chord_tones']
            })
        output_lines.append(f"[UI] Using {len(harmonic_analysis)} context harmonic regions")
    else:
        harmonic_analysis = analyze_harmony(all_events)
        output_lines.append(f"[UI] Analyzed {len(harmonic_analysis)} harmonic regions")
    
    # Step 4: Create intelligent layered arrangement with custom parameters
    arrangement = create_layered_arrangement_with_params(phrases, harmonic_analysis, params)
    
    # Step 5: Apply voice leading optimization
    optimized_arrangement = optimize_voice_leading(arrangement)

    # Optional: AI generation and blending
    if params.get('aiEnabled'):
        try:
            primer_source = params.get('aiPrimerSource', 'context')  # 'context'|'vocals'|'first_phrase'
            primer_events = []
            if primer_source == 'context' and context and context.get('intro'):
                beat_ms = 60000.0 / max(1, bpm)
                # Take first bars worth of context intro as primer
                max_ms = int((params.get('aiPrimerBars', 2)) * 4 * beat_ms)
                for n in context['intro']:
                    start_ms = int(round(n['start_beats'] * beat_ms))
                    dur_ms = int(round(n['dur_beats'] * beat_ms))
                    if start_ms > max_ms:
                        continue
                    primer_events.append({'note': int(n['note']), 'start': start_ms, 'duration': max(60, dur_ms), 'velocity': int(n.get('velocity', 80))})
            elif primer_source == 'vocals' and 'vocals' in midi_files:
                vocal_events = extract_all_musical_events({'vocals': midi_files['vocals']})
                primer_events = [
                    {'note': ev['note'], 'start': ev['start'], 'duration': ev['duration'], 'velocity': int(ev.get('velocity', 80))}
                    for ev in vocal_events[:200]
                ]
            else:  # first_phrase of current selection
                if phrases:
                    ph = phrases[0]
                    # Focus on melody-range notes for better AI primer
                    melody_events = [ev for ev in ph['events'] if 60 <= ev['note'] <= 84]  # Melody range
                    if not melody_events:  # Fallback to all events if no melody found
                        melody_events = ph['events']
                    
                    primer_events = [
                        {'note': ev['note'], 'start': ev['start'] - ph['start'], 'duration': ev['duration'], 'velocity': int(ev.get('velocity', 80))}
                        for ev in melody_events[:50]  # Limit to first 50 melody notes for better AI focus
                    ]
                    print(f"[AI] First phrase primer: {len(primer_events)} melody events from phrase with {len(ph['events'])} total events")

            ai_params = {
                'aiModel': params.get('aiModel', 'music_transformer'),
                'aiBars': params.get('aiBars', 8),
                'aiTemperature': params.get('aiTemperature', 1.0),
                'aiBeamSize': params.get('aiBeamSize', 1),
                'useGPU': params.get('useGPU', True),
                'checkpoint': params.get('aiCheckpoint')
            }
            ai_events = generate_with_ai(primer_events, int(bpm), ai_params)
            blend = params.get('aiBlendMode', 'replace_intro')  # 'replace_intro'|'overlay'|'append'
            if ai_events:
                if blend == 'append':
                    shift = (max([n['start'] + n['duration'] for n in optimized_arrangement]) if optimized_arrangement else 0) + 50
                    for e in ai_events:
                        e['start'] += shift
                    optimized_arrangement.extend(ai_events)
                elif blend == 'overlay':
                    optimized_arrangement.extend(ai_events)
                else:  # replace_intro
                    ai_end = max(e['start'] + e['duration'] for e in ai_events)
                    optimized_arrangement = [n for n in optimized_arrangement if n['start'] >= ai_end]
                    optimized_arrangement.extend(ai_events)
                optimized_arrangement.sort(key=lambda x: x['start'])
                output_lines.append(f"[AI] Added {len(ai_events)} AI-generated notes (blend={blend})")
        except Exception as e:
            output_lines.append(f"[AI] Generation failed: {e}")
    
    # Step 6: Create smoother, more connected arrangement
    if params.get('legatoEnabled', True):
        smoothed_arrangement = create_smooth_legato_arrangement(optimized_arrangement, params)
    else:
        smoothed_arrangement = optimized_arrangement
    
    # Step 7: Ensure piano playability
    playable = ensure_piano_playability(smoothed_arrangement)

    # Optional: Use context as intro
    if context and params.get('useContextIntro'):
        try:
            intro_sec = int(params.get('contextIntroSeconds', 8))
            beat_ms = 60000.0 / max(1, bpm)
            intro_ms = intro_sec * 1000
            # Build intro notes from context up to intro_ms
            intro_notes = []
            for n in context.get('intro', []):
                start_ms = int(round(n['start_beats'] * beat_ms))
                dur_ms = int(round(n['dur_beats'] * beat_ms))
                if start_ms >= intro_ms:
                    continue
                note_num = int(n['note'])
                # Fit into piano range by octave shifting
                while note_num < 24:
                    note_num += 12
                while note_num > 96:
                    note_num -= 12
                intro_notes.append({'note': note_num, 'start': start_ms, 'duration': max(100, dur_ms), 'velocity': int(n.get('velocity',80)), 'part': 'context', 'importance': 1.0})
            if intro_notes:
                # Remove original notes in intro window
                playable = [n for n in playable if n['start'] >= intro_ms]
                # Keep context intro
                playable.extend(intro_notes)
                playable.sort(key=lambda x: x['start'])
                output_lines.append(f"[UI] Applied context intro ({len(intro_notes)} notes, {intro_sec}s)")
        except Exception as e:
            output_lines.append(f"[UI] Context intro failed: {e}")

    # Step 8: Quantize/merge and prepare sustain pedal (to reduce choppiness)
    grid = params.get('quantizeGrid', '1/16')  # 'none', '1/8', '1/16'
    merge_gap = int(params.get('mergeGapMs', 80))
    min_dur = int(params.get('minNoteDuration', 200))
    final_arrangement = quantize_and_merge(playable, bpm=bpm, grid=(None if str(grid).lower()=='none' else grid), merge_gap_ms=merge_gap, min_duration_ms=min_dur)
    pedal_enabled = bool(params.get('sustainPedal', True))
    pedal_window = params.get('pedalWindowMs')
    pedal_events = generate_sustain_pedal_events(final_arrangement, bpm=bpm, enabled=pedal_enabled, window_ms=(int(pedal_window) if pedal_window else None))
    
    output_lines.append(f"[UI] Final arrangement: {len(final_arrangement)} notes")
    
    # Convert to MIDI with tempo-aware tick mapping and sustain pedal
    convert_to_midi(final_arrangement, new_track, bpm=bpm, ticks_per_beat=new_midi.ticks_per_beat, control_events=pedal_events)
    new_track.append(mido.MetaMessage('end_of_track', time=0))
    new_midi.save(str(output_midi))
    
    output_lines.append(f"[UI] Saved arrangement: {output_midi.name}")
    
    # Prepare lightweight events for UI (note, start, duration, velocity)
    ui_events = [
        {
            'note': int(n['note']),
            'start': int(n['start']),
            'duration': int(n['duration']),
            'velocity': int(n.get('velocity', 70))
        }
        for n in final_arrangement
    ]

    return {
        'output': '\n'.join(output_lines),
        'noteCount': len(final_arrangement),
        'phrases': len(phrases),
        'bpm': bpm,
        'events': ui_events
    }

def create_smooth_legato_arrangement(arrangement, params):
    """Create smoother, more connected arrangement by filling gaps and extending notes"""
    if not arrangement:
        return arrangement
    
    # Sort by start time
    sorted_notes = sorted(arrangement, key=lambda x: x['start'])
    smooth_arrangement = []
    
    # Group notes by voice/register for better legato connections
    melody_notes = [n for n in sorted_notes if n.get('part') == 'vocals' or (n['note'] >= 60 and n['note'] <= 84)]
    bass_notes = [n for n in sorted_notes if n.get('part') == 'bass' or (n['note'] >= 24 and n['note'] <= 55)]
    harmony_notes = [n for n in sorted_notes if n not in melody_notes and n not in bass_notes]
    
    # Process each voice separately for smooth connections
    smooth_melody = create_legato_voice(melody_notes, voice_type='melody', params=params)
    smooth_bass = create_legato_voice(bass_notes, voice_type='bass', params=params)
    smooth_harmony = create_legato_voice(harmony_notes, voice_type='harmony', params=params)
    
    # Combine all voices
    smooth_arrangement.extend(smooth_melody)
    smooth_arrangement.extend(smooth_bass)
    smooth_arrangement.extend(smooth_harmony)
    
    print(f"[smooth] Created legato arrangement: {len(arrangement)} -> {len(smooth_arrangement)} notes")
    return smooth_arrangement

def create_legato_voice(notes, voice_type='melody', params=None):
    """Create legato connections within a single voice"""
    if len(notes) < 2:
        return notes
    
    if params is None:
        params = {'legatoGapThreshold': 300, 'minNoteDuration': 200}
    
    # Sort by start time
    notes = sorted(notes, key=lambda x: x['start'])
    legato_notes = []
    
    legato_threshold = params.get('legatoGapThreshold', 300)
    min_note_duration = params.get('minNoteDuration', 200)
    
    for i, note in enumerate(notes):
        current_note = note.copy()
        
        # Find the next note in this voice
        next_note = notes[i + 1] if i + 1 < len(notes) else None
        
        if next_note:
            gap = next_note['start'] - (current_note['start'] + current_note['duration'])
            
            # If gap is small (< legato_threshold), create legato connection
            if 0 <= gap <= legato_threshold:
                # Extend current note to connect with next note
                current_note['duration'] = next_note['start'] - current_note['start']
                
            # If gap is medium, add connecting note for bass/harmony
            elif legato_threshold < gap <= (legato_threshold * 2.5) and voice_type in ['bass', 'harmony']:
                # Add the original note
                legato_notes.append(current_note)
                
                # Create a connecting note (lower velocity, bridge note)
                bridge_note = {
                    'note': current_note['note'],
                    'start': current_note['start'] + current_note['duration'],
                    'duration': gap,
                    'velocity': max(25, current_note.get('velocity', 64) - 25),
                    'part': current_note.get('part', 'harmony'),
                    'importance': current_note.get('importance', 1.0) * 0.3
                }
                legato_notes.append(bridge_note)
                continue
        
        # Ensure minimum note duration for smoothness
        if current_note['duration'] < min_note_duration:
            current_note['duration'] = min_note_duration
        
        legato_notes.append(current_note)
    
    return legato_notes

def create_sustained_harmonies(harmony_notes):
    """Create sustained harmony notes to fill gaps and provide continuity"""
    if not harmony_notes:
        return []
    
    sustained_notes = []
    
    # Group harmony notes by time windows (every 2 seconds)
    time_windows = {}
    for note in harmony_notes:
        window = note['start'] // 2000  # 2-second windows
        if window not in time_windows:
            time_windows[window] = []
        time_windows[window].append(note)
    
    # Create sustained harmonies for each window
    for window, notes in time_windows.items():
        if len(notes) < 2:
            sustained_notes.extend(notes)
            continue
        
        # Find the most important harmony notes in this window
        notes_by_importance = sorted(notes, key=lambda x: x.get('importance', 1.0), reverse=True)
        top_harmony_notes = notes_by_importance[:3]  # Take top 3 harmony notes
        
        window_start = window * 2000
        window_end = window_start + 2000
        
        # Extend these harmony notes to fill the window
        for note in top_harmony_notes:
            sustained_note = note.copy()
            sustained_note['start'] = max(sustained_note['start'], window_start)
            sustained_note['duration'] = min(window_end - sustained_note['start'], 1800)  # Leave small gap
            sustained_note['velocity'] = max(25, sustained_note['velocity'] - 15)  # Softer for background
            sustained_notes.append(sustained_note)
    
    return sustained_notes
