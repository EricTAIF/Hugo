#!/usr/bin/env python3
"""
Enhanced Note Extraction and Musical Intelligence Module (2025)
Implements advanced algorithms for better note detection, duration estimation, and harmonic analysis.
"""

import numpy as np
import librosa
import mido
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import scipy.signal
import scipy.stats
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings("ignore")

def enhance_basic_pitch_parameters(stem_name: str) -> Dict[str, float]:
    """
    Enhanced Basic Pitch parameters based on 2025 research for different instruments.
    Uses adaptive thresholds and instrument-specific optimizations.
    """
    base_params = {
        'onset_threshold': 0.5,
        'frame_threshold': 0.3,
        'minimum_note_length': 127.70,  # Default Basic Pitch value
        'minimum_frequency': 80.0,
        'maximum_frequency': 2000.0,
        'multiple_pitch_bends': True,
        'melodia_trick': True
    }
    
    # Instrument-specific parameter optimization based on 2025 research
    if "vocal" in stem_name.lower():
        # Vocals: Optimized for melodic lines with vibrato and pitch bends
        return {
            **base_params,
            'onset_threshold': 0.15,      # Lower for soft vocal onsets
            'frame_threshold': 0.10,      # Lower for sustained notes
            'minimum_note_length': 350.0, # Longer for vocal phrasing
            'minimum_frequency': 80.0,     # Human vocal range
            'maximum_frequency': 1200.0,   # Upper vocal range
            'multiple_pitch_bends': True,  # Essential for vocal expression
            'melodia_trick': True          # Helps with vocal pitch tracking
        }
    elif "bass" in stem_name.lower():
        # Bass: Optimized for low-frequency content and sustained notes
        return {
            **base_params,
            'onset_threshold': 0.3,        # Higher for cleaner bass detection
            'frame_threshold': 0.2,        # Good for sustained bass notes
            'minimum_note_length': 400.0,  # Longer for bass note duration
            'minimum_frequency': 40.0,     # Low bass frequencies
            'maximum_frequency': 400.0,    # Bass range limit
            'multiple_pitch_bends': False, # Less pitch bend in bass
            'melodia_trick': False
        }
    elif "drum" in stem_name.lower():
        # Drums: Optimized for percussive attacks
        return {
            **base_params,
            'onset_threshold': 0.7,        # High for clear drum hits
            'frame_threshold': 0.6,        # High for percussive content
            'minimum_note_length': 50.0,   # Short for drum hits
            'minimum_frequency': 60.0,     # Kick drum range
            'maximum_frequency': 8000.0,   # Cymbals and hi-hats
            'multiple_pitch_bends': False,
            'melodia_trick': False
        }
    elif "guitar" in stem_name.lower():
        # Guitar: Balanced for chord detection and single notes
        return {
            **base_params,
            'onset_threshold': 0.25,       # Good for guitar attacks
            'frame_threshold': 0.2,        # Sustained notes and chords
            'minimum_note_length': 200.0,  # Guitar note duration
            'minimum_frequency': 80.0,     # Low E string
            'maximum_frequency': 2000.0,   # Guitar harmonics
            'multiple_pitch_bends': True,  # Guitar bends
            'melodia_trick': True
        }
    elif "piano" in stem_name.lower():
        # Piano: Optimized for polyphonic content and chord detection
        return {
            **base_params,
            'onset_threshold': 0.2,        # Good for piano attacks
            'frame_threshold': 0.15,       # Sustained piano notes
            'minimum_note_length': 250.0,  # Piano note duration
            'minimum_frequency': 27.5,     # Piano A0
            'maximum_frequency': 4200.0,   # Piano C8 harmonics
            'multiple_pitch_bends': False, # Piano doesn't bend
            'melodia_trick': True          # Helps with polyphonic content
        }
    else:
        # Other instruments: Balanced parameters
        return {
            **base_params,
            'onset_threshold': 0.35,
            'frame_threshold': 0.25,
            'minimum_note_length': 180.0,
            'multiple_pitch_bends': True,
            'melodia_trick': True
        }

def enhance_note_events_with_harmonic_analysis(note_events: List[Dict], audio_path: Path = None) -> List[Dict]:
    """
    Enhance note events with harmonic analysis and improved duration estimation.
    Based on 2025 research in multi-f0 estimation and harmonic envelope detection.
    """
    if not note_events:
        return note_events
    
    enhanced_events = []
    
    # Sort events by start time
    events_sorted = sorted(note_events, key=lambda x: x['start'])
    
    # Harmonic analysis window
    for i, event in enumerate(events_sorted):
        enhanced_event = event.copy()
        
        # Analyze harmonic context
        harmonic_context = analyze_harmonic_context(event, events_sorted, i)
        enhanced_event.update(harmonic_context)
        
        # Improve duration estimation
        improved_duration = estimate_improved_duration(event, events_sorted, i)
        enhanced_event['duration'] = improved_duration
        
        # Add confidence score
        confidence = calculate_note_confidence(enhanced_event, events_sorted)
        enhanced_event['confidence'] = confidence
        
        enhanced_events.append(enhanced_event)
    
    # Post-processing: Remove low-confidence notes and merge duplicates
    filtered_events = post_process_enhanced_events(enhanced_events)
    
    return filtered_events

def analyze_harmonic_context(event: Dict, all_events: List[Dict], index: int) -> Dict[str, any]:
    """
    Analyze harmonic context of a note using surrounding notes.
    Implements 2025 research on harmonic envelope estimation.
    """
    note = event['note']
    start_time = event['start']
    window_ms = 1000  # 1-second analysis window
    
    # Find contemporaneous notes (notes playing at the same time)
    contemporary_notes = []
    for other_event in all_events:
        other_start = other_event['start']
        other_end = other_start + other_event['duration']
        
        # Check if notes overlap in time
        if not (other_end <= start_time or other_start >= start_time + event['duration']):
            if other_event != event:  # Don't include self
                contemporary_notes.append(other_event['note'])
    
    # Analyze harmonic relationships
    harmonic_info = {
        'is_root': False,
        'is_bass': False,
        'harmonic_strength': 0.0,
        'chord_context': None,
        'harmonic_intervals': []
    }
    
    if contemporary_notes:
        # Calculate intervals from this note to others
        intervals = [(other_note - note) % 12 for other_note in contemporary_notes]
        harmonic_info['harmonic_intervals'] = intervals
        
        # Check for common harmonic intervals (3rd, 5th, 7th)
        harmonic_intervals = [3, 4, 7, 10, 11]  # Major/minor 3rd, perfect 5th, etc.
        harmonic_matches = sum(1 for interval in intervals if interval in harmonic_intervals)
        harmonic_info['harmonic_strength'] = harmonic_matches / len(intervals) if intervals else 0
        
        # Determine if this is likely a bass note (lowest in the harmonic context)
        if note <= min(contemporary_notes + [note]):
            harmonic_info['is_bass'] = True
        
        # Simple chord detection
        all_pitches = sorted(set([note] + contemporary_notes))
        if len(all_pitches) >= 3:
            chord_type = detect_chord_type([p % 12 for p in all_pitches])
            harmonic_info['chord_context'] = chord_type
    
    return harmonic_info

def detect_chord_type(pitch_classes: List[int]) -> str:
    """Simple chord type detection based on pitch class intervals"""
    if len(pitch_classes) < 3:
        return "unknown"
    
    # Convert to intervals from root
    root = pitch_classes[0]
    intervals = sorted(set((pc - root) % 12 for pc in pitch_classes))
    
    # Common chord patterns
    if intervals == [0, 4, 7]:
        return "major"
    elif intervals == [0, 3, 7]:
        return "minor"
    elif intervals == [0, 4, 7, 10]:
        return "dominant7"
    elif intervals == [0, 3, 7, 10]:
        return "minor7"
    elif intervals == [0, 4, 7, 11]:
        return "major7"
    elif intervals == [0, 3, 6]:
        return "diminished"
    elif intervals == [0, 4, 8]:
        return "augmented"
    else:
        return f"complex_{len(intervals)}note"

def estimate_improved_duration(event: Dict, all_events: List[Dict], index: int) -> int:
    """
    Improved duration estimation using context analysis.
    Based on 2025 research in note offset detection.
    """
    original_duration = event['duration']
    note = event['note']
    start_time = event['start']
    
    # Find the next note with same pitch (for duration limiting)
    next_same_pitch = None
    for i in range(index + 1, len(all_events)):
        if all_events[i]['note'] == note:
            next_same_pitch = all_events[i]
            break
    
    # Limit duration if same pitch appears again
    if next_same_pitch:
        max_duration = next_same_pitch['start'] - start_time
        original_duration = min(original_duration, max_duration)
    
    # Apply duration smoothing based on harmonic context
    # Notes in harmonic context tend to have more stable durations
    harmonic_neighbors = []
    for other_event in all_events:
        if (abs(other_event['start'] - start_time) < 500 and  # Within 500ms
            other_event != event):
            # Check if harmonically related (octave, fifth, third)
            interval = abs(other_event['note'] - note) % 12
            if interval in [0, 3, 4, 7, 8, 9]:  # Harmonic intervals
                harmonic_neighbors.append(other_event['duration'])
    
    if harmonic_neighbors:
        # Smooth duration towards harmonic neighbors
        neighbor_median = np.median(harmonic_neighbors)
        smoothing_factor = 0.3
        smoothed_duration = (1 - smoothing_factor) * original_duration + smoothing_factor * neighbor_median
        original_duration = int(smoothed_duration)
    
    # Apply minimum and maximum duration limits
    min_duration = 50   # 50ms minimum
    max_duration = 8000 # 8s maximum
    
    return max(min_duration, min(max_duration, original_duration))

def calculate_note_confidence(event: Dict, all_events: List[Dict]) -> float:
    """
    Calculate confidence score for a note based on multiple factors.
    Higher confidence = more likely to be a real musical note.
    """
    confidence = 0.5  # Base confidence
    
    # Factor 1: Duration (longer notes are more confident)
    duration = event['duration']
    if duration >= 500:
        confidence += 0.3
    elif duration >= 200:
        confidence += 0.2
    elif duration < 100:
        confidence -= 0.2
    
    # Factor 2: Harmonic strength
    if 'harmonic_strength' in event:
        confidence += event['harmonic_strength'] * 0.3
    
    # Factor 3: Pitch range (middle ranges are more confident)
    note = event['note']
    if 36 <= note <= 84:  # Piano range C2-C6
        confidence += 0.1
    elif note < 24 or note > 96:  # Very high/low notes less confident
        confidence -= 0.2
    
    # Factor 4: Velocity (if available)
    if 'velocity' in event:
        velocity = event['velocity']
        if velocity >= 64:  # Strong attack
            confidence += 0.1
        elif velocity < 32:  # Very soft
            confidence -= 0.1
    
    # Factor 5: Bass notes get bonus confidence
    if event.get('is_bass', False):
        confidence += 0.1
    
    return max(0.1, min(1.0, confidence))

def post_process_enhanced_events(events: List[Dict]) -> List[Dict]:
    """
    Post-process enhanced events to remove duplicates and low-confidence notes.
    """
    # Sort by confidence (descending)
    events_sorted = sorted(events, key=lambda x: x.get('confidence', 0.5), reverse=True)
    
    # Remove low-confidence notes
    min_confidence = 0.3
    filtered_events = [e for e in events_sorted if e.get('confidence', 0.5) >= min_confidence]
    
    # Remove near-duplicate notes (same pitch within 100ms)
    final_events = []
    for event in filtered_events:
        is_duplicate = False
        for existing in final_events:
            if (existing['note'] == event['note'] and
                abs(existing['start'] - event['start']) < 100):
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_events.append(event)
    
    # Sort by start time
    return sorted(final_events, key=lambda x: x['start'])

def create_intelligent_phrase_detection(events: List[Dict], bpm: float) -> List[Dict]:
    """
    Advanced phrase detection using musical structure analysis.
    Based on 2025 research in music structure analysis.
    """
    if not events:
        return []
    
    # Calculate beat length in milliseconds
    beat_ms = 60000 / max(1, bpm)
    measure_ms = beat_ms * 4  # Assuming 4/4 time
    
    phrases = []
    events_sorted = sorted(events, key=lambda x: x['start'])
    total_duration = max(e['start'] + e['duration'] for e in events_sorted)
    
    # Detect phrase boundaries using multiple criteria
    phrase_boundaries = detect_phrase_boundaries(events_sorted, measure_ms)
    
    # Create phrases based on boundaries
    for i in range(len(phrase_boundaries) - 1):
        start_time = phrase_boundaries[i]
        end_time = phrase_boundaries[i + 1]
        
        # Get events in this phrase
        phrase_events = [e for e in events_sorted 
                        if start_time <= e['start'] < end_time]
        
        if phrase_events:
            # Analyze phrase characteristics
            phrase_analysis = analyze_phrase_characteristics(phrase_events, measure_ms)
            
            phrases.append({
                'events': phrase_events,
                'start': start_time,
                'end': end_time,
                'duration_ms': end_time - start_time,
                'note_count': len(phrase_events),
                'density': phrase_analysis['density'],
                'harmonic_complexity': phrase_analysis['harmonic_complexity'],
                'rhythmic_activity': phrase_analysis['rhythmic_activity'],
                'is_climax': phrase_analysis['is_climax']
            })
    
    return phrases

def detect_phrase_boundaries(events: List[Dict], measure_ms: float) -> List[int]:
    """
    Detect phrase boundaries using silence analysis and harmonic changes.
    """
    boundaries = [0]  # Start with beginning
    
    # Method 1: Silence-based boundaries (gaps in music)
    for i in range(len(events) - 1):
        current_end = events[i]['start'] + events[i]['duration']
        next_start = events[i + 1]['start']
        gap = next_start - current_end
        
        # If gap is longer than half a measure, it's likely a phrase boundary
        if gap > measure_ms * 0.5:
            boundaries.append(next_start)
    
    # Method 2: Harmonic change boundaries
    window_size = int(measure_ms * 2)  # 2-measure windows
    for i in range(0, len(events) - 4, 4):  # Check every 4 notes
        if i + 8 < len(events):
            # Analyze harmonic content before and after
            before_notes = [e['note'] % 12 for e in events[i:i+4]]
            after_notes = [e['note'] % 12 for e in events[i+4:i+8]]
            
            # If pitch content changes significantly, mark boundary
            before_set = set(before_notes)
            after_set = set(after_notes)
            overlap = len(before_set & after_set)
            total_unique = len(before_set | after_set)
            
            if total_unique > 0 and overlap / total_unique < 0.5:
                boundary_time = events[i + 4]['start']
                if boundary_time not in boundaries:
                    boundaries.append(boundary_time)
    
    # Add end boundary
    if events:
        end_time = max(e['start'] + e['duration'] for e in events)
        boundaries.append(end_time)
    
    return sorted(set(boundaries))

def analyze_phrase_characteristics(events: List[Dict], measure_ms: float) -> Dict[str, float]:
    """
    Analyze musical characteristics of a phrase.
    """
    if not events:
        return {
            'density': 0.0,
            'harmonic_complexity': 0.0,
            'rhythmic_activity': 0.0,
            'is_climax': False
        }
    
    duration_s = (max(e['start'] + e['duration'] for e in events) - 
                  min(e['start'] for e in events)) / 1000
    
    # Density: notes per second
    density = len(events) / max(0.1, duration_s)
    
    # Harmonic complexity: unique pitch classes
    pitch_classes = set(e['note'] % 12 for e in events)
    harmonic_complexity = len(pitch_classes) / 12  # Normalized to 0-1
    
    # Rhythmic activity: variation in onset timing
    onsets = [e['start'] for e in events]
    if len(onsets) > 1:
        onset_intervals = np.diff(onsets)
        rhythmic_activity = np.std(onset_intervals) / np.mean(onset_intervals)
    else:
        rhythmic_activity = 0.0
    
    # Climax detection: high pitch + high density + loud dynamics
    avg_pitch = np.mean([e['note'] for e in events])
    avg_velocity = np.mean([e.get('velocity', 64) for e in events])
    is_climax = (avg_pitch > 72 and density > 5 and avg_velocity > 80)
    
    return {
        'density': min(10.0, density),  # Cap at reasonable value
        'harmonic_complexity': harmonic_complexity,
        'rhythmic_activity': min(2.0, rhythmic_activity),
        'is_climax': is_climax
    }

def create_adaptive_arrangement(phrases: List[Dict], params: Dict) -> List[Dict]:
    """
    Create arrangement that adapts to phrase characteristics.
    Uses 2025 research in adaptive music arrangement.
    """
    arrangement = []
    
    for phrase in phrases:
        # Adapt selection strategy based on phrase characteristics
        if phrase['is_climax']:
            # Climax phrases: keep more notes for dramatic effect
            selection_ratio = 0.8
            min_notes = min(20, len(phrase['events']))
        elif phrase['harmonic_complexity'] > 0.6:
            # Complex harmony: moderate selection to preserve richness
            selection_ratio = 0.6
            min_notes = min(15, len(phrase['events']))
        elif phrase['density'] > 8:
            # High density: more aggressive filtering
            selection_ratio = 0.4
            min_notes = min(12, len(phrase['events']))
        else:
            # Normal phrases: balanced selection
            selection_ratio = 0.5
            min_notes = min(10, len(phrase['events']))
        
        target_notes = max(min_notes, int(len(phrase['events']) * selection_ratio))
        
        # Select notes using enhanced scoring
        selected_notes = select_notes_with_enhanced_scoring(
            phrase['events'], target_notes, phrase, params
        )
        
        arrangement.extend(selected_notes)
    
    return arrangement

def select_notes_with_enhanced_scoring(events: List[Dict], target_count: int, 
                                     phrase: Dict, params: Dict) -> List[Dict]:
    """
    Enhanced note selection using 2025 research in musical importance scoring.
    """
    scored_events = []
    
    for event in events:
        score = calculate_enhanced_importance_score(event, phrase, params)
        scored_events.append((score, event))
    
    # Sort by score and select top notes
    scored_events.sort(key=lambda x: x[0], reverse=True)
    selected = [event for score, event in scored_events[:target_count]]
    
    return selected

def calculate_enhanced_importance_score(event: Dict, phrase: Dict, params: Dict) -> float:
    """
    Calculate enhanced importance score using multiple musical factors.
    Based on 2025 research in computational musicology.
    """
    score = 0.0
    
    # Base confidence score
    score += event.get('confidence', 0.5) * 10
    
    # Duration factor (with diminishing returns)
    duration = event['duration']
    if duration < 100:
        score -= 5  # Penalty for very short notes
    else:
        score += min(8, np.log(duration / 100) * 3)
    
    # Harmonic importance
    if event.get('is_bass', False):
        score += 6  # Bass notes are structurally important
    
    harmonic_strength = event.get('harmonic_strength', 0)
    score += harmonic_strength * 5
    
    # Pitch range preferences
    note = event['note']
    if 48 <= note <= 84:  # Sweet spot for piano arrangement
        score += 4
    elif 36 <= note < 48 or 84 < note <= 96:  # Extended but acceptable
        score += 2
    else:
        score -= 2  # Very high or low notes
    
    # Rhythmic position (beat alignment)
    if phrase and 'rhythmic_activity' in phrase:
        # In active phrases, prefer notes on strong beats
        beat_ms = 60000 / params.get('bpm', 120)
        beat_position = (event['start'] % beat_ms) / beat_ms
        if beat_position < 0.1 or beat_position > 0.9:  # Close to beat
            score += 3
    
    # Phrase context bonuses
    if phrase:
        if phrase['is_climax']:
            score += 2  # Bonus in climax sections
        
        # Bonus for notes in harmonically complex sections
        if phrase['harmonic_complexity'] > 0.6:
            score += harmonic_strength * 2
    
    # Velocity/dynamics (if available)
    velocity = event.get('velocity', 64)
    if velocity >= 80:
        score += 2  # Strong notes are important
    elif velocity < 40:
        score -= 1  # Weak notes less important
    
    return score