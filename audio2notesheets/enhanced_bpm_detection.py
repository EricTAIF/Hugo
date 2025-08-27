#!/usr/bin/env python3
"""
Enhanced BPM Detection Module using Advanced Algorithms (2024)
Combines multiple state-of-the-art approaches for robust tempo estimation.
"""

import numpy as np
import librosa
import mido
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import scipy.signal
import scipy.stats

def enhanced_bpm_detection(audio_path: Path, confidence_threshold: float = 0.7) -> Dict[str, float]:
    """
    Enhanced BPM detection using multiple algorithms and ensemble voting.
    
    Args:
        audio_path: Path to audio file
        confidence_threshold: Minimum confidence for accepting BPM estimate
        
    Returns:
        Dict containing BPM estimate, confidence, and method used
    """
    try:
        # Load audio with librosa
        y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
        
        # Method 1: Enhanced Librosa Beat Tracking with Prior
        bpm_librosa, conf_librosa = detect_bpm_librosa_enhanced(y, sr)
        
        # Method 2: Spectral Flux-based Detection
        bpm_spectral, conf_spectral = detect_bpm_spectral_flux(y, sr)
        
        # Method 3: Autocorrelation-based Detection
        bpm_autocorr, conf_autocorr = detect_bpm_autocorrelation(y, sr)
        
        # Method 4: Multi-scale Analysis
        bpm_multiscale, conf_multiscale = detect_bpm_multiscale(y, sr)
        
        # Ensemble voting with confidence weighting
        candidates = [
            (bpm_librosa, conf_librosa, 'librosa_enhanced'),
            (bpm_spectral, conf_spectral, 'spectral_flux'),
            (bpm_autocorr, conf_autocorr, 'autocorrelation'),
            (bpm_multiscale, conf_multiscale, 'multiscale')
        ]
        
        # Filter out low-confidence estimates
        valid_candidates = [(bpm, conf, method) for bpm, conf, method in candidates 
                           if conf >= confidence_threshold and 40 <= bpm <= 200]
        
        if not valid_candidates:
            # Fall back to best estimate even if low confidence
            best_bpm, best_conf, best_method = max(candidates, key=lambda x: x[1])
            return {
                'bpm': max(60, min(180, best_bpm)),
                'confidence': best_conf,
                'method': f'{best_method}_fallback',
                'all_estimates': candidates
            }
        
        # Weighted average of valid candidates
        total_weight = sum(conf for _, conf, _ in valid_candidates)
        weighted_bpm = sum(bpm * conf for bpm, conf, _ in valid_candidates) / total_weight
        avg_confidence = total_weight / len(valid_candidates)
        
        # Check for octave errors and correct
        final_bpm = correct_octave_errors(weighted_bpm, [bpm for bpm, _, _ in valid_candidates])
        
        return {
            'bpm': final_bpm,
            'confidence': avg_confidence,
            'method': 'ensemble_weighted',
            'all_estimates': candidates
        }
        
    except Exception as e:
        print(f"[enhanced-bpm] Error processing {audio_path}: {e}")
        return {
            'bpm': 120,
            'confidence': 0.1,
            'method': 'fallback',
            'all_estimates': []
        }

def detect_bpm_librosa_enhanced(y: np.ndarray, sr: int) -> Tuple[float, float]:
    """Enhanced librosa beat tracking with improved parameters"""
    try:
        # Use onset detection to improve beat tracking
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
        
        # Enhanced beat tracking with prior distribution
        tempo, beats = librosa.beat.beat_track(
            y=y, sr=sr,
            onset_envelope=librosa.onset.onset_strength(y=y, sr=sr),
            start_bpm=120,
            tightness=100,  # Higher tightness for more stable tempo
            trim=True
        )
        
        # Calculate confidence based on beat consistency
        if len(beats) > 4:
            beat_intervals = np.diff(librosa.frames_to_time(beats, sr=sr))
            consistency = 1.0 - np.std(beat_intervals) / np.mean(beat_intervals)
            confidence = max(0.1, min(1.0, consistency))
        else:
            confidence = 0.3
            
        return float(tempo), confidence
        
    except Exception:
        return 120.0, 0.1

def detect_bpm_spectral_flux(y: np.ndarray, sr: int) -> Tuple[float, float]:
    """BPM detection using spectral flux analysis"""
    try:
        # Compute spectral flux
        stft = librosa.stft(y, hop_length=512)
        mag_spec = np.abs(stft)
        
        # Enhanced spectral flux with frequency weighting
        spectral_flux = np.sum(np.diff(mag_spec, axis=1), axis=0)
        spectral_flux = np.maximum(0, spectral_flux)  # Half-wave rectification
        
        # Apply Gaussian smoothing
        spectral_flux = scipy.signal.gaussian_filter1d(spectral_flux, sigma=2)
        
        # Find peaks in spectral flux
        peaks, properties = scipy.signal.find_peaks(
            spectral_flux,
            height=np.percentile(spectral_flux, 70),
            distance=int(sr * 0.1 / 512)  # Minimum 0.1s between peaks
        )
        
        if len(peaks) < 4:
            return 120.0, 0.1
            
        # Convert peak indices to time
        peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=512)
        
        # Calculate inter-onset intervals
        iois = np.diff(peak_times)
        
        # Filter reasonable IOIs (0.25s to 2.5s)
        valid_iois = iois[(iois > 0.25) & (iois < 2.5)]
        
        if len(valid_iois) < 3:
            return 120.0, 0.2
            
        # Estimate BPM from median IOI
        median_ioi = np.median(valid_iois)
        bpm = 60.0 / median_ioi
        
        # Calculate confidence
        ioi_consistency = 1.0 - np.std(valid_iois) / np.mean(valid_iois)
        confidence = max(0.1, min(1.0, ioi_consistency * 0.8))
        
        return float(bpm), confidence
        
    except Exception:
        return 120.0, 0.1

def detect_bpm_autocorrelation(y: np.ndarray, sr: int) -> Tuple[float, float]:
    """BPM detection using autocorrelation of onset strength"""
    try:
        # Get onset strength signal
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Compute autocorrelation
        autocorr = librosa.autocorrelate(onset_strength)
        
        # Define lag range for reasonable BPMs (40-200 BPM)
        min_lag = int(60.0 / 200.0 * sr / 512)  # 200 BPM
        max_lag = int(60.0 / 40.0 * sr / 512)   # 40 BPM
        
        # Find peaks in autocorrelation within reasonable range
        valid_autocorr = autocorr[min_lag:max_lag]
        peaks, _ = scipy.signal.find_peaks(valid_autocorr, height=0.1)
        
        if len(peaks) == 0:
            return 120.0, 0.1
            
        # Get the strongest peak
        strongest_peak_idx = peaks[np.argmax(valid_autocorr[peaks])]
        lag = strongest_peak_idx + min_lag
        
        # Convert lag to BPM
        bpm = 60.0 / (lag * 512 / sr)
        
        # Calculate confidence based on peak strength
        peak_strength = valid_autocorr[strongest_peak_idx]
        confidence = min(1.0, peak_strength * 2.0)
        
        return float(bpm), confidence
        
    except Exception:
        return 120.0, 0.1

def detect_bpm_multiscale(y: np.ndarray, sr: int) -> Tuple[float, float]:
    """Multi-scale tempo analysis using different time windows"""
    try:
        # Analyze different time scales
        window_sizes = [2048, 4096, 8192]  # Different analysis windows
        bpm_estimates = []
        confidences = []
        
        for win_size in window_sizes:
            try:
                # Compute onset strength with different window sizes
                onset_env = librosa.onset.onset_strength(
                    y=y, sr=sr, 
                    hop_length=win_size//4,
                    n_fft=win_size
                )
                
                # Estimate tempo
                tempo = librosa.beat.tempo(
                    onset_envelope=onset_env,
                    sr=sr,
                    hop_length=win_size//4,
                    start_bpm=120
                )[0]
                
                bpm_estimates.append(tempo)
                confidences.append(0.5)  # Base confidence
                
            except Exception:
                continue
        
        if not bpm_estimates:
            return 120.0, 0.1
            
        # Calculate weighted average
        weights = np.array(confidences)
        weighted_bpm = np.average(bpm_estimates, weights=weights)
        avg_confidence = np.mean(confidences)
        
        return float(weighted_bpm), avg_confidence
        
    except Exception:
        return 120.0, 0.1

def correct_octave_errors(estimated_bpm: float, candidate_bpms: List[float]) -> float:
    """Correct common octave errors (2x, 0.5x, 3x, 0.33x)"""
    target_range = (60, 180)  # Preferred BPM range
    
    # If already in good range, return as-is
    if target_range[0] <= estimated_bpm <= target_range[1]:
        return estimated_bpm
    
    # Try common octave corrections
    corrections = [0.5, 2.0, 1/3, 3.0]
    
    for correction in corrections:
        corrected_bpm = estimated_bpm * correction
        if target_range[0] <= corrected_bpm <= target_range[1]:
            # Check if this correction is supported by other candidates
            support = sum(1 for bpm in candidate_bpms 
                         if abs(bpm * correction - corrected_bpm) < 5)
            if support >= 1:  # At least one other estimate supports this
                return corrected_bpm
    
    # If no good correction found, clamp to reasonable range
    return max(target_range[0], min(target_range[1], estimated_bpm))

def detect_time_signature(y: np.ndarray, sr: int, bpm: float) -> Tuple[int, int]:
    """Detect time signature based on beat pattern analysis"""
    try:
        # Get beat tracking
        _, beats = librosa.beat.beat_track(y=y, sr=sr, bpm=bpm)
        
        if len(beats) < 8:
            return (4, 4)  # Default fallback
        
        # Convert to time
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        # Analyze beat strength patterns
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
        beat_strengths = [onset_strength[min(len(onset_strength)-1, int(beat))] for beat in beats]
        
        # Simple pattern analysis for common time signatures
        if len(beat_strengths) >= 8:
            # Check for strong-weak-medium-weak pattern (4/4)
            pattern_4_4 = analyze_beat_pattern(beat_strengths, 4)
            # Check for strong-weak-weak pattern (3/4)
            pattern_3_4 = analyze_beat_pattern(beat_strengths, 3)
            
            if pattern_4_4 > pattern_3_4:
                return (4, 4)
            else:
                return (3, 4)
        
        return (4, 4)  # Default
        
    except Exception:
        return (4, 4)

def analyze_beat_pattern(beat_strengths: List[float], pattern_length: int) -> float:
    """Analyze how well beat strengths match a given pattern length"""
    if len(beat_strengths) < pattern_length * 2:
        return 0.0
    
    # Group beats by pattern position
    pattern_groups = [[] for _ in range(pattern_length)]
    for i, strength in enumerate(beat_strengths):
        pattern_groups[i % pattern_length].append(strength)
    
    # Calculate average strength for each pattern position
    avg_strengths = [np.mean(group) if group else 0 for group in pattern_groups]
    
    if not any(avg_strengths):
        return 0.0
    
    # Calculate pattern clarity (difference between strong and weak beats)
    max_strength = max(avg_strengths)
    min_strength = min(avg_strengths)
    clarity = (max_strength - min_strength) / (max_strength + 0.001)
    
    return clarity

def enhanced_midi_bpm_detection(midi_path: Path) -> Dict[str, float]:
    """Enhanced BPM detection from MIDI files with better IOI analysis"""
    try:
        midi_file = mido.MidiFile(str(midi_path))
        
        # Method 1: Check tempo meta messages
        tempos = []
        for track in midi_file.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    bpm = 60000000 / msg.tempo
                    if 40 <= bpm <= 200:
                        tempos.append(bpm)
        
        if tempos:
            return {
                'bpm': np.median(tempos),
                'confidence': 0.9,
                'method': 'midi_tempo_meta'
            }
        
        # Method 2: Enhanced IOI analysis
        events = extract_note_events_enhanced(midi_file)
        if not events:
            return {'bpm': 120, 'confidence': 0.1, 'method': 'fallback'}
        
        # Get onset times
        onsets = sorted(set(e['start'] for e in events))
        
        if len(onsets) < 4:
            return {'bpm': 120, 'confidence': 0.2, 'method': 'insufficient_data'}
        
        # Calculate inter-onset intervals
        iois = np.array([onsets[i+1] - onsets[i] for i in range(len(onsets)-1)])
        
        # Filter reasonable IOIs (250ms to 2000ms)
        valid_iois = iois[(iois >= 250) & (iois <= 2000)]
        
        if len(valid_iois) < 3:
            return {'bpm': 120, 'confidence': 0.3, 'method': 'few_valid_iois'}
        
        # Multi-modal IOI analysis for better rhythm detection
        bpm_estimates = []
        
        # Method 2a: Median IOI
        median_ioi = np.median(valid_iois)
        bpm_median = 60000 / median_ioi
        bpm_estimates.append(('median', bpm_median, 0.5))
        
        # Method 2b: Most common IOI (mode)
        hist, edges = np.histogram(valid_iois, bins=20)
        most_common_bin = edges[np.argmax(hist)]
        mode_ioi = (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2
        bpm_mode = 60000 / mode_ioi
        bpm_estimates.append(('mode', bpm_mode, 0.6))
        
        # Method 2c: Clustering-based IOI analysis
        try:
            from sklearn.cluster import KMeans
            if len(valid_iois) >= 6:
                kmeans = KMeans(n_clusters=min(3, len(valid_iois)//2), random_state=42)
                clusters = kmeans.fit(valid_iois.reshape(-1, 1))
                
                # Find the cluster with most points (main rhythm)
                main_cluster_center = clusters.cluster_centers_[
                    np.argmax(np.bincount(clusters.labels_))
                ][0]
                bpm_cluster = 60000 / main_cluster_center
                bpm_estimates.append(('cluster', bpm_cluster, 0.7))
        except ImportError:
            pass
        
        # Choose best estimate
        valid_estimates = [(method, bpm, conf) for method, bpm, conf in bpm_estimates 
                          if 40 <= bpm <= 200]
        
        if valid_estimates:
            best_method, best_bpm, best_conf = max(valid_estimates, key=lambda x: x[2])
            
            # Calculate overall confidence based on IOI consistency
            ioi_std = np.std(valid_iois)
            ioi_mean = np.mean(valid_iois)
            consistency = max(0.1, 1.0 - (ioi_std / ioi_mean))
            final_confidence = best_conf * consistency
            
            return {
                'bpm': best_bpm,
                'confidence': final_confidence,
                'method': f'midi_ioi_{best_method}'
            }
        
        return {'bpm': 120, 'confidence': 0.2, 'method': 'fallback'}
        
    except Exception as e:
        print(f"[enhanced-midi-bpm] Error: {e}")
        return {'bpm': 120, 'confidence': 0.1, 'method': 'error_fallback'}

def extract_note_events_enhanced(midi_file) -> List[Dict]:
    """Enhanced note event extraction with better tempo handling"""
    events = []
    ticks_per_beat = midi_file.ticks_per_beat or 480
    current_tempo = 500000  # Default 120 BPM
    
    for track in midi_file.tracks:
        current_time_ticks = 0
        current_time_ms = 0.0
        active_notes = {}
        
        for msg in track:
            current_time_ticks += msg.time
            
            # Update current time in milliseconds
            if hasattr(msg, 'time') and msg.time > 0:
                delta_ms = mido.tick2second(msg.time, ticks_per_beat, current_tempo) * 1000
                current_time_ms += delta_ms
            
            if msg.type == 'set_tempo':
                current_tempo = msg.tempo
                continue
            
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = (current_time_ms, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    start_time, velocity = active_notes[msg.note]
                    duration = max(1, current_time_ms - start_time)
                    
                    events.append({
                        'note': msg.note,
                        'start': int(start_time),
                        'duration': int(duration),
                        'velocity': velocity
                    })
                    del active_notes[msg.note]
    
    return sorted(events, key=lambda x: x['start'])