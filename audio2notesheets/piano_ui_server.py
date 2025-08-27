#!/usr/bin/env python3

import json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, send_file
import sys
import tempfile
import traceback

# Add current directory to path
sys.path.append('.')

# Import our UI functions
from ui_functions import create_intelligent_piano_arrangement_with_ui_params
from main import run_demucs, transcribe_with_basic_pitch
import os
import werkzeug

# Import enhanced modules
try:
    from enhanced_bpm_detection import enhanced_bpm_detection, enhanced_midi_bpm_detection
    from enhanced_note_extraction import enhance_note_events_with_harmonic_analysis
    from test_enhancements import (
        run_bpm_comparison_test, run_midi_analysis_test, 
        run_arrangement_quality_test, analyze_arrangement_complexity
    )
    ENHANCED_MODULES_AVAILABLE = True
    print("[UI Server] Enhanced 2025 modules loaded successfully")
except ImportError as e:
    print(f"[UI Server] Enhanced modules not available: {e}")
    ENHANCED_MODULES_AVAILABLE = False

app = Flask(__name__)

# Serve the HTML UI
@app.route('/')
def index():
    return send_from_directory('.', 'piano_ui.html')

# API endpoint for generating arrangements
@app.route('/generate', methods=['POST'])
def generate_arrangement():
    try:
        params = request.json
        song_name = params['song']
        arrangement_mode = params.get('arrangementMode', 'combined')
        single_stem = params.get('stem')
        
        print(f"[UI] Generating arrangement for: {song_name}")
        print(f"[UI] Parameters: {json.dumps(params, indent=2)}")
        
        # Set up paths
        midi_dir = Path(f"outputs/midi/{song_name}")
        if not midi_dir.exists():
            return jsonify({
                'success': False,
                'error': f'Song directory not found: {midi_dir}'
            })
        
        # Create temporary output file
        output_midi = midi_dir / f"intelligent_piano_ui_{hash(json.dumps(params, sort_keys=True)) % 10000}.mid"
        
        # Get existing MIDI files
        available_stems = ["vocals", "bass", "guitar", "piano", "other"]
        midi_files = {}
        for stem in available_stems:
            midi_path = midi_dir / f"{stem}_basic_pitch.mid"
            if midi_path.exists():
                midi_files[stem] = midi_path
        
        if not midi_files:
            return jsonify({
                'success': False,
                'error': 'No MIDI files found for this song'
            })
        
        # Filter for single-stem mode if requested
        print(f"[UI] Arrangement mode: {arrangement_mode}, Single stem: {single_stem}")
        print(f"[UI] Available MIDI files: {list(midi_files.keys())}")
        
        if arrangement_mode == 'single':
            if not single_stem:
                error_msg = 'Single-stem mode requires "stem" parameter'
                print(f"[UI] Error: {error_msg}")
                return jsonify({'success': False, 'error': error_msg}), 400
            if single_stem not in midi_files:
                error_msg = f'Stem "{single_stem}" not available. Available stems: {list(midi_files.keys())}'
                print(f"[UI] Error: {error_msg}")
                return jsonify({'success': False, 'error': error_msg}), 400
            midi_files = {single_stem: midi_files[single_stem]}
            print(f"[UI] Using single stem: {single_stem}")

        # Attach context file path for this song if present
        for ext in ('.mid', '.midi', '.musicxml', '.xml'):
            p = midi_dir / f"context{ext}"
            if p.exists():
                params['__contextPath'] = str(p)
                break

        # Add enhanced parameters if available
        if ENHANCED_MODULES_AVAILABLE:
            params['enableEnhancedBPM'] = params.get('enableEnhancedBPM', True)
            params['bpmConfidenceThreshold'] = params.get('bpmConfidenceThreshold', 0.7)
            params['enhancedNoteProcessing'] = params.get('enhancedNoteProcessing', True)
            params['noteConfidenceThreshold'] = params.get('noteConfidenceThreshold', 0.3)
        
        # Generate the arrangement using UI functions
        try:
            print(f"[UI] Starting arrangement generation...")
            result = create_intelligent_piano_arrangement_with_ui_params(midi_files, output_midi, params)
            print(f"[UI] Arrangement generation completed successfully")
            
            return jsonify({
                'success': True,
                'output': result['output'],
                'noteCount': result['noteCount'],
                'phrases': result['phrases'],
                'filePath': str(output_midi),
                'downloadUrl': f"/files/{output_midi}",
                'bpm': result.get('bpm'),
                'events': result.get('events', []),
                'mode': arrangement_mode,
                'stem': single_stem if arrangement_mode == 'single' else None,
                'hasContext': bool(params.get('__contextPath'))
            })
            
        except Exception as arrangement_error:
            error_msg = f"Arrangement generation failed: {str(arrangement_error)}"
            print(f"[UI] Error: {error_msg}")
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500
            
    except Exception as e:
        print(f"[UI] Error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/songs', methods=['GET'])
def list_songs():
    """List available songs with their available stems"""
    root = Path('outputs/midi')
    songs = []
    stem_types = ["vocals", "bass", "guitar", "piano", "other", "drums"]
    
    if root.exists():
        for song_dir in root.iterdir():
            if song_dir.is_dir():
                available_stems = []
                for stem in stem_types:
                    midi_file = song_dir / f"{stem}_basic_pitch.mid"
                    if midi_file.exists():
                        available_stems.append(stem)
                
                songs.append({
                    "name": song_dir.name,
                    "stems": available_stems
                })
    
    return jsonify({'songs': sorted(songs, key=lambda x: x["name"])})

@app.route('/upload', methods=['POST'])
def upload_song():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded (field name should be "file")'}), 400
        f = request.files['file']
        if f.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400

        # Sanitize filename
        filename = werkzeug.utils.secure_filename(f.filename)
        upload_dir = Path('uploads')
        upload_dir.mkdir(parents=True, exist_ok=True)
        save_path = upload_dir / filename
        f.save(str(save_path))

        # Derive song name from filename stem
        song_name = save_path.stem
        out_root = Path('outputs')

        # Run separation (Demucs) and transcription (Basic Pitch)
        sep_dir = run_demucs(save_path, out_root, model='htdemucs', device='cpu')

        expected = {
            "vocals": ["vocals.mp3", "vocals.wav"],
            "bass": ["bass.mp3", "bass.wav"],
            "other": ["other.mp3", "other.wav"],
            "drums": ["drums.mp3", "drums.wav"],
            "guitar": ["guitar.mp3", "guitar.wav"],
            "piano": ["piano.mp3", "piano.wav"]
        }
        midi_out_dir = out_root / 'midi' / song_name
        midi_out_dir.mkdir(parents=True, exist_ok=True)

        any_midi = False
        for stem_name, candidates in expected.items():
            wav = None
            for c in candidates:
                p = sep_dir / c
                if p.exists():
                    wav = p
                    break
            if not wav:
                continue
            transcribe_with_basic_pitch(wav, midi_out_dir, stem_name)
            any_midi = True

        if not any_midi:
            return jsonify({'success': False, 'error': 'No stems were transcribed to MIDI'}), 500

        return jsonify({'success': True, 'song': song_name})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/files/<path:subpath>', methods=['GET'])
def serve_file(subpath: str):
    # Restrict to outputs directory
    base = Path.cwd()
    target = (base / subpath).resolve()
    outputs = (base / 'outputs').resolve()
    if not str(target).startswith(str(outputs)):
        return jsonify({'success': False, 'error': 'Access denied'}), 403
    if not target.exists():
        return jsonify({'success': False, 'error': 'File not found'}), 404
    return send_file(str(target), as_attachment=True)

@app.route('/upload-context', methods=['POST'])
def upload_context():
    try:
        song = request.form.get('song')
        if not song:
            return jsonify({'success': False, 'error': 'Missing song parameter'}), 400
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded (field name should be "file")'}), 400
        f = request.files['file']
        if f.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'}), 400

        filename = werkzeug.utils.secure_filename(f.filename)
        dest_dir = Path('outputs/midi')/song
        dest_dir.mkdir(parents=True, exist_ok=True)
        # Save as canonical context.* keeping extension
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ('.mid', '.midi', '.musicxml', '.xml'):
            return jsonify({'success': False, 'error': 'Unsupported context format'}), 400
        save_path = dest_dir / f"context{ext}"
        # Remove existing context files
        for other_ext in ('.mid', '.midi', '.musicxml', '.xml'):
            p = dest_dir / f"context{other_ext}"
            if p.exists():
                p.unlink()
        f.save(str(save_path))
        return jsonify({'success': True, 'path': str(save_path)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/test-enhancements', methods=['POST'])
def test_enhancements():
    """Test enhanced BPM detection and note processing"""
    if not ENHANCED_MODULES_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'Enhanced modules not available'
        })
    
    try:
        data = request.json
        song_name = data.get('song')
        
        if not song_name:
            return jsonify({
                'success': False,
                'error': 'Song name required'
            })
        
        # Set up paths
        midi_dir = Path(f"outputs/midi/{song_name}")
        audio_files = []
        
        # Look for original audio file
        data_dirs = [Path("../../data"), Path("../data"), Path("data")]
        for data_dir in data_dirs:
            if data_dir.exists():
                for ext in ['.mp3', '.wav', '.flac']:
                    audio_file = data_dir / f"{song_name}{ext}"
                    if audio_file.exists():
                        audio_files.append(audio_file)
        
        results = {}
        
        # Test BPM detection if audio file found
        if audio_files and ENHANCED_MODULES_AVAILABLE:
            try:
                bpm_result = enhanced_bpm_detection(audio_files[0])
                results['bpm'] = bpm_result
                print(f"[UI Test] BPM Detection: {bpm_result['bpm']:.1f} (confidence: {bpm_result['confidence']:.2f})")
            except Exception as e:
                print(f"[UI Test] BPM detection error: {e}")
                results['bpm'] = {'error': str(e)}
        
        # Test MIDI analysis if MIDI files exist
        if midi_dir.exists():
            try:
                midi_files = list(midi_dir.glob("*_basic_pitch.mid"))
                if midi_files:
                    # Analyze first MIDI file for note statistics
                    midi_result = analyze_midi_file_for_ui(midi_files[0])
                    results['notes'] = midi_result
                    
                    # Test arrangement quality
                    arrangement_files = list(midi_dir.glob("*intelligent*.mid"))
                    if arrangement_files:
                        arrangement_result = analyze_arrangement_for_ui(arrangement_files[0])
                        results['arrangement'] = arrangement_result
                        
            except Exception as e:
                print(f"[UI Test] MIDI analysis error: {e}")
                results['notes'] = {'error': str(e)}
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        print(f"[UI Test] Error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

def analyze_midi_file_for_ui(midi_path: Path):
    """Analyze MIDI file and return UI-friendly results"""
    try:
        import mido
        from main import extract_note_events
        
        midi_file = mido.MidiFile(str(midi_path))
        original_events = extract_note_events(midi_file)
        
        # Test enhanced note processing
        if ENHANCED_MODULES_AVAILABLE:
            enhanced_events = enhance_note_events_with_harmonic_analysis(original_events)
            
            return {
                'original_count': len(original_events),
                'enhanced_count': len(enhanced_events),
                'improvement_ratio': len(enhanced_events) / max(1, len(original_events))
            }
        else:
            return {
                'original_count': len(original_events),
                'enhanced_count': len(original_events),
                'improvement_ratio': 1.0
            }
            
    except Exception as e:
        return {'error': str(e)}

def analyze_arrangement_for_ui(midi_path: Path):
    """Analyze arrangement quality for UI display"""
    try:
        import mido
        import numpy as np
        
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
                            'duration': duration
                        })
                        del active_notes[msg.note]
        
        if not notes:
            return {'error': 'No notes found'}
        
        # Calculate metrics
        total_duration = max(n['start'] + n['duration'] for n in notes)
        note_density = len(notes) / (total_duration / 1000)
        
        # Polyphony analysis
        max_polyphony = 0
        time_points = set(n['start'] for n in notes)
        time_points.update(n['start'] + n['duration'] for n in notes)
        
        for time_point in sorted(time_points):
            active_count = sum(1 for n in notes 
                             if n['start'] <= time_point < n['start'] + n['duration'])
            max_polyphony = max(max_polyphony, active_count)
        
        # Playability score
        playability_score = 10.0
        if max_polyphony > 10:
            playability_score -= (max_polyphony - 10) * 0.5
        if note_density > 8:
            playability_score -= (note_density - 8) * 0.5
        playability_score = max(0, playability_score)
        
        return {
            'note_count': len(notes),
            'duration_seconds': total_duration / 1000,
            'note_density': note_density,
            'max_polyphony': max_polyphony,
            'playability_score': playability_score
        }
        
    except Exception as e:
        return {'error': str(e)}

# Functions moved to ui_functions.py to keep this file clean

if __name__ == '__main__':
    print("🎹 Piano Arrangement UI Server")
    print("📂 Serving from:", Path.cwd())
    print("🌐 Open your browser to: http://localhost:5000")
    print("⚠️  Make sure you have Flask installed: pip install flask")
    print("")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
