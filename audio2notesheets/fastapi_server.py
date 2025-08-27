#!/usr/bin/env python3
"""
FastAPI backend for the intelligent piano arrangement system
Provides RESTful API endpoints that can be easily tested with curl
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import sys
import traceback
import tempfile
import shutil
from pathlib import Path
import werkzeug.utils

# Add current directory to path
sys.path.append('.')

# Import our modules
from ui_functions import create_intelligent_piano_arrangement_with_ui_params
from main import run_demucs, transcribe_with_basic_pitch

# Import enhanced transcription modules
try:
    from transcribe_to_piano import transcribe_with_basic_pitch as enhanced_basic_pitch_transcribe
    from piano_transcription_inference import PianoTranscription, sample_rate
    import librosa
    ENHANCED_TRANSCRIPTION_AVAILABLE = True
    print("[FastAPI] Enhanced transcription modules loaded successfully")
except ImportError as e:
    print(f"[FastAPI] Enhanced transcription modules not available: {e}")
    ENHANCED_TRANSCRIPTION_AVAILABLE = False

# Import enhanced modules
try:
    from enhanced_bpm_detection import enhanced_bpm_detection, enhanced_midi_bpm_detection
    from enhanced_note_extraction import enhance_note_events_with_harmonic_analysis
    ENHANCED_MODULES_AVAILABLE = True
    print("[FastAPI] Enhanced 2025 modules loaded successfully")
except ImportError as e:
    print(f"[FastAPI] Enhanced modules not available: {e}")
    ENHANCED_MODULES_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="Intelligent Piano Arrangement API",
    description="Advanced MP3-to-piano conversion with AI and enhanced musical intelligence",
    version="2025.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class ArrangementParams(BaseModel):
    song: str
    arrangementMode: str = "combined"
    stem: Optional[str] = None
    
    # Phrase detection
    phraseDuration: int = 12000
    phraseOverlap: int = 2000
    
    # Note density
    minNotesPerSec: float = 4.0
    maxNotesPerSec: float = 8.0
    
    # Selection ratios
    veryDenseRatio: float = 0.25
    denseRatio: float = 0.35
    mediumRatio: float = 0.50
    sparseRatio: float = 0.70
    
    # Duration filtering
    veryShortPenalty: int = 20
    shortPenalty: int = 10
    minDuration: int = 150
    goodDuration: int = 500
    
    # Legato and smoothness
    legatoEnabled: bool = True
    legatoGapThreshold: int = 300
    minNoteDuration: int = 200
    quantizeGrid: str = "1/16"
    mergeGapMs: int = 80
    sustainPedal: bool = True
    pedalWindowMs: Optional[int] = None
    sustainedHarmony: bool = True
    
    # Part weights
    vocalsWeight: float = 1.0
    pianoWeight: float = 0.8
    guitarWeight: float = 0.6
    bassWeight: float = 0.7
    
    # Enhanced 2025 parameters
    enableEnhancedBPM: bool = True
    bpmConfidenceThreshold: float = 0.7
    enhancedNoteProcessing: bool = True
    noteConfidenceThreshold: float = 0.3
    
    # AI Arranger parameters
    aiEnabled: bool = False
    aiModel: str = "music_transformer"
    aiPrimerSource: str = "context"
    aiBars: int = 8
    aiTemperature: float = 1.0
    aiBeamSize: int = 1
    aiBlendMode: str = "replace_intro"
    useGPU: bool = True
    aiCheckpoint: Optional[str] = None
    
    # Context parameters
    useContextTempo: bool = True
    useContextHarmony: bool = True
    useContextIntro: bool = False
    contextIntroSeconds: int = 8
    
    # Enhanced Transcription parameters (2025)
    transcriptionMethod: str = "current"  # "current", "enhanced_basic_pitch", "piano_inference"
    enhancedQuantization: bool = True
    leftRightSplit: bool = False
    pianoOptimized: bool = True

class ArrangementResponse(BaseModel):
    success: bool
    output: Optional[str] = None
    noteCount: Optional[int] = None
    phrases: Optional[int] = None
    bpm: Optional[float] = None
    events: Optional[List[Dict[str, Any]]] = None
    filePath: Optional[str] = None
    downloadUrl: Optional[str] = None
    mode: Optional[str] = None
    stem: Optional[str] = None
    hasContext: Optional[bool] = None
    error: Optional[str] = None

class SongInfo(BaseModel):
    name: str
    stems: List[str]

class SongsResponse(BaseModel):
    songs: List[SongInfo]

class TestRequest(BaseModel):
    song: str

class TestResponse(BaseModel):
    success: bool
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

async def handle_enhanced_transcription(song_name: str, method: str, params: ArrangementParams, existing_midi_files: dict):
    """Handle enhanced transcription methods"""
    try:
        # Find the original audio file
        data_dirs = [Path("../../data"), Path("../data"), Path("data")]
        audio_file = None
        
        for data_dir in data_dirs:
            if data_dir.exists():
                for ext in ['.mp3', '.wav', '.flac']:
                    candidate = data_dir / f"{song_name}{ext}"
                    if candidate.exists():
                        audio_file = candidate
                        break
                if audio_file:
                    break
        
        if not audio_file:
            print(f"[Enhanced] Audio file not found for {song_name}, using existing MIDI files")
            return existing_midi_files
        
        print(f"[Enhanced] Found audio file: {audio_file}")
        
        # Create enhanced output directory
        enhanced_dir = Path(f"outputs/enhanced/{song_name}")
        enhanced_dir.mkdir(parents=True, exist_ok=True)
        
        new_midi_files = existing_midi_files.copy()
        
        if method == "enhanced_basic_pitch":
            print("[Enhanced] Running Enhanced Basic Pitch transcription...")
            
            piano_midi, split_midi = enhanced_basic_pitch_transcribe(
                str(audio_file),
                str(enhanced_dir),
                make_lr_split=params.leftRightSplit,
                quantize_grid=16 if params.enhancedQuantization else None
            )
            
            # Replace the combined MIDI with enhanced version
            if piano_midi:
                new_midi_files['enhanced_piano'] = Path(piano_midi)
                print(f"[Enhanced] Created enhanced piano MIDI: {piano_midi}")
            
            if split_midi and params.leftRightSplit:
                new_midi_files['left_right_split'] = Path(split_midi)
                print(f"[Enhanced] Created L/R split MIDI: {split_midi}")
        
        elif method == "piano_inference":
            print("[Enhanced] Running Piano Transcription Inference...")
            
            # Use the 'other' stem if available, otherwise full audio
            if 'other' in existing_midi_files:
                # Find the other stem audio
                stem_audio = None
                stem_dirs = [Path(f"outputs/htdemucs/{song_name}"), 
                           Path(f"outputs/htdemucs_6s/{song_name}")]
                
                for stem_dir in stem_dirs:
                    other_file = stem_dir / "other.mp3"
                    if other_file.exists():
                        stem_audio = other_file
                        break
                
                if stem_audio:
                    print(f"[Enhanced] Using piano stem: {stem_audio}")
                    audio_to_use = stem_audio
                else:
                    print("[Enhanced] Piano stem not found, using full audio")
                    audio_to_use = audio_file
            else:
                audio_to_use = audio_file
            
            # Load audio and transcribe
            audio, _ = librosa.load(str(audio_to_use), sr=sample_rate, mono=True)
            transcriptor = PianoTranscription(device='cpu')
            
            piano_midi_path = enhanced_dir / f"{song_name}_piano_inference.mid"
            transcriptor.transcribe(audio, str(piano_midi_path))
            
            new_midi_files['piano_inference'] = piano_midi_path
            print(f"[Enhanced] Created piano inference MIDI: {piano_midi_path}")
        
        return new_midi_files
        
    except Exception as e:
        print(f"[Enhanced] Error in enhanced transcription: {e}")
        traceback.print_exc()
        return existing_midi_files

# Serve the HTML frontend
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML interface"""
    try:
        html_path = Path("piano_ui.html")
        if html_path.exists():
            return FileResponse(str(html_path))
        else:
            raise HTTPException(status_code=404, detail=f"Frontend HTML file not found: {html_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving frontend: {str(e)}")

# API Endpoints

@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "enhanced_modules": ENHANCED_MODULES_AVAILABLE,
        "version": "2025.1.0"
    }

@app.get("/api/songs", response_model=SongsResponse)
async def list_songs():
    """List available songs with their stems"""
    try:
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
                    
                    if available_stems:  # Only include songs with stems
                        songs.append(SongInfo(
                            name=song_dir.name,
                            stems=available_stems
                        ))
        
        return SongsResponse(songs=sorted(songs, key=lambda x: x.name))
    
    except Exception as e:
        print(f"[FastAPI] Error listing songs: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to list songs: {str(e)}")

@app.post("/api/generate", response_model=ArrangementResponse)
async def generate_arrangement(params: ArrangementParams):
    """Generate piano arrangement with given parameters"""
    try:
        print(f"[FastAPI] Generating arrangement for: {params.song}")
        print(f"[FastAPI] Mode: {params.arrangementMode}, Stem: {params.stem}")
        
        # Set up paths
        midi_dir = Path(f"outputs/midi/{params.song}")
        if not midi_dir.exists():
            raise HTTPException(
                status_code=404, 
                detail=f'Song directory not found: {midi_dir}'
            )
        
        # Create output file
        params_hash = hash(params.json()) % 10000
        output_midi = midi_dir / f"intelligent_piano_api_{params_hash}.mid"
        
        # Get existing MIDI files
        available_stems = ["vocals", "bass", "guitar", "piano", "other", "drums"]
        midi_files = {}
        for stem in available_stems:
            midi_path = midi_dir / f"{stem}_basic_pitch.mid"
            if midi_path.exists():
                midi_files[stem] = midi_path
        
        if not midi_files:
            raise HTTPException(
                status_code=400, 
                detail='No MIDI files found for this song'
            )
        
        # Filter for single-stem mode
        if params.arrangementMode == 'single':
            if not params.stem:
                raise HTTPException(
                    status_code=400, 
                    detail='Single-stem mode requires "stem" parameter'
                )
            if params.stem not in midi_files:
                raise HTTPException(
                    status_code=400,
                    detail=f'Stem "{params.stem}" not available. Available: {list(midi_files.keys())}'
                )
            midi_files = {params.stem: midi_files[params.stem]}
        
        # Check for context file
        context_path = None
        for ext in ('.mid', '.midi', '.musicxml', '.xml'):
            p = midi_dir / f"context{ext}"
            if p.exists():
                context_path = str(p)
                break
        
        # Convert params to dict and add context path
        params_dict = params.dict()
        if context_path:
            params_dict['__contextPath'] = context_path
        
        # Add enhanced parameters if available
        if ENHANCED_MODULES_AVAILABLE:
            params_dict['enableEnhancedBPM'] = params.enableEnhancedBPM
            params_dict['bpmConfidenceThreshold'] = params.bpmConfidenceThreshold
            params_dict['enhancedNoteProcessing'] = params.enhancedNoteProcessing
            params_dict['noteConfidenceThreshold'] = params.noteConfidenceThreshold
        
        # Handle enhanced transcription methods
        if ENHANCED_TRANSCRIPTION_AVAILABLE and params.transcriptionMethod != "current":
            print(f"[FastAPI] Using enhanced transcription method: {params.transcriptionMethod}")
            midi_files = await handle_enhanced_transcription(
                params.song, params.transcriptionMethod, params, midi_files
            )
        
        # Generate arrangement
        print("[FastAPI] Starting arrangement generation...")
        result = create_intelligent_piano_arrangement_with_ui_params(
            midi_files, output_midi, params_dict
        )
        print("[FastAPI] Arrangement generation completed")
        
        return ArrangementResponse(
            success=True,
            output=result['output'],
            noteCount=result['noteCount'],
            phrases=result['phrases'],
            bpm=result.get('bpm'),
            events=result.get('events', []),
            filePath=str(output_midi),
            downloadUrl=f"/api/files/{output_midi}",
            mode=params.arrangementMode,
            stem=params.stem if params.arrangementMode == 'single' else None,
            hasContext=bool(context_path)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[FastAPI] Error generating arrangement: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Arrangement generation failed: {str(e)}"
        )

@app.post("/api/upload")
async def upload_song(file: UploadFile = File(...)):
    """Upload and process new audio file"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Sanitize filename
        filename = werkzeug.utils.secure_filename(file.filename)
        upload_dir = Path('uploads')
        upload_dir.mkdir(parents=True, exist_ok=True)
        save_path = upload_dir / filename
        
        # Save uploaded file
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Derive song name
        song_name = save_path.stem
        out_root = Path('outputs')
        
        print(f"[FastAPI] Processing uploaded file: {filename} -> {song_name}")
        
        # Run separation and transcription
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
        
        transcribed_stems = []
        for stem_name, candidates in expected.items():
            wav = None
            for c in candidates:
                p = sep_dir / c
                if p.exists():
                    wav = p
                    break
            if wav:
                transcribe_with_basic_pitch(wav, midi_out_dir, stem_name)
                transcribed_stems.append(stem_name)
        
        if not transcribed_stems:
            raise HTTPException(
                status_code=500, 
                detail='No stems were transcribed to MIDI'
            )
        
        print(f"[FastAPI] Upload successful: {song_name}, stems: {transcribed_stems}")
        return {
            "success": True, 
            "song": song_name,
            "stems": transcribed_stems
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[FastAPI] Upload error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/upload-context")
async def upload_context(song: str = Form(...), file: UploadFile = File(...)):
    """Upload context file for a song"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        filename = werkzeug.utils.secure_filename(file.filename)
        dest_dir = Path('outputs/midi') / song
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as canonical context file
        ext = Path(filename).suffix.lower()
        if ext not in ('.mid', '.midi', '.musicxml', '.xml'):
            raise HTTPException(
                status_code=400, 
                detail='Unsupported context format'
            )
        
        save_path = dest_dir / f"context{ext}"
        
        # Remove existing context files
        for other_ext in ('.mid', '.midi', '.musicxml', '.xml'):
            p = dest_dir / f"context{other_ext}"
            if p.exists():
                p.unlink()
        
        # Save new context file
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"[FastAPI] Context uploaded for {song}: {save_path}")
        return {"success": True, "path": str(save_path)}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[FastAPI] Context upload error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Context upload failed: {str(e)}")

@app.post("/api/test-enhancements", response_model=TestResponse)
async def test_enhancements(request: TestRequest):
    """Test enhanced BPM detection and note processing"""
    if not ENHANCED_MODULES_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail='Enhanced modules not available'
        )
    
    try:
        song_name = request.song
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
        
        # Test BPM detection
        if audio_files:
            try:
                bpm_result = enhanced_bpm_detection(audio_files[0])
                results['bpm'] = bpm_result
                print(f"[FastAPI Test] BPM: {bpm_result['bpm']:.1f} (confidence: {bpm_result['confidence']:.2f})")
            except Exception as e:
                results['bpm'] = {'error': str(e)}
        
        # Test MIDI analysis
        if midi_dir.exists():
            try:
                midi_files = list(midi_dir.glob("*_basic_pitch.mid"))
                if midi_files:
                    midi_result = analyze_midi_file_for_test(midi_files[0])
                    results['notes'] = midi_result
                    
                    arrangement_files = list(midi_dir.glob("*intelligent*.mid"))
                    if arrangement_files:
                        arrangement_result = analyze_arrangement_for_test(arrangement_files[0])
                        results['arrangement'] = arrangement_result
            except Exception as e:
                results['notes'] = {'error': str(e)}
        
        return TestResponse(success=True, results=results)
        
    except Exception as e:
        print(f"[FastAPI Test] Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")

@app.get("/api/files/{file_path:path}")
async def serve_file(file_path: str):
    """Serve files from outputs directory"""
    try:
        base = Path.cwd()
        target = (base / file_path).resolve()
        outputs = (base / 'outputs').resolve()
        
        # Security check
        if not str(target).startswith(str(outputs)):
            raise HTTPException(status_code=403, detail='Access denied')
        
        if not target.exists():
            raise HTTPException(status_code=404, detail='File not found')
        
        return FileResponse(str(target))
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[FastAPI] File serve error: {e}")
        raise HTTPException(status_code=500, detail=f"File serve failed: {str(e)}")

# Helper functions for testing
def analyze_midi_file_for_test(midi_path: Path):
    """Analyze MIDI file for testing"""
    try:
        import mido
        from main import extract_note_events
        
        midi_file = mido.MidiFile(str(midi_path))
        original_events = extract_note_events(midi_file)
        
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

def analyze_arrangement_for_test(midi_path: Path):
    """Analyze arrangement quality for testing"""
    try:
        import mido
        
        midi_file = mido.MidiFile(str(midi_path))
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
        
        total_duration = max(n['start'] + n['duration'] for n in notes)
        note_density = len(notes) / (total_duration / 1000)
        
        # Calculate polyphony
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

if __name__ == "__main__":
    import uvicorn
    print("🎹 FastAPI Piano Arrangement Server")
    print("📂 Serving from:", Path.cwd())
    print("🌐 API docs: http://localhost:8000/docs")
    print("🎵 Frontend: http://localhost:8000")
    print("📡 All endpoints available for curl testing")
    print("")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)