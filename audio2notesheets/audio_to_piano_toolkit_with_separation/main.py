#!/usr/bin/env python3
"""
Main separation + transcription utilities.

Exports:
- run_demucs(input_audio: Path, out_root: Path, model: str, device: str) -> Path
- run_spleeter(input_audio: Path, out_root: Path, stems: int = 5) -> Path
- transcribe_with_basic_pitch(stem_path: Path, out_dir: Path, stem_name: str = "")
    -> tuple[Path, Path | None, Path | None]
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pretty_midi as pm
import librosa

# ---------- constants (from your old pipeline) ----------
PIANO_MIN, PIANO_MAX = 21, 108   # A0..C8
DEFAULT_GRID = 4                 # 4 = sixteenths; use 2 for eighths


# ----------------- helpers -----------------
def _resolve(p: Path) -> Path:
    return Path(p).expanduser().resolve()


def _check_typing_extensions():
    """Demucs (via torch) on recent versions requires typing-extensions providing TypeIs.
    If Spleeter pinned typing-extensions to 4.5.0, torch import will fail.
    """
    try:
        import typing_extensions as te  # type: ignore
        _ = te.TypeIs  # raises AttributeError on too-old versions
    except Exception:
        raise RuntimeError(
            "typing-extensions is too old for torch/demucs.\n"
            "Upgrade in THIS venv: pip install -U typing-extensions==4.12.2"
        )


# --- Demucs separation ---
def run_demucs(input_audio: Path, out_root: Path,
               model: str = "htdemucs", device: str = "cpu") -> Path:
    """Separate a track using Demucs via its Python entrypoint."""
    _check_typing_extensions()
    try:
        import demucs.separate as demucs_separate  # imports torch and friends
    except Exception as e:
        raise RuntimeError(
            "Failed to import Demucs (torch/typing-extensions mismatch likely).\n"
            "Try: pip install -U demucs torch typing-extensions==4.12.2\n"
            f"Original import error: {e!r}"
        )

    input_audio = _resolve(input_audio)
    out_root = _resolve(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    args = ["--mp3", "-d", device, "-n", model, "-o", str(out_root), str(input_audio)]
    print(f"[demucs] demucs.separate.main({args})")
    demucs_separate.main(args)

    # Demucs writes to out_root / model / track_stem
    sep_dir = out_root / model / input_audio.stem
    if not sep_dir.exists():
        # Fallback search
        candidates = list(out_root.rglob(input_audio.stem))
        if not candidates:
            raise FileNotFoundError(f"Demucs output not found in {out_root} for {input_audio.stem}")
        sep_dir = candidates[0]
    print(f"[demucs] Separated stems at: {sep_dir}")
    return sep_dir


# --- Spleeter separation ---
def run_spleeter(input_audio: Path, out_root: Path, stems: int = 5) -> Path:
    """Separate a track using Spleeter (2/4/5 stems). Returns the directory with WAVs.
    Note: Spleeter pins TensorFlow 2.12.x and old typing-extensions; use a dedicated venv.
    """
    try:
        import spleeter.separator  # type: ignore
        from spleeter.audio.adapter import AudioAdapter  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Spleeter import failed. This is usually due to TensorFlow/TFLite collisions or a "
            "downgraded typing-extensions in the same venv.\n"
            "Either run Spleeter in its own venv (recommended) OR install pins compatible with "
            "Spleeter (typing-extensions<4.6, tensorflow==2.12.1).\n"
            f"Original import error: {e!r}"
        )

    input_audio = _resolve(input_audio)
    out_root = _resolve(out_root)
    sep_dir = out_root / "spleeter" / input_audio.stem
    sep_dir.mkdir(parents=True, exist_ok=True)

    print(f"[spleeter] Separating {input_audio.name} with {stems}-stems...")
    separator = spleeter.separator.Separator(f"spleeter:{stems}stems")
    audio_adapter = AudioAdapter.default()
    sample_rate = 44100
    waveform, _ = audio_adapter.load(str(input_audio), sample_rate=sample_rate)
    prediction = separator.separate(waveform)

    for inst, audio in prediction.items():
        out_wav = sep_dir / f"{inst}.wav"
        audio_adapter.save(str(out_wav), audio, sample_rate)
        print(f"[spleeter] Saved {out_wav}")

    print(f"[spleeter] Stems at: {sep_dir}")
    return sep_dir


# --- Post-processing utilities (from your old script) ---
def load_audio_for_beats(audio_path: Path):
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='frames')
    beat_times = librosa.frames_to_time(beats, sr=sr)
    return float(tempo), beat_times


def make_grid(beat_times, grid_per_beat: int = DEFAULT_GRID, tail_sec: float = 2.0):
    if len(beat_times) < 2:
        # fallback grid (0.5s steps up to 10 min)
        return np.arange(0, 600, 0.5)
    grid = []
    for i in range(len(beat_times) - 1):
        t0, t1 = beat_times[i], beat_times[i + 1]
        step = (t1 - t0) / grid_per_beat
        grid.extend([t0 + k * step for k in range(grid_per_beat)])
    grid.append(beat_times[-1])
    grid = np.array(grid, dtype=float)
    # allow tails to snap cleanly
    step = grid[1] - grid[0] if len(grid) > 1 else 0.25
    grid = np.concatenate([grid, [grid[-1] + step * 2, grid[-1] + tail_sec]])
    return grid


def _snap(x, grid):
    idx = np.searchsorted(grid, x)
    if idx <= 0:
        return grid[0]
    if idx >= len(grid):
        return grid[-1]
    return grid[idx] if abs(grid[idx] - x) < abs(grid[idx - 1] - x) else grid[idx - 1]


def quantize_pm(pmidi: pm.PrettyMIDI, grid):
    for inst in pmidi.instruments:
        for n in inst.notes:
            n.start = _snap(n.start, grid)
            n.end = max(_snap(n.end, grid), n.start + 0.03)


def clamp_and_clean(notes, min_ms=60):
    out = []
    for n in notes:
        if n.pitch < PIANO_MIN or n.pitch > PIANO_MAX:
            continue
        if (n.end - n.start) * 1000 < min_ms:
            continue
        out.append(n)
    return out


def remove_pitch_bends(pmidi):
    for inst in pmidi.instruments:
        inst.pitch_bends = []


def to_piano_program(pmidi):
    for inst in pmidi.instruments:
        inst.program = 0
        inst.is_drum = False


def group_by_grid(notes, grid):
    buckets = {}
    for n in notes:
        gi = int(np.searchsorted(grid, n.start) - 1)
        gi = max(0, min(gi, len(grid) - 1))
        buckets.setdefault(gi, []).append(n)
    return buckets


def dedupe_and_merge(notes, join_gap=0.01):
    """Merge immediately repeated notes of same pitch with tiny gaps (post-quantize)."""
    by_pitch = {}
    for n in sorted(notes, key=lambda x: (x.pitch, x.start)):
        key = n.pitch
        by_pitch.setdefault(key, [])
        if by_pitch[key] and n.start - by_pitch[key][-1].end <= join_gap:
            by_pitch[key][-1].end = max(by_pitch[key][-1].end, n.end)
        else:
            by_pitch[key].append(pm.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end))
    merged = []
    for arr in by_pitch.values():
        merged.extend(arr)
    return merged


def piano_voice_split(pmidi, grid, max_rh_voices=2, min_ms=60):
    # gather all notes
    pool = []
    for inst in pmidi.instruments:
        pool.extend(inst.notes)
    pool = clamp_and_clean(pool, min_ms=min_ms)
    buckets = group_by_grid(pool, grid)

    LH = pm.Instrument(program=0, name="Left Hand")
    RH = pm.Instrument(program=0, name="Right Hand")

    for gi, notes in sorted(buckets.items()):
        if not notes:
            continue
        notes_sorted = sorted(notes, key=lambda n: n.pitch)
        bass = notes_sorted[0]
        # Extend LH to next grid line for sustain feel
        next_edge = grid[min(gi + 1, len(grid) - 1)]
        LH.notes.append(pm.Note(velocity=min(100, bass.velocity + 5),
                                pitch=bass.pitch, start=bass.start, end=max(bass.end, next_edge)))
        # RH: keep up to N highest notes (avoids dense clusters)
        ups = sorted(notes_sorted[1:], key=lambda n: n.pitch, reverse=True)[:max_rh_voices]
        for u in ups:
            RH.notes.append(pm.Note(velocity=u.velocity, pitch=u.pitch, start=u.start, end=u.end))

    # Clean tiny duplicates after split
    LH.notes = dedupe_and_merge(LH.notes)
    RH.notes = dedupe_and_merge(RH.notes)

    out = pm.PrettyMIDI(initial_tempo=pmidi.estimate_tempo() or 120.0)
    # Drop an empty hand if the source is monophonic
    hands = [h for h in (LH, RH) if h.notes]
    out.instruments = hands if hands else [pm.Instrument(program=0, name="Piano")]
    return out


# --- Enhanced vocal processing functions ---
def process_vocals_advanced(pmidi: pm.PrettyMIDI, mode: str = "clean") -> pm.PrettyMIDI:
    """Advanced vocal processing with multiple modes for different vocal styles."""
    if mode == "clean":
        return _clean_vocal_mode(pmidi)
    elif mode == "harmony":
        return _harmony_vocal_mode(pmidi)
    elif mode == "lead":
        return _lead_vocal_mode(pmidi)
    elif mode == "original":
        return pmidi  # Keep original with pitch bends
    else:
        return _clean_vocal_mode(pmidi)  # Default to clean

def _clean_vocal_mode(pmidi: pm.PrettyMIDI) -> pm.PrettyMIDI:
    """Clean vocal mode: removes pitch bends, quantizes heavily, focuses on main melody."""
    # Remove pitch bends first
    remove_pitch_bends(pmidi)
    
    # More aggressive note filtering for vocals
    for inst in pmidi.instruments:
        # Filter out very short notes (vocals need sustain)
        inst.notes = [n for n in inst.notes if (n.end - n.start) * 1000 >= 100]
        
        # Remove notes outside comfortable vocal range (C3-C6)
        inst.notes = [n for n in inst.notes if 48 <= n.pitch <= 84]
        
        # Reduce polyphony - keep strongest notes per time window
        inst.notes = _reduce_vocal_polyphony(inst.notes)
    
    return pmidi

def _harmony_vocal_mode(pmidi: pm.PrettyMIDI) -> pm.PrettyMIDI:
    """Harmony mode: preserves multiple voices, better for complex arrangements."""
    remove_pitch_bends(pmidi)
    
    for inst in pmidi.instruments:
        # Less aggressive filtering to preserve harmonies
        inst.notes = [n for n in inst.notes if (n.end - n.start) * 1000 >= 80]
        
        # Wider vocal range for harmonies
        inst.notes = [n for n in inst.notes if 36 <= n.pitch <= 96]
        
        # Group notes by time and keep up to 4 voices
        inst.notes = _preserve_vocal_harmonies(inst.notes)
    
    return pmidi

def _lead_vocal_mode(pmidi: pm.PrettyMIDI) -> pm.PrettyMIDI:
    """Lead vocal mode: extracts main melody line only."""
    remove_pitch_bends(pmidi)
    
    for inst in pmidi.instruments:
        # Strong filtering for clean melody
        inst.notes = [n for n in inst.notes if (n.end - n.start) * 1000 >= 120]
        
        # Main vocal range
        inst.notes = [n for n in inst.notes if 48 <= n.pitch <= 84]
        
        # Extract only the melody line
        inst.notes = _extract_melody_line(inst.notes)
    
    return pmidi

def _reduce_vocal_polyphony(notes, max_voices=2):
    """Reduce polyphony by keeping strongest notes per time window."""
    if not notes:
        return notes
    
    # Sort by start time
    notes = sorted(notes, key=lambda n: n.start)
    
    # Group overlapping notes
    time_windows = []
    current_window = [notes[0]]
    
    for note in notes[1:]:
        # If note overlaps with any note in current window
        if any(note.start < existing.end for existing in current_window):
            current_window.append(note)
        else:
            time_windows.append(current_window)
            current_window = [note]
    
    if current_window:
        time_windows.append(current_window)
    
    # Keep only strongest notes per window
    result = []
    for window in time_windows:
        # Sort by velocity and keep top N
        window_sorted = sorted(window, key=lambda n: n.velocity, reverse=True)
        result.extend(window_sorted[:max_voices])
    
    return result

def _preserve_vocal_harmonies(notes, max_voices=4):
    """Preserve harmonies while reducing excessive polyphony."""
    if not notes:
        return notes
    
    # Group by time windows (50ms tolerance)
    time_groups = {}
    for note in notes:
        time_key = round(note.start * 20)  # 50ms buckets
        if time_key not in time_groups:
            time_groups[time_key] = []
        time_groups[time_key].append(note)
    
    result = []
    for group in time_groups.values():
        if len(group) <= max_voices:
            result.extend(group)
        else:
            # Keep most prominent voices
            group_sorted = sorted(group, key=lambda n: n.velocity, reverse=True)
            result.extend(group_sorted[:max_voices])
    
    return result

def _extract_melody_line(notes):
    """Extract the main melody line from vocal notes."""
    if not notes:
        return notes
    
    # Sort by start time
    notes = sorted(notes, key=lambda n: n.start)
    
    # Use a simple melody extraction: highest note in each time window
    time_windows = {}
    for note in notes:
        time_key = round(note.start * 8)  # 125ms buckets
        if time_key not in time_windows:
            time_windows[time_key] = []
        time_windows[time_key].append(note)
    
    melody = []
    for window in time_windows.values():
        # Take highest pitch note (melody tends to be on top)
        highest = max(window, key=lambda n: n.pitch)
        melody.append(highest)
    
    return melody

# --- Basic Pitch transcription + enhanced vocal post-processing ---
def transcribe_with_basic_pitch(stem_path: Path, out_dir: Path, stem_name: str = ""
                                ) -> tuple[Path, Path | None, Path | None]:
    """Transcribe a separated stem with Basic Pitch, then quantize + (optionally) split to hands.
    Returns (raw_basic_pitch_mid, piano_mid, piano_lr_mid).
    """
    # Lazy imports to avoid warnings when separation-only is used.
    from basic_pitch.inference import predict_and_save  # type: ignore
    from basic_pitch import ICASSP_2022_MODEL_PATH  # type: ignore

    stem_path = _resolve(stem_path)
    out_dir = _resolve(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[basic-pitch] Transcribing {stem_path.name} -> {out_dir}")
    predict_and_save(
        [str(stem_path)], str(out_dir),
        True, False, False, True,  # save_notes=True for debugging
        ICASSP_2022_MODEL_PATH,
    )

    # Find produced .mid (Basic Pitch usually names it <stem>_basic_pitch.mid)
    raw_mid = latest_child_with_suffix(out_dir, stem_path.stem, ".mid")
    print(f"[basic-pitch] Wrote {raw_mid}")

    # --- Old pipeline's post-processing ---
    try:
        _, beat_times = load_audio_for_beats(stem_path)
        grid = make_grid(beat_times, grid_per_beat=DEFAULT_GRID)

        m = pm.PrettyMIDI(str(raw_mid))
        remove_pitch_bends(m)  # helpful for guitar->piano artifacts
        quantize_pm(m, grid)
        to_piano_program(m)

        # Save merged as piano
        stem_label = stem_name or stem_path.stem
        piano_mid_path = out_dir / f"{stem_label}.piano.mid"
        m.write(str(piano_mid_path))

        # Split hands
        piano_lr = piano_voice_split(m, grid, max_rh_voices=2, min_ms=60)
        piano_lr_path = out_dir / f"{stem_label}.piano.LR.mid"
        piano_lr.write(str(piano_lr_path))

        return raw_mid, piano_mid_path, piano_lr_path
    except Exception as e:
        print(f"[basic-pitch] Post-processing failed, returning raw MIDI only: {e}")
        return raw_mid, None, None

def transcribe_vocals_enhanced(stem_path: Path, out_dir: Path, stem_name: str = ""
                              ) -> tuple[Path, Path, Path, Path, Path]:
    """Enhanced vocal transcription with multiple processing modes.
    Returns (raw_basic_pitch, clean_vocal, harmony_vocal, lead_vocal, original_with_bends).
    """
    # Lazy imports to avoid warnings when separation-only is used.
    from basic_pitch.inference import predict_and_save  # type: ignore
    from basic_pitch import ICASSP_2022_MODEL_PATH  # type: ignore

    stem_path = _resolve(stem_path)
    out_dir = _resolve(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[vocal-enhanced] Transcribing vocals {stem_path.name} -> {out_dir}")
    predict_and_save(
        [str(stem_path)], str(out_dir),
        True, False, False, True,  # save_notes=True for debugging
        ICASSP_2022_MODEL_PATH,
    )

    # Find produced .mid
    raw_mid = latest_child_with_suffix(out_dir, stem_path.stem, ".mid")
    print(f"[vocal-enhanced] Raw MIDI: {raw_mid}")

    # Generate multiple vocal processing modes
    try:
        _, beat_times = load_audio_for_beats(stem_path)
        grid = make_grid(beat_times, grid_per_beat=DEFAULT_GRID)

        # Load original MIDI
        m_original = pm.PrettyMIDI(str(raw_mid))
        
        # Create different processing modes
        stem_label = stem_name or stem_path.stem
        
        # 1. Original with pitch bends (for comparison)
        original_path = out_dir / f"{stem_label}.vocals.original.mid"
        quantize_pm(m_original, grid)  # Quantize but keep pitch bends
        to_piano_program(m_original)
        m_original.write(str(original_path))
        
        # 2. Clean vocal mode (main melody, no pitch bends)
        m_clean = pm.PrettyMIDI(str(raw_mid))
        quantize_pm(m_clean, grid)
        m_clean = process_vocals_advanced(m_clean, mode="clean")
        to_piano_program(m_clean)
        clean_path = out_dir / f"{stem_label}.vocals.clean.mid"
        m_clean.write(str(clean_path))
        
        # 3. Harmony mode (preserves multiple voices)
        m_harmony = pm.PrettyMIDI(str(raw_mid))
        quantize_pm(m_harmony, grid)
        m_harmony = process_vocals_advanced(m_harmony, mode="harmony")
        to_piano_program(m_harmony)
        harmony_path = out_dir / f"{stem_label}.vocals.harmony.mid"
        m_harmony.write(str(harmony_path))
        
        # 4. Lead vocal mode (melody line only)
        m_lead = pm.PrettyMIDI(str(raw_mid))
        quantize_pm(m_lead, grid)
        m_lead = process_vocals_advanced(m_lead, mode="lead")
        to_piano_program(m_lead)
        lead_path = out_dir / f"{stem_label}.vocals.lead.mid"
        m_lead.write(str(lead_path))

        print(f"[vocal-enhanced] Clean vocals: {clean_path}")
        print(f"[vocal-enhanced] Harmony vocals: {harmony_path}")
        print(f"[vocal-enhanced] Lead vocals: {lead_path}")
        print(f"[vocal-enhanced] Original vocals: {original_path}")

        return raw_mid, clean_path, harmony_path, lead_path, original_path
        
    except Exception as e:
        print(f"[vocal-enhanced] Enhanced processing failed, returning raw MIDI only: {e}")
        return raw_mid, raw_mid, raw_mid, raw_mid, raw_mid


def latest_child_with_suffix(folder: Path, stem_name: str, suffix: str = ".mid") -> Path:
    folder = _resolve(folder)
    # Prefer files containing the stem name
    candidates = sorted(folder.glob(f"*{stem_name}*{suffix}"))
    if not candidates:
        candidates = sorted(folder.rglob(f"*{suffix}"))
    if not candidates:
        raise FileNotFoundError(f"No {suffix} files found in {folder}")
    return candidates[-1]
