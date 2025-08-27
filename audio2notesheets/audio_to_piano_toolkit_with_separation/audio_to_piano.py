import argparse, pathlib, os
import numpy as np
import pretty_midi as pm
import librosa

# --- BASIC PITCH (polyphonic, general) ---
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH  # default model used by Basic Pitch

PIANO_MIN, PIANO_MAX = 21, 108   # A0..C8
DEFAULT_GRID = 4                 # 4 = sixteenths; use 2 for eighths

# ---------------- AMT BACKENDS ----------------

def run_basic_pitch(audio_path, out_dir):
    """General polyphonic AMT -> MIDI"""
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = pathlib.Path(audio_path)
    # Positional signature that works across BP versions; last arg is the model path
    predict_and_save(
        [str(audio_path)], str(out_dir),
        True, False, False, False,
        ICASSP_2022_MODEL_PATH
    )
    stem = audio_path.stem
    for cand in out_dir.glob(f"{stem}*.mid"):
        return cand
    raise FileNotFoundError("Basic Pitch produced no MIDI")

def run_piano_pti(audio_path, out_dir):
    """Piano-only AMT -> MIDI (pedals, onsets/frames)"""
    try:
        from piano_transcription_inference import PianoTranscription, sample_rate
    except ImportError as e:
        raise RuntimeError("piano_transcription_inference not installed. pip install piano_transcription_inference") from e
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    y, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    transcriptor = PianoTranscription(device='cpu')
    midi_path = out_dir / (pathlib.Path(audio_path).stem + ".pti.mid")
    transcriptor.transcribe(y, str(midi_path))
    return midi_path

def run_crepe_mono(audio_path, out_dir, conf_thresh=0.5, min_ms=60, pitch_dev_cents=50):
    """Monophonic AMT via CREPE f0 -> note segmentation -> MIDI"""
    try:
        import crepe
    except ImportError as e:
        raise RuntimeError("crepe not installed. pip install crepe") from e
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    time, freq, conf, _ = crepe.predict(y, sr, viterbi=True)  # 10ms hop

    # Segment f0 into notes
    midi_f0 = librosa.hz_to_midi(freq)
    voiced = conf >= conf_thresh
    notes = []
    i = 0
    hop = time[1] - time[0]
    while i < len(time):
        if not voiced[i]: i += 1; continue
        j = i + 1
        while j < len(time) and voiced[j] and abs(midi_f0[j] - midi_f0[i]) * 100 <= pitch_dev_cents:
            j += 1
        st, en = float(time[i]), float(time[j-1] + hop)
        if (en - st) * 1000 >= min_ms:
            pitch = int(round(np.median(midi_f0[i:j])))
            pitch = int(np.clip(pitch, PIANO_MIN, PIANO_MAX))
            notes.append((pitch, st, en))
        i = j

    pmidi = pm.PrettyMIDI()
    inst = pm.Instrument(program=0, name="Melody")
    for p, st, en in notes:
        inst.notes.append(pm.Note(velocity=92, pitch=p, start=st, end=en))
    pmidi.instruments.append(inst)

    midi_path = out_dir / (pathlib.Path(audio_path).stem + ".mono.mid")
    pmidi.write(str(midi_path))
    return midi_path

# --------------- POST-PROCESSING ----------------

def load_audio_for_beats(audio_path):
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='frames')
    beat_times = librosa.frames_to_time(beats, sr=sr)
    return float(tempo), beat_times

def make_grid(beat_times, grid_per_beat=DEFAULT_GRID, tail_sec=2.0):
    if len(beat_times) < 2:
        # fallback grid (0.5s steps up to 10 min)
        return np.arange(0, 600, 0.5)
    grid = []
    for i in range(len(beat_times)-1):
        t0, t1 = beat_times[i], beat_times[i+1]
        step = (t1 - t0) / grid_per_beat
        grid.extend([t0 + k*step for k in range(grid_per_beat)])
    grid.append(beat_times[-1])
    grid = np.array(grid, dtype=float)
    # allow tails to snap cleanly
    step = grid[1] - grid[0] if len(grid) > 1 else 0.25
    grid = np.concatenate([grid, [grid[-1] + step*2, grid[-1] + tail_sec]])
    return grid

def snap(x, grid):
    idx = np.searchsorted(grid, x)
    if idx <= 0: return grid[0]
    if idx >= len(grid): return grid[-1]
    return grid[idx] if abs(grid[idx]-x) < abs(grid[idx-1]-x) else grid[idx-1]

def quantize_pm(pmidi: pm.PrettyMIDI, grid):
    for inst in pmidi.instruments:
        for n in inst.notes:
            n.start = snap(n.start, grid)
            n.end   = max(snap(n.end, grid), n.start + 0.03)

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
        gi = max(0, min(gi, len(grid)-1))
        buckets.setdefault(gi, []).append(n)
    return buckets

def dedupe_and_merge(notes, join_gap=0.01):
    """Merge immediately repeated notes of same pitch with tiny gaps (post-quantize)."""
    by_pitch = {}
    for n in sorted(notes, key=lambda x: (x.pitch, x.start)):
        key = n.pitch
        if key not in by_pitch: by_pitch[key] = []
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
        if not notes: continue
        notes_sorted = sorted(notes, key=lambda n: n.pitch)
        bass = notes_sorted[0]
        # Extend LH to next grid line for sustain feel
        next_edge = grid[min(gi+1, len(grid)-1)]
        LH.notes.append(pm.Note(velocity=min(100, bass.velocity+5),
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

# ----------------- MAIN PIPELINE -----------------

def transcribe_to_piano(audio_path, out_dir, instrument="auto", grid_per_beat=DEFAULT_GRID,
                        max_rh_voices=None, min_note_ms=60, nosplit=False):
    """
    instrument: auto|guitar|piano|monophonic|vocals|flute|violin|sax
    """
    audio_path = pathlib.Path(audio_path)
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Choose backend + defaults
    inst = instrument.lower()
    if inst in ("piano",):
        raw_mid = run_piano_pti(audio_path, out_dir)
        default_rh = 3
    elif inst in ("monophonic","vocals","flute","violin","sax","trumpet","clarinet"):
        raw_mid = run_crepe_mono(audio_path, out_dir, min_ms=min_note_ms)
        default_rh = 1
    else:  # auto/guitar/other polyphonic
        raw_mid = run_basic_pitch(audio_path, out_dir)
        default_rh = 2

    # Beat grid and quantize
    tempo, beat_times = load_audio_for_beats(audio_path)
    grid = make_grid(beat_times, grid_per_beat=grid_per_beat)
    m = pm.PrettyMIDI(str(raw_mid))
    remove_pitch_bends(m)  # harmless for PTI/CREPE; essential for guitar->piano
    quantize_pm(m, grid)
    to_piano_program(m)

    # Save merged as piano first
    stem = audio_path.stem
    piano_mid_path = out_dir / f"{stem}.piano.mid"
    m.write(str(piano_mid_path))

    # Hand split (optional)
    split_path = None
    if not nosplit:
        piano_lr = piano_voice_split(m, grid, max_rh_voices=(max_rh_voices or default_rh),
                                     min_ms=min_note_ms)
        split_path = out_dir / f"{stem}.piano.LR.mid"
        piano_lr.write(str(split_path))

    return str(piano_mid_path), (str(split_path) if split_path else None), str(raw_mid)

# ----------------- CLI -----------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Audio stem -> piano-playable MIDI")
    ap.add_argument("audio", help="Stem path (mp3/wav/flac...)")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--instrument", default="auto",
                    help="auto|guitar|piano|monophonic|vocals|flute|violin|sax")
    ap.add_argument("--grid", type=int, default=DEFAULT_GRID,
                    help="Subdivisions per beat (2=eighths, 4=sixteenths, 3/6 for 6/8 feel)")
    ap.add_argument("--max-rh-voices", type=int, default=None,
                    help="Limit RH voices per grid cell (defaults per instrument)")
    ap.add_argument("--min-note-ms", type=int, default=60, help="Drop notes shorter than this")
    ap.add_argument("--nosplit", action="store_true", help="Do not split into L/R hands")
    args = ap.parse_args()

    piano_mid, piano_lr, raw_mid = transcribe_to_piano(
        args.audio, args.out, instrument=args.instrument,
        grid_per_beat=args.grid, max_rh_voices=args.max_rh_voices,
        min_note_ms=args.min_note_ms, nosplit=args.nosplit
    )
    print("Raw MIDI (from AMT):", raw_mid)
    print("Piano MIDI (merged):", piano_mid)
    if piano_lr:
        print("Piano MIDI (Left/Right hands):", piano_lr)
