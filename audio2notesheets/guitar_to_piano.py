import argparse, pathlib, os
import numpy as np
import pretty_midi as pm
import librosa
from basic_pitch.inference import predict_and_save

PIANO_MIN, PIANO_MAX = 21, 108        # A0..C8
GRID_PER_BEAT = 4                      # 4 = sixteenth-notes; try 2 for eighths
MIN_NOTE_MS = 60
# --- replace your run_basic_pitch(...) with this ---
from pathlib import Path
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH  # <-- default model path

def run_basic_pitch(audio_path, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    audio_path = Path(audio_path)

    # Use positional args (works across versions)
    predict_and_save(
        [str(audio_path)],            # <input-audio-path-list>
        str(out_dir),                 # <output-directory>
        True,                         # save_midi
        False,                        # sonify_midi
        False,                        # save_model_outputs
        False,                        # save_notes
        ICASSP_2022_MODEL_PATH        # <model path>  <-- required in your version
    )

    # Pick the produced .mid
    stem = audio_path.stem
    for cand in out_dir.glob(f"{stem}*.mid"):
        return cand
    raise FileNotFoundError("No MIDI produced by Basic Pitch")



def load_audio_for_beats(audio_path):
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units='frames')
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return float(tempo), beat_times

def make_grid(beat_times, grid_per_beat=GRID_PER_BEAT, tail_sec=2.0):
    if len(beat_times) < 2:
        # fall back to a simple fixed grid if beats failed
        return np.arange(0, 600, 0.5)  # generous dummy grid
    grid = []
    for i in range(len(beat_times)-1):
        t0, t1 = beat_times[i], beat_times[i+1]
        step = (t1 - t0) / grid_per_beat
        for k in range(grid_per_beat):
            grid.append(t0 + k*step)
    grid.append(beat_times[-1])
    grid = np.array(grid)
    # extend a little to catch note tails
    grid = np.concatenate([grid, [grid[-1] + (grid[1]-grid[0]) * 2, grid[-1] + tail_sec]])
    return grid

def snap(x, grid):
    idx = np.searchsorted(grid, x)
    if idx <= 0: return grid[0]
    if idx >= len(grid): return grid[-1]
    return grid[idx] if abs(grid[idx]-x) < abs(grid[idx-1]-x) else grid[idx-1]

def clamp_and_clean(notes):
    out = []
    for n in notes:
        if n.pitch < PIANO_MIN or n.pitch > PIANO_MAX: 
            continue
        if (n.end - n.start) * 1000 < MIN_NOTE_MS:
            continue
        out.append(n)
    return out

def quantize_pm(pmidi, grid):
    for inst in pmidi.instruments:
        for n in inst.notes:
            n.start = snap(n.start, grid)
            n.end   = max(snap(n.end, grid), n.start + 0.03)

def remove_pitch_bends(pmidi):
    for inst in pmidi.instruments:
        inst.pitch_bends = []

def to_piano_program(pmidi):
    for inst in pmidi.instruments:
        inst.program = 0
        inst.is_drum = False

def group_by_grid(notes, grid):
    # bucket notes by nearest grid index based on start time
    buckets = dict()
    for n in notes:
        gi = int(np.searchsorted(grid, n.start) - 1)
        gi = max(0, min(gi, len(grid)-1))
        buckets.setdefault(gi, []).append(n)
    return buckets

def piano_voice_split(pmidi, grid, max_rh_voices=2):
    # Merge all source instruments into one pool first
    pool = []
    for inst in pmidi.instruments:
        pool.extend(inst.notes)
    pool = clamp_and_clean(pool)

    buckets = group_by_grid(pool, grid)

    LH = pm.Instrument(program=0, name="Left Hand")
    RH = pm.Instrument(program=0, name="Right Hand")

    for gi, notes in sorted(buckets.items()):
        if not notes: continue
        # choose left hand = lowest pitch for the cell
        notes_sorted = sorted(notes, key=lambda n: n.pitch)
        bass = notes_sorted[0]
        # slightly lengthen LH to next grid (simulate sustain)
        if gi + 1 < len(grid):
            bass_end = max(bass.end, grid[gi+1])
        else:
            bass_end = bass.end
        LH.notes.append(pm.Note(velocity=min(100, bass.velocity+5),
                                pitch=bass.pitch, start=bass.start, end=bass_end))

        # right hand: take up to max_rh_voices highest pitches
        ups = sorted(notes_sorted[1:], key=lambda n: n.pitch, reverse=True)[:max_rh_voices]
        for u in ups:
            RH.notes.append(pm.Note(velocity=u.velocity, pitch=u.pitch, start=u.start, end=u.end))

    out = pm.PrettyMIDI(initial_tempo=pmidi.estimate_tempo() or 120.0)
    out.instruments = [LH, RH]
    return out

def process(audio_path, out_dir, grid_per_beat=GRID_PER_BEAT):
    audio_path = pathlib.Path(audio_path)
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) AMT: guitar audio -> MIDI
    raw_mid = run_basic_pitch(audio_path, out_dir)

    # 2) Beat grid (from audio) and quantize
    tempo, beat_times = load_audio_for_beats(audio_path)
    grid = make_grid(beat_times, grid_per_beat=grid_per_beat)

    m = pm.PrettyMIDI(str(raw_mid))
    remove_pitch_bends(m)   # guitar PBs aren’t useful for piano tutorials
    quantize_pm(m, grid)
    to_piano_program(m)

    # 3) Voice split & simplification for piano tutorial
    piano_lr = piano_voice_split(m, grid, max_rh_voices=2)

    # 4) Save
    stem = audio_path.stem
    piano_mid_path = out_dir / f"{stem}.piano.mid"
    piano_lr_path  = out_dir / f"{stem}.piano.LR.mid"
    m.write(str(piano_mid_path))
    piano_lr.write(str(piano_lr_path))
    return str(piano_mid_path), str(piano_lr_path), str(raw_mid)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="Guitar stem (mp3/wav/flac...)")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--grid", type=int, default=GRID_PER_BEAT, help="Subdivisions per beat (2=eighths, 4=sixteenths)")
    args = ap.parse_args()
    piano_mid, piano_lr, raw_mid = process(args.audio, args.out, grid_per_beat=args.grid)
    print("Raw MIDI (from AMT):", raw_mid)
    print("Piano MIDI (merged):", piano_mid)
    print("Piano MIDI (Left/Right hands):", piano_lr)
