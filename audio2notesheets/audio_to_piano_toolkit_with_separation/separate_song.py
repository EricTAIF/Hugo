#!/usr/bin/env python3
"""
CLI: Separate a song into stems (Demucs or Spleeter) and transcribe each stem to MIDI with Basic Pitch.

Examples:
  # Demucs with explicit engine
  python separate_song.py "song.mp3" --out outputs --engine demucs --model htdemucs_6s --device cpu

  # Demucs but engine passed as a model name (shorthand). This now works:
  python separate_song.py "song.mp3" --out outputs --engine htdemucs_6s --device cpu

  # Spleeter (run in a separate venv ideally)
  python separate_song.py "song.mp3" --out outputs --engine spleeter --stems 5
"""
import argparse
from pathlib import Path
from main import run_demucs, run_spleeter, transcribe_with_basic_pitch, transcribe_vocals_enhanced

EXPECTED = {
    "vocals": ["vocals.mp3", "vocals.wav"],
    "bass":   ["bass.mp3", "bass.wav"],
    "drums":  ["drums.mp3", "drums.wav"],
    "other":  ["other.mp3", "other.wav"],
    "guitar": ["guitar.mp3", "guitar.wav"],
    "piano":  ["piano.mp3", "piano.wav"],
}


def _normalize_engine_and_model(engine: str, model: str):
    e = (engine or "").lower()
    if e in ("demucs", "spleeter"):
        return e, model
    # Shorthand: treat unknown engine value as a Demucs model name.
    print(f"[cli] Interpreting --engine={engine!r} as a Demucs model name.")
    return "demucs", engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to input audio (mp3/wav/flac)")
    ap.add_argument("--out", default="outputs", help="Output root directory")
    ap.add_argument("--engine", default="demucs", help="demucs | spleeter | <demucs-model-name>")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device for Demucs")
    ap.add_argument("--model", default="htdemucs", help="Demucs model name (e.g., htdemucs, htdemucs_6s)")
    ap.add_argument("--stems", type=int, choices=[2, 4, 5], default=5, help="Spleeter stems (for --engine spleeter)")
    args = ap.parse_args()

    engine, model = _normalize_engine_and_model(args.engine, args.model)

    input_audio = Path(args.input).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Separate
    try:
        if engine == "spleeter":
            sep_dir = run_spleeter(input_audio, out_root, stems=args.stems)
        else:
            sep_dir = run_demucs(input_audio, out_root, model=model, device=args.device)
    except Exception as e:
        print(f"[cli] Separation failed: {e}")
        raise SystemExit(1)

    # 2) Transcribe each available stem to MIDI (+ post-processed piano versions)
    midi_out_dir = out_root / "midi" / input_audio.stem
    midi_out_dir.mkdir(parents=True, exist_ok=True)
    made_any = False

    for stem_name, candidates in EXPECTED.items():
        audio_file = None
        for c in candidates:
            p = sep_dir / c
            if p.exists():
                audio_file = p
                break
        if not audio_file:
            continue
        print(f"[cli] Transcribing stem: {stem_name} from {audio_file.name}")
        try:
            # Use enhanced vocal processing for vocals, regular processing for others
            if stem_name == "vocals":
                midi_raw, midi_clean, midi_harmony, midi_lead, midi_original = transcribe_vocals_enhanced(audio_file, midi_out_dir, stem_name)
                # For vocals, we'll handle the friendly naming differently
                friendly = midi_out_dir / f"{stem_name}_basic_pitch.mid"
                try:
                    if Path(midi_raw).resolve() != friendly.resolve():
                        try:
                            friendly.unlink(missing_ok=True)
                        except Exception:
                            pass
                        Path(midi_raw).replace(friendly)
                        print(f"[cli] -> {friendly}")
                    else:
                        print(f"[cli] Output already at desired name: {friendly}")
                except FileNotFoundError:
                    print(f"[cli] Warning: MIDI not found when renaming: {midi_raw}")
                
                print(f"[cli] Vocals Clean (no pitch bends): {midi_clean}")
                print(f"[cli] Vocals Harmony (multi-voice): {midi_harmony}")
                print(f"[cli] Vocals Lead (melody only): {midi_lead}")
                print(f"[cli] Vocals Original (with pitch bends): {midi_original}")
                made_any = True
                continue
            else:
                midi_raw, midi_piano, midi_lr = transcribe_with_basic_pitch(audio_file, midi_out_dir, stem_name)
        except Exception as e:
            print(f"[cli] Failed to transcribe {stem_name}: {e}")
            continue

        # Also save with a friendly name for the raw Basic Pitch output
        friendly = midi_out_dir / f"{stem_name}_basic_pitch.mid"
        try:
            if Path(midi_raw).resolve() != friendly.resolve():
                try:
                    friendly.unlink(missing_ok=True)
                except Exception:
                    pass
                Path(midi_raw).replace(friendly)
                print(f"[cli] -> {friendly}")
            else:
                print(f"[cli] Output already at desired name: {friendly}")
        except FileNotFoundError:
            print(f"[cli] Warning: MIDI not found when renaming: {midi_raw}")

        if midi_piano:
            print(f"[cli] Piano (quantized): {midi_piano}")
        if midi_lr:
            print(f"[cli] Piano L/R split:  {midi_lr}")
        made_any = True

    if not made_any:
        print("[cli] Warning: no stems were transcribed to MIDI. Check separation outputs.")

    print(f"[cli] Done. Stems in: {sep_dir}")
    print(f"[cli] MIDIs in: {midi_out_dir}")


if __name__ == "__main__":
    main()
