#!/usr/bin/env python3

"""
AI Arranger backends.

Provides a pluggable interface:
- If Magenta / note-seq Music Transformer is available (and a checkpoint path is configured),
  use it to generate a polyphonic continuation conditioned on a primer.
- Otherwise, fall back to a lightweight n-gram/Markov generator trained on primer/context events.

All times are milliseconds in and out. Notes are MIDI integers 0..127.
"""

from __future__ import annotations
from typing import List, Dict, Any

import os


def _has_magenta() -> bool:
    try:
        import note_seq  # noqa: F401
        from magenta.models.transformer import transformer_model  # noqa: F401
        return True
    except Exception:
        return False


def generate_with_ai(primer_events: List[Dict[str, int]], bpm: int, params: Dict[str, Any]) -> List[Dict[str, int]]:
    """Generate events using AI backend.

    - primer_events: list of {'note', 'start', 'duration', 'velocity'} in ms
    - bpm: int
    - params: contains keys like 'aiModel', 'aiTemperature', 'aiBars', 'device'
    """
    model = params.get('aiModel', 'music_transformer')
    if model == 'music_transformer' and _has_magenta():
        try:
            return _generate_with_magenta(primer_events, bpm, params)
        except Exception as e:
            print(f"[ai] Magenta backend failed: {e}. Falling back to Markov.")
    # Fallback
    return _generate_with_markov(primer_events, bpm, params)


def _generate_with_magenta(primer_events, bpm: int, params: Dict[str, Any]):
    """Use Magenta Music Transformer to generate continuation.
    Requires note_seq + magenta to be installed and a checkpoint path in env AI_MXT_CHECKPOINT or params['checkpoint'].
    """
    import note_seq
    from note_seq.protobuf import music_pb2
    from note_seq import sequences_lib
    from magenta.models.transformer import transformer_model
    from magenta.models.transformer import transformer_recorder
    import tensorflow as tf

    # Build NoteSequence from primer
    qpm = float(bpm)
    seq = music_pb2.NoteSequence()
    seq.tempos.add(qpm=qpm)
    for ev in primer_events:
        n = seq.notes.add()
        n.pitch = int(ev['note'])
        n.start_time = max(0.0, ev['start'] / 1000.0)
        n.end_time = n.start_time + max(0.05, ev['duration'] / 1000.0)
        n.velocity = int(ev.get('velocity', 80))
    seq.total_time = max([getattr(n, 'end_time', 0.0) for n in seq.notes] + [0.0])

    # Load model/checkpoint
    ckpt = params.get('checkpoint') or os.environ.get('AI_MXT_CHECKPOINT')
    if not ckpt or not os.path.exists(ckpt):
        raise RuntimeError("Music Transformer checkpoint not found. Set AI_MXT_CHECKPOINT or pass 'checkpoint' in params.")

    # Configure generation length
    bars = int(params.get('aiBars', 8))
    seconds_per_bar = 60.0 / max(1.0, qpm) * 4.0
    generate_seconds = max(2.0, bars * seconds_per_bar)

    temperature = float(params.get('aiTemperature', 1.0))
    beam_size = int(params.get('aiBeamSize', 1))

    # Use GPU if available
    device = '/GPU:0' if (params.get('useGPU', True) and tf.config.list_physical_devices('GPU')) else '/CPU:0'

    config = transformer_model.TransformerConfig()
    with tf.device(device):
        model = transformer_model.Transformer(config)
        # Restore
        ckpt_obj = tf.train.Checkpoint(model=model)
        ckpt_obj.restore(ckpt).expect_partial()

        # Encode primer and generate
        # Note: exact API differs by Magenta version; this is a placeholder outline.
        generator = transformer_recorder.TransformerGenerator(model)
        generated = generator.generate(seq, total_seconds=generate_seconds, temperature=temperature, beam_size=beam_size)

    # Convert back to events (ms)
    events = []
    for n in generated.notes:
        events.append({
            'note': int(n.pitch),
            'start': int(round(n.start_time * 1000)),
            'duration': int(round(max(50, (n.end_time - n.start_time) * 1000))),
            'velocity': int(getattr(n, 'velocity', 80))
        })
    return sorted(events, key=lambda e: e['start'])


def _generate_with_markov(primer_events, bpm: int, params: Dict[str, Any]):
    """Simple n-gram Markov generator on (pitch class, interval) with duration bins.
    Not state-of-the-art but provides a testable AI-like continuation without external weights.
    """
    import random
    random.seed(42)

    if not primer_events:
        return []

    # Prepare sequences
    events = sorted(primer_events, key=lambda e: e['start'])
    pcs = [e['note'] % 12 for e in events]
    intervals = [0] + [((events[i]['note'] - events[i-1]['note'])) for i in range(1, len(events))]
    dur_bins = [max(1, int(round(e['duration'] / 50.0))) for e in events]  # ~50ms bins

    # Build transition maps
    trans = {}
    for i in range(1, len(pcs)):
        key = (pcs[i-1], intervals[i-1], dur_bins[i-1])
        nxt = (pcs[i], intervals[i], dur_bins[i])
        trans.setdefault(key, {})
        trans[key][nxt] = trans[key].get(nxt, 0) + 1

    def sample_next(state):
        choices = trans.get(state)
        if not choices:
            # Backoff: random
            pc = random.choice(pcs)
            return (pc, random.choice([-2, -1, 0, 1, 2]), random.choice(dur_bins))
        items = list(choices.items())
        total = sum(w for _, w in items)
        r = random.uniform(0, total)
        cum = 0
        for nxt, w in items:
            cum += w
            if r <= cum:
                return nxt
        return items[-1][0]

    # Generate continuation for requested bars
    qpm = bpm
    beat_ms = 60000.0 / max(1, qpm)
    seconds_per_bar = 4 * beat_ms / 1000.0
    bars = int(params.get('aiBars', 8))
    total_ms = int(bars * seconds_per_bar * 1000)

    start_time = max(e['start'] + e['duration'] for e in events)
    # Start near the last event state
    state = (pcs[-1], intervals[-1], dur_bins[-1])

    out = []
    abs_t = start_time
    current_pitch = events[-1]['note']
    while abs_t - start_time < total_ms:
        pc, interval, db = sample_next(state)
        current_pitch += interval
        # Constrain to piano range via wrapping
        while current_pitch < 24:
            current_pitch += 12
        while current_pitch > 96:
            current_pitch -= 12
        dur = max(50, int(db * 50))
        out.append({'note': int(current_pitch), 'start': int(abs_t), 'duration': int(dur), 'velocity': 80})
        abs_t += dur
        state = (pc, interval, db)

    return out

