#!/usr/bin/env python3
"""
Debug script to test single stem piano functionality directly
"""

import json
import sys
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from ui_functions import create_intelligent_piano_arrangement_with_ui_params

def test_single_stem_piano():
    """Test single stem piano arrangement directly"""
    
    # Test parameters (similar to what would come from UI)
    test_params = {
        'song': 'The Beatles - Blackbird (lyrics)',
        'arrangementMode': 'single',
        'stem': 'piano',
        'phraseDuration': 15000,  # 15 seconds in ms
        'phraseOverlap': 1000,    # 1 second in ms
        'minNotesPerSec': 2,
        'maxNotesPerSec': 4,
        'veryDenseRatio': 0.15,
        'denseRatio': 0.20,
        'mediumRatio': 0.30,
        'sparseRatio': 0.50,
        # Duration parameters that were missing
        'veryShortPenalty': 30,
        'shortPenalty': 16,
        'minDuration': 200,
        'goodDuration': 600,
        # Legato parameters
        'legatoEnabled': True,
        'legatoGapThreshold': 400,
        'minNoteDuration': 300,
        'sustainedHarmony': True,
        # Weight parameters
        'vocalsWeight': 1.0,
        'pianoWeight': 0.8,
        'guitarWeight': 0.6,
        'bassWeight': 0.7,
        # Enhanced parameters
        'enableEnhancedBPM': True,
        'bpmConfidenceThreshold': 0.7,
        'enhancedNoteProcessing': True,
        'noteConfidenceThreshold': 0.3
    }
    
    print(f"Testing single stem piano with parameters:")
    print(json.dumps(test_params, indent=2))
    
    # Set up paths
    song_name = test_params['song']
    midi_dir = Path(f"outputs/midi/{song_name}")
    
    print(f"\nChecking MIDI directory: {midi_dir}")
    if not midi_dir.exists():
        print(f"ERROR: MIDI directory not found: {midi_dir}")
        return False
    
    # Check for available MIDI files
    available_stems = ["vocals", "bass", "guitar", "piano", "other", "drums"]
    midi_files = {}
    
    print(f"\nLooking for MIDI files:")
    for stem in available_stems:
        midi_path = midi_dir / f"{stem}_basic_pitch.mid"
        if midi_path.exists():
            midi_files[stem] = midi_path
            print(f"  ✓ Found: {stem}_basic_pitch.mid")
        else:
            print(f"  ✗ Missing: {stem}_basic_pitch.mid")
    
    if not midi_files:
        print("ERROR: No MIDI files found")
        return False
    
    # Check if piano stem is available
    requested_stem = test_params['stem']
    print(f"\nRequested stem: {requested_stem}")
    print(f"Available stems: {list(midi_files.keys())}")
    
    if requested_stem not in midi_files:
        print(f"ERROR: Requested stem '{requested_stem}' not available")
        print(f"Available stems: {list(midi_files.keys())}")
        return False
    
    # Filter to single stem
    single_stem_files = {requested_stem: midi_files[requested_stem]}
    print(f"Using single stem files: {single_stem_files}")
    
    # Create output file
    output_midi = midi_dir / "debug_single_stem_piano.mid"
    print(f"Output file: {output_midi}")
    
    try:
        print(f"\nStarting arrangement generation...")
        result = create_intelligent_piano_arrangement_with_ui_params(
            single_stem_files, 
            output_midi, 
            test_params
        )
        
        print(f"\n✅ SUCCESS!")
        print(f"Output: {result.get('output', 'No output message')}")
        print(f"Note count: {result.get('noteCount', 'Unknown')}")
        print(f"Phrases: {result.get('phrases', 'Unknown')}")
        print(f"BPM: {result.get('bpm', 'Unknown')}")
        print(f"Generated file: {output_midi}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during arrangement generation:")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎹 Single Stem Piano Debug Test")
    print("=" * 50)
    
    success = test_single_stem_piano()
    
    if success:
        print("\n🎉 Single stem piano test completed successfully!")
    else:
        print("\n💥 Single stem piano test failed!")
        
    print("=" * 50)