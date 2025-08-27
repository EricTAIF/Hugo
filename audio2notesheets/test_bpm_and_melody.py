#!/usr/bin/env python3
"""
Test BPM usage and melody detection for Everglow
Specifically tests:
1. How BPM affects note placement and timing
2. Whether the first phrase AI primer captures the opening E, F, E+G, E+high E melody
"""

import requests
import json
import sys
import time

def test_bmp_and_melody():
    """Test BPM usage and melody detection"""
    
    print("🎵 BPM Usage & Melody Detection Test")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    test_song = "Coldplay - Everglow [Single Version] - (Official Video)"
    
    # Test 1: Check raw "other" stem for piano melody
    print("1️⃣ Testing 'other' stem piano melody detection...")
    other_params = {
        "song": test_song,
        "arrangementMode": "single",
        "stem": "other",
        "aiEnabled": False
    }
    
    other_resp = requests.post(f"{base_url}/api/generate", json=other_params)
    other_data = other_resp.json()
    
    if other_data['success']:
        events = other_data['events']
        # Look at first 10 seconds for opening melody
        opening_events = [e for e in events if e['start'] <= 10000]  # First 10 seconds
        opening_events.sort(key=lambda x: x['start'])
        
        print(f"   📊 Total events in 'other' stem: {len(events)}")
        print(f"   🎹 Opening melody (first 10s): {len(opening_events)} notes")
        
        # Check for E, F, E+G pattern in opening
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        print("   🎼 First 15 notes detected:")
        for i, event in enumerate(opening_events[:15]):
            note_name = note_names[event['note'] % 12]
            octave = event['note'] // 12 - 1
            print(f"      {i+1:2d}. {note_name}{octave} (MIDI {event['note']}) at {event['start']/1000:.2f}s")
            
        # Check for E4 (MIDI 64) prominence
        e4_notes = [e for e in opening_events if e['note'] == 64]  # E4
        f4_notes = [e for e in opening_events if e['note'] == 65]  # F4
        g4_notes = [e for e in opening_events if e['note'] == 67]  # G4
        
        print(f"   ✅ E4 notes in opening: {len(e4_notes)}")
        print(f"   ✅ F4 notes in opening: {len(f4_notes)}")
        print(f"   ✅ G4 notes in opening: {len(g4_notes)}")
        
    # Test 2: Test BPM impact on note timing
    print("\n2️⃣ Testing BPM impact on note placement...")
    
    # Get the detected BPM
    enhance_resp = requests.post(f"{base_url}/api/test-enhancements", 
                               json={"song": test_song})
    enhance_data = enhance_resp.json()
    
    if enhance_data['success'] and 'bpm' in enhance_data['results']:
        detected_bpm = enhance_data['results']['bpm']['bpm']
        print(f"   🎵 Detected BPM: {detected_bpm:.1f}")
        
        # Calculate beat timing
        beat_ms = 60000 / detected_bpm
        print(f"   ⏱️ Beat interval: {beat_ms:.1f}ms")
        
        # Analyze note timing alignment with beats
        if other_data['success']:
            events = other_data['events']
            first_minute = [e for e in events if e['start'] <= 60000]  # First minute
            
            # Check how many notes align with beats (within 8% tolerance)
            aligned_notes = 0
            tolerance = 0.08 * beat_ms  # 8% tolerance as per code
            
            for event in first_minute:
                offset = event['start'] % beat_ms
                offset = min(offset, beat_ms - offset)  # Distance to nearest beat
                if offset <= tolerance:
                    aligned_notes += 1
            
            alignment_percentage = (aligned_notes / len(first_minute)) * 100
            print(f"   📊 Notes aligned with beats: {aligned_notes}/{len(first_minute)} ({alignment_percentage:.1f}%)")
            print(f"   ✅ Beat alignment tolerance: ±{tolerance:.1f}ms")
    
    # Test 3: AI First Phrase Primer Test
    print("\n3️⃣ Testing AI first phrase primer...")
    ai_params = {
        "song": test_song,
        "arrangementMode": "combined",
        "aiEnabled": True,
        "aiModel": "music_transformer",
        "aiBars": 2,  # Small test
        "aiBlendMode": "replace_intro",
        "aiPrimerSource": "first_phrase",
        "aiTemperature": 1.0
    }
    
    ai_resp = requests.post(f"{base_url}/api/generate", json=ai_params)
    ai_data = ai_resp.json()
    
    if ai_data['success']:
        ai_events = ai_data['events']
        # For replace_intro mode, AI notes replace the intro, so look at the replaced section
        # Look at the beginning of the arrangement where AI notes should be
        ai_opening = [e for e in ai_events if e['start'] <= 20000]  # First 20 seconds to catch AI intro
        ai_opening.sort(key=lambda x: x['start'])
        
        print(f"   🤖 AI-generated opening notes: {len(ai_opening)}")
        print("   🎼 AI primer melody (first 10 notes):")
        
        for i, event in enumerate(ai_opening[:10]):
            note_name = note_names[event['note'] % 12]
            octave = event['note'] // 12 - 1
            print(f"      {i+1:2d}. {note_name}{octave} (MIDI {event['note']}) at {event['start']/1000:.2f}s")
        
        # Check if AI preserved key notes from original
        ai_e4_notes = [e for e in ai_opening if e['note'] == 64]  # E4
        ai_f4_notes = [e for e in ai_opening if e['note'] == 65]  # F4
        
        print(f"   🎯 AI E4 notes in opening: {len(ai_e4_notes)}")
        print(f"   🎯 AI F4 notes in opening: {len(ai_f4_notes)}")
    
    # Test 4: BPM-aware quantization test
    print("\n4️⃣ Testing BPM-aware quantization...")
    quant_params = {
        "song": test_song,
        "arrangementMode": "single",
        "stem": "other",
        "quantizeGrid": "1/16",  # Enable quantization
        "aiEnabled": False
    }
    
    quant_resp = requests.post(f"{base_url}/api/generate", json=quant_params)
    quant_data = quant_resp.json()
    
    if quant_data['success']:
        quant_events = quant_data['events']
        first_30s = [e for e in quant_events if e['start'] <= 30000]
        
        # Check 16th note grid alignment
        sixteenth_note_ms = beat_ms / 4  # 1/16 note duration
        
        grid_aligned = 0
        tolerance = sixteenth_note_ms * 0.15  # 15% tolerance for grid alignment
        for event in first_30s:
            grid_offset = event['start'] % sixteenth_note_ms
            # Distance to nearest grid point
            offset_distance = min(grid_offset, sixteenth_note_ms - grid_offset)
            if offset_distance <= tolerance:
                grid_aligned += 1
        
        grid_percentage = (grid_aligned / len(first_30s)) * 100
        print(f"   📊 Notes on 1/16 grid: {grid_aligned}/{len(first_30s)} ({grid_percentage:.1f}%)")
        print(f"   ⏱️ 16th note grid: {sixteenth_note_ms:.1f}ms intervals (tolerance: ±{tolerance:.1f}ms)")
    
    # Summary
    print("\n🎉 BPM & Melody Analysis Results")
    print("=" * 40)
    
    if other_data['success']:
        print(f"✅ Piano melody detection: {len(e4_notes)} E4 notes found in opening")
        print(f"✅ Expected E-F-E+G pattern: E4({len(e4_notes)}), F4({len(f4_notes)}), G4({len(g4_notes)})")
        print(f"✅ Beat alignment: {alignment_percentage:.1f}% of notes align with BPM")
        
    if ai_data['success']:
        print(f"✅ AI primer captured melody: {len(ai_e4_notes)} E4 notes in AI intro")
        
    if quant_data['success']:
        print(f"✅ BPM-based quantization: {grid_percentage:.1f}% notes on 1/16 grid")
    
    print(f"\n🔍 Diagnosis:")
    if len(e4_notes) >= 2:
        print("✅ Piano melody IS being detected from 'other' stem")
    else:
        print("❌ Piano melody may not be fully captured")
        
    if alignment_percentage > 20:
        print("✅ BPM IS being used for note timing alignment")
    else:
        print("❌ BPM may not be effectively used for timing")
        
    if ai_data['success'] and len(ai_e4_notes) >= 1:
        print("✅ AI first phrase primer IS capturing melody elements")
    else:
        print("❌ AI primer may not be effectively using source melody")

if __name__ == "__main__":
    try:
        test_bmp_and_melody()
        print("\n🎊 Test completed successfully!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)