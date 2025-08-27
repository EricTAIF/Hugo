#!/usr/bin/env python3

import requests
import json

# Test the UI server with smoothness parameters
url = 'http://localhost:5000/generate'

# Test parameters including the new smoothness controls
test_params = {
    'song': 'Snow White - Laufey (Lyrics)',
    'phraseDuration': 12000,
    'phraseOverlap': 2000,
    'minNotesPerSec': 4.0,
    'maxNotesPerSec': 8.0,
    'veryDenseRatio': 0.25,
    'denseRatio': 0.35,
    'mediumRatio': 0.50,
    'sparseRatio': 0.70,
    'veryShortPenalty': 20,
    'shortPenalty': 10,
    'minDuration': 150,
    'goodDuration': 500,
    # NEW SMOOTHNESS PARAMETERS
    'legatoEnabled': True,
    'legatoGapThreshold': 350,
    'minNoteDuration': 250,
    'sustainedHarmony': True,
    'vocalsWeight': 1.0,
    'pianoWeight': 0.8,
    'guitarWeight': 0.6,
    'bassWeight': 0.7
}

print("Testing UI server with smoothness parameters...")
print(f"Parameters: {json.dumps(test_params, indent=2)}")

try:
    response = requests.post(url, json=test_params, timeout=120)
    
    if response.status_code == 200:
        result = response.json()
        if result['success']:
            print(f"✅ SUCCESS!")
            print(f"Generated: {result['noteCount']} notes")
            print(f"Phrases: {result['phrases']}")
            print(f"Output: {result['output']}")
            print(f"File: {result['filePath']}")
        else:
            print(f"❌ Generation failed: {result['error']}")
    else:
        print(f"❌ HTTP Error: {response.status_code}")
        print(response.text)

except requests.exceptions.RequestException as e:
    print(f"❌ Connection error: {e}")
    print("Make sure the server is running with: python piano_ui_server.py")