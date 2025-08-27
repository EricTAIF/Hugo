#!/bin/bash
"""
Comprehensive curl test script for FastAPI backend
Tests all endpoints to ensure the API is working correctly
"""

BASE_URL="http://localhost:8000"
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🎹 Testing FastAPI Piano Arrangement Server"
echo "============================================="
echo "Base URL: $BASE_URL"
echo ""

# Test 1: Health Check
echo -e "${YELLOW}Test 1: Health Check${NC}"
echo "Command: curl -s $BASE_URL/api/health"
HEALTH_RESPONSE=$(curl -s "$BASE_URL/api/health")
HEALTH_STATUS=$?

if [ $HEALTH_STATUS -eq 0 ]; then
    echo -e "${GREEN}✅ Health check passed${NC}"
    echo "Response: $HEALTH_RESPONSE"
else
    echo -e "${RED}❌ Health check failed${NC}"
    echo "Make sure the server is running: python fastapi_server.py"
    exit 1
fi
echo ""

# Test 2: List Songs
echo -e "${YELLOW}Test 2: List Available Songs${NC}"
echo "Command: curl -s $BASE_URL/api/songs"
SONGS_RESPONSE=$(curl -s "$BASE_URL/api/songs")
SONGS_STATUS=$?

if [ $SONGS_STATUS -eq 0 ]; then
    echo -e "${GREEN}✅ Songs list retrieved${NC}"
    echo "Response: $SONGS_RESPONSE"
    
    # Extract first song name for testing
    FIRST_SONG=$(echo "$SONGS_RESPONSE" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if data['songs']:
        print(data['songs'][0]['name'])
    else:
        print('NO_SONGS')
except:
    print('PARSE_ERROR')
")
    echo "First song for testing: $FIRST_SONG"
else
    echo -e "${RED}❌ Failed to retrieve songs${NC}"
    exit 1
fi
echo ""

# Test 3: Generate Arrangement (if songs available)
if [ "$FIRST_SONG" != "NO_SONGS" ] && [ "$FIRST_SONG" != "PARSE_ERROR" ]; then
    echo -e "${YELLOW}Test 3: Generate Piano Arrangement${NC}"
    
    # Create test parameters JSON
    cat > /tmp/test_params.json << EOF
{
    "song": "$FIRST_SONG",
    "arrangementMode": "combined",
    "phraseDuration": 12000,
    "phraseOverlap": 2000,
    "minNotesPerSec": 4.0,
    "maxNotesPerSec": 8.0,
    "veryDenseRatio": 0.25,
    "denseRatio": 0.35,
    "mediumRatio": 0.50,
    "sparseRatio": 0.70,
    "veryShortPenalty": 20,
    "shortPenalty": 10,
    "minDuration": 150,
    "goodDuration": 500,
    "legatoEnabled": true,
    "legatoGapThreshold": 300,
    "minNoteDuration": 200,
    "quantizeGrid": "1/16",
    "mergeGapMs": 80,
    "sustainPedal": true,
    "sustainedHarmony": true,
    "vocalsWeight": 1.0,
    "pianoWeight": 0.8,
    "guitarWeight": 0.6,
    "bassWeight": 0.7,
    "enableEnhancedBPM": true,
    "bpmConfidenceThreshold": 0.7,
    "enhancedNoteProcessing": true,
    "noteConfidenceThreshold": 0.3,
    "aiEnabled": false
}
EOF

    echo "Command: curl -s -X POST $BASE_URL/api/generate -H 'Content-Type: application/json' -d @/tmp/test_params.json"
    GENERATE_RESPONSE=$(curl -s -X POST "$BASE_URL/api/generate" \
        -H "Content-Type: application/json" \
        -d @/tmp/test_params.json)
    GENERATE_STATUS=$?
    
    if [ $GENERATE_STATUS -eq 0 ]; then
        echo -e "${GREEN}✅ Arrangement generation completed${NC}"
        
        # Parse key metrics
        NOTE_COUNT=$(echo "$GENERATE_RESPONSE" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if data.get('success'):
        print(f\"Notes: {data.get('noteCount', 'Unknown')}\")
        print(f\"Phrases: {data.get('phrases', 'Unknown')}\")
        print(f\"BPM: {data.get('bpm', 'Unknown')}\")
        print(f\"Events: {len(data.get('events', []))}\")
        print(f\"Download URL: {data.get('downloadUrl', 'None')}\")
    else:
        print(f\"Error: {data.get('error', 'Unknown error')}\")
except Exception as e:
    print(f\"Parse error: {e}\")
")
        echo "Metrics: $NOTE_COUNT"
        
        # Save download URL for file test
        DOWNLOAD_URL=$(echo "$GENERATE_RESPONSE" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(data.get('downloadUrl', ''))
except:
    print('')
")
    else
        echo -e "${RED}❌ Arrangement generation failed${NC}"
        echo "Response: $GENERATE_RESPONSE"
    fi
    echo ""
    
    # Test 4: Download Generated File
    if [ -n "$DOWNLOAD_URL" ]; then
        echo -e "${YELLOW}Test 4: Download Generated MIDI File${NC}"
        echo "Command: curl -s -I $BASE_URL$DOWNLOAD_URL"
        FILE_RESPONSE=$(curl -s -I "$BASE_URL$DOWNLOAD_URL")
        FILE_STATUS=$?
        
        if [ $FILE_STATUS -eq 0 ] && echo "$FILE_RESPONSE" | grep -q "200 OK"; then
            echo -e "${GREEN}✅ File download available${NC}"
            echo "Headers: $(echo "$FILE_RESPONSE" | head -3)"
        else
            echo -e "${RED}❌ File download failed${NC}"
            echo "Response: $FILE_RESPONSE"
        fi
        echo ""
    fi
    
    # Test 5: Test Enhancements
    echo -e "${YELLOW}Test 5: Test Enhanced Features${NC}"
    
    cat > /tmp/test_enhance.json << EOF
{
    "song": "$FIRST_SONG"
}
EOF

    echo "Command: curl -s -X POST $BASE_URL/api/test-enhancements -H 'Content-Type: application/json' -d @/tmp/test_enhance.json"
    ENHANCE_RESPONSE=$(curl -s -X POST "$BASE_URL/api/test-enhancements" \
        -H "Content-Type: application/json" \
        -d @/tmp/test_enhance.json)
    ENHANCE_STATUS=$?
    
    if [ $ENHANCE_STATUS -eq 0 ]; then
        echo -e "${GREEN}✅ Enhancement test completed${NC}"
        
        # Parse enhancement results
        ENHANCE_RESULTS=$(echo "$ENHANCE_RESPONSE" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if data.get('success'):
        results = data.get('results', {})
        if 'bmp' in results:
            bmp = results['bmp']
            if 'error' not in bmp:
                print(f\"BPM: {bmp.get('bpm', 'Unknown'):.1f} (confidence: {bmp.get('confidence', 'Unknown'):.2f})\")
        if 'notes' in results:
            notes = results['notes']
            if 'error' not in notes:
                print(f\"Note analysis: {notes.get('original_count', 'Unknown')} -> {notes.get('enhanced_count', 'Unknown')} notes\")
        if 'arrangement' in results:
            arr = results['arrangement']
            if 'error' not in arr:
                print(f\"Arrangement: {arr.get('note_count', 'Unknown')} notes, playability: {arr.get('playability_score', 'Unknown'):.1f}/10\")
    else:
        print(f\"Enhancement test failed: {data.get('error', 'Unknown')}\")
except Exception as e:
    print(f\"Parse error: {e}\")
")
        echo "Enhancement Results: $ENHANCE_RESULTS"
    else
        echo -e "${RED}❌ Enhancement test failed${NC}"
        echo "Response: $ENHANCE_RESPONSE"
    fi
    echo ""
    
else
    echo -e "${YELLOW}⚠️ No songs available - skipping generation tests${NC}"
    echo "Upload a song first or ensure existing songs have MIDI stems"
    echo ""
fi

# Test 6: API Documentation
echo -e "${YELLOW}Test 6: API Documentation${NC}"
echo "Command: curl -s $BASE_URL/docs"
DOCS_RESPONSE=$(curl -s "$BASE_URL/docs")
DOCS_STATUS=$?

if [ $DOCS_STATUS -eq 0 ] && echo "$DOCS_RESPONSE" | grep -q "swagger"; then
    echo -e "${GREEN}✅ API documentation available${NC}"
    echo "Visit $BASE_URL/docs for interactive API documentation"
else
    echo -e "${RED}❌ API documentation not accessible${NC}"
fi
echo ""

# Test 7: Frontend Availability
echo -e "${YELLOW}Test 7: Frontend Interface${NC}"
echo "Command: curl -s -I $BASE_URL/"
FRONTEND_RESPONSE=$(curl -s -I "$BASE_URL/")
FRONTEND_STATUS=$?

if [ $FRONTEND_STATUS -eq 0 ] && echo "$FRONTEND_RESPONSE" | grep -q "200 OK"; then
    echo -e "${GREEN}✅ Frontend interface available${NC}"
    echo "Visit $BASE_URL/ for the web interface"
else
    echo -e "${RED}❌ Frontend interface not accessible${NC}"
    echo "Response: $FRONTEND_RESPONSE"
fi
echo ""

# Cleanup
rm -f /tmp/test_params.json /tmp/test_enhance.json

echo "============================================="
echo -e "${GREEN}🎉 API Testing Complete!${NC}"
echo ""
echo -e "${YELLOW}Quick Start Guide:${NC}"
echo "1. Start server: python fastapi_server.py"
echo "2. Open browser: $BASE_URL"
echo "3. API docs: $BASE_URL/docs"
echo "4. Upload songs or use existing ones"
echo "5. Generate arrangements and preview enhanced audio"
echo ""
echo -e "${YELLOW}curl Examples:${NC}"
echo "# Health check"
echo "curl $BASE_URL/api/health"
echo ""
echo "# List songs"
echo "curl $BASE_URL/api/songs"
echo ""
echo "# Generate arrangement"
echo "curl -X POST $BASE_URL/api/generate \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"song\":\"YourSongName\",\"arrangementMode\":\"combined\"}'"
echo ""