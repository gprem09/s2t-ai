from flask import Flask, request, jsonify
import sys
sys.path.append('src')
from speech import LiveAudioTranscriber, AudioProcessor

app = Flask(__name__)

audio_processor = AudioProcessor()
transcriber = LiveAudioTranscriber(audio_processor)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    return jsonify({"transcription": "Transcribed text goes here"})

if __name__ == '__main__':
    app.run(debug=True, port=8080)