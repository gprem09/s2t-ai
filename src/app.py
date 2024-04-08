from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
sys.path.append('src')
from speech import LiveAudioTranscriber, AudioProcessor

app = Flask(__name__)

audio_processor = AudioProcessor()
transcriber = LiveAudioTranscriber(audio_processor)

@app.route('/', methods=['GET', 'POST'])
def transcribe_audio():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(debug=True, port=8080)