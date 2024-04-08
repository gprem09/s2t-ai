from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import speech

app = Flask(__name__)
CORS(app)

audio_processor = speech.AudioProcessor()
transcriber = speech.LiveAudioTranscriber(audio_processor)

@app.route('/transcribe', methods=['GET','POST'])
def transcribe_audio():
    audio_data = request.json['audioData']
    sample_rate = request.json['sampleRate']
    
    transcription = transcriber.audio_processor.transcribe(audio_data, sample_rate)
    return jsonify({"transcription": transcription})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
