"""
Flask Backend for Audio Translator
Deploy to Railway or Heroku
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import librosa
import speech_recognition as sr
import numpy as np
from gtts import gTTS
from transformers import MarianMTModel, MarianTokenizer
import os
import io
from werkzeug.utils import secure_filename
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'opus', 'flac', 'm4a'}
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Language configuration
LANGUAGES = {
    "tamil": {"code": "ta", "label": "Tamil"},
    "english": {"code": "en", "label": "English"},
    "hindi": {"code": "hi", "label": "Hindi"}
}

TRANSLATION_MODELS = {
    ("tamil", "english"): "Helsinki-NLP/Opus-MT-ta-en",
    ("tamil", "hindi"): "Helsinki-NLP/Opus-MT-ta-hi",
    ("english", "tamil"): "Helsinki-NLP/Opus-MT-en-ta",
    ("english", "hindi"): "Helsinki-NLP/Opus-MT-en-hi",
    ("hindi", "tamil"): "Helsinki-NLP/Opus-MT-hi-ta",
    ("hindi", "english"): "Helsinki-NLP/Opus-MT-hi-en",
}

# ============================================
# HELPER FUNCTIONS
# ============================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_audio(audio_input, sr_rate):
    """Clean audio for better recognition"""
    processed = np.array(audio_input, dtype=np.float32)
    
    # Normalize
    max_val = np.max(np.abs(processed))
    if max_val > 0:
        processed = processed / max_val * 0.9
    
    # Amplify if quiet
    rms = np.sqrt(np.mean(processed ** 2))
    if rms < 0.05:
        processed = processed * 2.0
    
    return processed

def audio_to_text(audio_path, language="english"):
    """Convert audio to text"""
    try:
        audio_data, sample_rate = librosa.load(audio_path, sr=16000)
        audio_data = preprocess_audio(audio_data, sample_rate)
        
        audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
        audio_object = sr.AudioData(audio_bytes, 16000, 2)
        
        recognizer = sr.Recognizer()
        lang_code = LANGUAGES[language]["code"]
        text = recognizer.recognize_google(audio_object, language=lang_code)
        
        return {"success": True, "text": text}
    except sr.UnknownValueError:
        return {"success": False, "error": "Could not understand audio"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def translate_text(text, source_lang, target_lang):
    """Translate text"""
    if source_lang == target_lang:
        return {"success": True, "text": text}
    
    try:
        model_key = (source_lang, target_lang)
        if model_key not in TRANSLATION_MODELS:
            return {"success": False, "error": "Language pair not supported"}
        
        model_name = TRANSLATION_MODELS[model_key]
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model.generate(**inputs)
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {"success": True, "text": translated}
    except Exception as e:
        return {"success": False, "error": str(e)}

def text_to_speech(text, language_code, slow=False):
    """Convert text to speech, return as bytes"""
    try:
        tts = gTTS(text=text, lang=language_code, slow=slow)
        
        # Save to bytes buffer instead of file
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return {"success": True, "audio": audio_buffer}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({"status": "ok", "message": "Audio Translator API is running"})

@app.route('/api/languages', methods=['GET'])
def get_languages():
    """Get available languages"""
    return jsonify({
        "languages": list(LANGUAGES.keys()),
        "details": LANGUAGES
    })

@app.route('/api/translate', methods=['POST'])
def translate():
    """
    Main translation endpoint
    
    Expected form data:
    - audio_file: Audio file (multipart)
    - source_lang: Source language (tamil/english/hindi)
    - target_lang: Target language (tamil/english/hindi)
    - slow_output: Boolean (optional)
    """
    
    try:
        # Validate request
        if 'audio_file' not in request.files:
            return jsonify({"success": False, "error": "No audio file provided"}), 400
        
        if 'source_lang' not in request.form or 'target_lang' not in request.form:
            return jsonify({"success": False, "error": "Language parameters missing"}), 400
        
        file = request.files['audio_file']
        source_lang = request.form['source_lang'].lower()
        target_lang = request.form['target_lang'].lower()
        slow_output = request.form.get('slow_output', 'false').lower() == 'true'
        
        # Validate file
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Invalid file format"}), 400
        
        # Validate languages
        if source_lang not in LANGUAGES or target_lang not in LANGUAGES:
            return jsonify({"success": False, "error": "Invalid language"}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        try:
            # Step 1: Speech to Text
            result = audio_to_text(temp_path, language=source_lang)
            if not result['success']:
                return jsonify(result), 400
            
            original_text = result['text']
            
            # Step 2: Translation
            result = translate_text(original_text, source_lang, target_lang)
            if not result['success']:
                return jsonify(result), 400
            
            translated_text = result['text']
            
            # Step 3: Text to Speech
            result = text_to_speech(
                translated_text, 
                LANGUAGES[target_lang]['code'],
                slow=slow_output
            )
            if not result['success']:
                return jsonify(result), 400
            
            audio_buffer = result['audio']
            
            # Return results
            return jsonify({
                "success": True,
                "original_text": original_text,
                "translated_text": translated_text,
                "source_language": source_lang,
                "target_language": target_lang,
                "has_audio": True
            })
        
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/translate-with-audio', methods=['POST'])
def translate_with_audio():
    """
    Translation endpoint that returns audio file
    Same as /api/translate but returns the audio file
    """
    
    try:
        if 'audio_file' not in request.files:
            return jsonify({"success": False, "error": "No audio file"}), 400
        
        if 'source_lang' not in request.form or 'target_lang' not in request.form:
            return jsonify({"success": False, "error": "Language parameters missing"}), 400
        
        file = request.files['audio_file']
        source_lang = request.form['source_lang'].lower()
        target_lang = request.form['target_lang'].lower()
        slow_output = request.form.get('slow_output', 'false').lower() == 'true'
        
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Invalid file"}), 400
        
        if source_lang not in LANGUAGES or target_lang not in LANGUAGES:
            return jsonify({"success": False, "error": "Invalid language"}), 400
        
        # Save temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        try:
            # Step 1: Speech to Text
            result = audio_to_text(temp_path, language=source_lang)
            if not result['success']:
                return jsonify(result), 400
            original_text = result['text']
            
            # Step 2: Translate
            result = translate_text(original_text, source_lang, target_lang)
            if not result['success']:
                return jsonify(result), 400
            translated_text = result['text']
            
            # Step 3: Text to Speech
            result = text_to_speech(
                translated_text,
                LANGUAGES[target_lang]['code'],
                slow=slow_output
            )
            if not result['success']:
                return jsonify(result), 400
            
            audio_buffer = result['audio']
            
            # Return audio file with metadata
            return send_file(
                audio_buffer,
                mimetype='audio/mpeg',
                as_attachment=True,
                download_name=f'translated_{target_lang}.mp3',
                headers={
                    'X-Original-Text': original_text,
                    'X-Translated-Text': translated_text
                }
            )
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ============================================
# ERROR HANDLERS
# ============================================

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"success": False, "error": "File too large (max 25MB)"}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": "Server error"}), 500

# ============================================
# RUN
# ============================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
