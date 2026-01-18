"""
APPLICATION FLASK : SPEECH-TO-TEXT + TRADUCTION MULTILINGUE
Déploiement du modèle Whisper fine-tuné + MarianMT
"""

from flask import Flask, render_template, request, jsonify
import torch
import os
import librosa
import soundfile as sf
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    MarianMTModel,
    MarianTokenizer
)
from werkzeug.utils import secure_filename
import traceback
import json
from datetime import datetime

 
# CONFIGURATION 

class Config:
    """Configuration de l'application"""
    
    # Chemins
    MODEL_PATH = "./models/whisper-trained"  # Chemin vers votre modèle
    UPLOAD_FOLDER = "./uploads"
    
    # Extensions autorisées
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a'}
    
    # Taille max fichier (16 MB)
    MAX_FILE_SIZE = 16 * 1024 * 1024
    
    # Langues supportées pour traduction
    SUPPORTED_LANGUAGES = {
        'fr': {'name': 'Français', 'model': 'Helsinki-NLP/opus-mt-en-fr'},
        'es': {'name': 'Español', 'model': 'Helsinki-NLP/opus-mt-en-es'},
        'ar': {'name': 'العربية', 'model': 'Helsinki-NLP/opus-mt-en-ar'},
        'de': {'name': 'Deutsch', 'model': 'Helsinki-NLP/opus-mt-en-de'},
        'it': {'name': 'Italiano', 'model': 'Helsinki-NLP/opus-mt-en-it'},
        'pt': {'name': 'Português', 'model': 'Helsinki-NLP/opus-mt-en-pt'},
    }
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# INITIALISATION FLASK


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_FILE_SIZE

# Créer dossier uploads
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)


print(" INITIALISATION DE L'APPLICATION")




# CHARGEMENT DES MODÈLES


class ModelManager:
    """Gestionnaire des modèles (STT + Traduction)"""
    
    def __init__(self):
        self.device = Config.DEVICE
        self.stt_model = None
        self.stt_processor = None
        self.translation_models = {}
        self.translation_tokenizers = {}
        
        print(f"\n Device: {self.device}")
        self.load_stt_model()
    
    def load_stt_model(self):
        """Charger le modèle STT Whisper"""
        
        print("\n Chargement du modèle STT...")
        
        try:
            if not os.path.exists(Config.MODEL_PATH):
                raise FileNotFoundError(
                    f" Modèle introuvable : {Config.MODEL_PATH}\n"
                    f" Assurez-vous d'avoir téléchargé et décompressé le modèle depuis Colab"
                )
            
            self.stt_processor = WhisperProcessor.from_pretrained(Config.MODEL_PATH)
            self.stt_model = WhisperForConditionalGeneration.from_pretrained(Config.MODEL_PATH)
            self.stt_model.to(self.device)
            self.stt_model.eval()
            
            print(f"Modèle STT chargé depuis {Config.MODEL_PATH}")
            
        except Exception as e:
            print(f" ERREUR lors du chargement du modèle STT : {e}")
            raise
    
    def load_translation_model(self, target_lang):
        """Charger un modèle de traduction à la demande"""
        
        if target_lang in self.translation_models:
            return  # Déjà chargé
        
        if target_lang not in Config.SUPPORTED_LANGUAGES:
            raise ValueError(f"Langue non supportée : {target_lang}")
        
        model_name = Config.SUPPORTED_LANGUAGES[target_lang]['model']
        print(f"\n Chargement du modèle de traduction {target_lang}...")
        
        try:
            self.translation_tokenizers[target_lang] = MarianTokenizer.from_pretrained(model_name)
            self.translation_models[target_lang] = MarianMTModel.from_pretrained(model_name)
            self.translation_models[target_lang].to(self.device)
            self.translation_models[target_lang].eval()
            
            print(f"Modèle de traduction {target_lang} chargé")
            
        except Exception as e:
            print(f" Erreur chargement traduction {target_lang}: {e}")
            raise
    
    def transcribe(self, audio_path):
        """Transcrire un fichier audio"""
        
        # Charger l'audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Préparer les inputs
        inputs = self.stt_processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device)
        
        # Transcrire
        with torch.no_grad():
            predicted_ids = self.stt_model.generate(
                input_features,
                max_length=225
            )
        
        # Décoder
        transcription = self.stt_processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        return transcription
    
    def translate(self, text, target_lang):
        """Traduire un texte"""
        
        # Charger le modèle si nécessaire
        if target_lang not in self.translation_models:
            self.load_translation_model(target_lang)
        
        # Tokenize
        inputs = self.translation_tokenizers[target_lang](
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Traduire
        with torch.no_grad():
            outputs = self.translation_models[target_lang].generate(**inputs)
        
        # Décoder
        translation = self.translation_tokenizers[target_lang].decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        return translation


# Initialiser le gestionnaire de modèles
model_manager = ModelManager()

print("\ Application prête!")


 
# UTILITAIRES


def allowed_file(filename):
    """Vérifier si l'extension du fichier est autorisée"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def convert_to_wav(input_path, output_path):
    """Convertir un fichier audio en WAV 16kHz"""
    try:
        audio, sr = librosa.load(input_path, sr=16000)
        sf.write(output_path, audio, 16000)
        return True
    except Exception as e:
        print(f"Erreur conversion audio : {e}")
        return False



# ROUTES


@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('home.html')


@app.route('/api/info', methods=['GET'])
def get_info():
    """Informations sur l'API"""
    return jsonify({
        'status': 'ok',
        'model': 'Whisper Fine-tuned',
        'device': str(Config.DEVICE),
        'supported_languages': list(Config.SUPPORTED_LANGUAGES.keys()),
        'max_file_size_mb': Config.MAX_FILE_SIZE / (1024*1024),
        'allowed_formats': list(Config.ALLOWED_EXTENSIONS)
    })


@app.route('/api/languages', methods=['GET'])
def get_languages():
    """Liste des langues supportées"""
    return jsonify({
        lang: info['name'] 
        for lang, info in Config.SUPPORTED_LANGUAGES.items()
    })


@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    """
    Endpoint principal : transcription + traduction
    
    Paramètres:
        - file: fichier audio (wav, mp3, etc.)
        - languages: liste des langues cibles (optionnel)
    
    Retour:
        - transcription: texte transcrit
        - translations: dictionnaire {lang: traduction}
        - processing_time: temps de traitement
    """
    
    try:
        print('\n--- /api/transcribe called ---')
        try:
            print('Incoming files:', list(request.files.keys()))
            print('Incoming form keys:', list(request.form.keys()))
        except Exception as _:
            print('Could not read request.files/form')
        # Vérifier le fichier
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier fourni'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Nom de fichier vide'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Format non supporté. Utilisez: {", ".join(Config.ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Récupérer les langues cibles
        target_languages = request.form.getlist('languages[]')
        if not target_languages:
            target_languages = ['fr']  # Par défaut
        
        # Valider les langues
        invalid_langs = [l for l in target_languages if l not in Config.SUPPORTED_LANGUAGES]
        if invalid_langs:
            return jsonify({
                'error': f'Langues non supportées: {", ".join(invalid_langs)}'
            }), 400
        
        # Sauvegarder le fichier
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Convertir en WAV si nécessaire
        wav_path = filepath
        if not filename.lower().endswith('.wav'):
            wav_path = filepath.rsplit('.', 1)[0] + '.wav'
            if not convert_to_wav(filepath, wav_path):
                return jsonify({'error': 'Erreur de conversion audio'}), 500
        
        # Mesurer le temps
        start_time = datetime.now()
        
        # Transcription
        print(f"\n Transcription de {filename}...")
        transcription = model_manager.transcribe(wav_path)
        print(f"Transcription : {transcription}")
        
        # Traductions
        translations = {}
        for lang in target_languages:
            print(f" Traduction en {lang}...")
            try:
                translation = model_manager.translate(transcription, lang)
                translations[lang] = translation
                print(f"{lang.upper()}: {translation}")
            except Exception as e:
                print(f" Erreur traduction {lang}: {e}")
                translations[lang] = f"Erreur: {str(e)}"
        
        # Temps de traitement
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Nettoyer les fichiers temporaires
        try:
            os.remove(filepath)
            if wav_path != filepath:
                os.remove(wav_path)
        except:
            pass
        
        # Retourner les résultats
        return jsonify({
            'success': True,
            'transcription': transcription,
            'translations': translations,
            'processing_time': round(processing_time, 2),
            'target_languages': target_languages
        })
    
    except Exception as e:
        print(f"\ ERREUR: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Vérification de l'état de l'API"""
    return jsonify({
        'status': 'healthy',
        'stt_model_loaded': model_manager.stt_model is not None,
        'device': str(Config.DEVICE),
        'timestamp': datetime.now().isoformat()
    })


# 
# GESTION DES ERREURS
# 

@app.errorhandler(413)
def request_entity_too_large(error):
    """Fichier trop volumineux"""
    return jsonify({
        'error': f'Fichier trop volumineux (max {Config.MAX_FILE_SIZE/(1024*1024):.0f} MB)'
    }), 413


@app.errorhandler(404)
def not_found(error):
    """Page non trouvée"""
    return jsonify({'error': 'Endpoint non trouvé'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Erreur interne"""
    return jsonify({'error': 'Erreur interne du serveur'}), 500


# UI routes for additional features

@app.route('/tts', methods=['GET'])
def tts_page():
    """Page TTS"""
    return render_template('tts_trad.html', languages=Config.SUPPORTED_LANGUAGES, device=str(Config.DEVICE))


@app.route('/diarization', methods=['GET'])
def diarization_page():
    """Page Diarization"""
    return render_template('diarization_sentiment.html', languages=Config.SUPPORTED_LANGUAGES, device=str(Config.DEVICE))


@app.route('/sentiment', methods=['GET'])
def sentiment_page():
    """Page Sentiment Analysis (placeholder)"""
    return render_template('diarization_sentiment.html', languages=Config.SUPPORTED_LANGUAGES, device=str(Config.DEVICE))


@app.route('/stt', methods=['GET'])
def stt_page():
    """Page STT + Traduction"""
    return render_template('stt.html', languages=Config.SUPPORTED_LANGUAGES, device=str(Config.DEVICE))


# API placeholders for future features


@app.route('/api/tts', methods=['POST'])
def api_tts():
    """Placeholder TTS endpoint. Expects JSON {text: ...}. Returns 501 for now."""
    try:
        data = request.get_json(force=True)
        text = data.get('text', '') if data else ''
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        # Placeholder response: real TTS not implemented yet
        return jsonify({'error': 'TTS not implemented', 'text': text}), 501
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/diarize', methods=['POST'])
def api_diarize():
    """Placeholder diarization endpoint. Accepts file upload. Returns 501."""
    return jsonify({'error': 'Diarization not implemented'}), 501


@app.route('/api/sentiment', methods=['POST'])
def api_sentiment():
    """Placeholder sentiment endpoint. Expects JSON {text: ...}. Returns demo response."""
    try:
        data = request.get_json(force=True)
        text = data.get('text', '') if data else ''
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        # Demo: very naive sentiment stub
        score = 0.0
        label = 'neutral'
        lowered = text.lower()
        if any(w in lowered for w in ['good', 'great', 'merci', 'bien', 'love', 'super']):
            score = 0.95
            label = 'positive'
        elif any(w in lowered for w in ['bad', 'hate', 'mauvais', 'pas', 'déteste', 'désolé']):
            score = 0.05
            label = 'negative'
        else:
            score = 0.5
            label = 'neutral'

        return jsonify({'label': label, 'score': score})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 
# DÉMARRAGE
# 

if __name__ == '__main__':
    print(" DÉMARRAGE DU SERVEUR FLASK")
    print(f"\n URL locale : http://127.0.0.1:5000")
    print(f" URL réseau : http://0.0.0.0:5000")
    print("\n Appuyez sur Ctrl+C pour arrêter")

    
    app.run(
        host='0.0.0.0',  # Accessible depuis le réseau
        port=5000,
        debug=True,      # Mode développement
        threaded=True    # Support multi-threads
    )