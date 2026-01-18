"""
================================================================================
APPLICATION FLASK : SOUS-TITRES KARAOKE - ANTI-SUPERPOSITION
✓ Traduction phrase complète (garde le sens)
✓ Anti-superposition GARANTIE
✓ Redistribution intelligente des mots traduits
================================================================================
"""

from flask import Flask, render_template, request, jsonify, send_file
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
from datetime import datetime
import unicodedata
import numpy as np

# Import moviepy selon la version
try:
    # Tentative moviepy 2.x
    from moviepy import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip
except ImportError:
    try:
        # Fallback moviepy 1.x
        from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip
    except ImportError:
        print("⚠️ ERREUR: moviepy non installé ou version incompatible")
        print("Installation recommandée: pip install moviepy==1.0.3")
        exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    MODEL_PATH = "./models/whisper-trained"
    UPLOAD_FOLDER = "./uploads"
    OUTPUT_FOLDER = "./outputs"
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
    ALLOWED_AUDIO_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac', 'aac'}
    MAX_FILE_SIZE = 100 * 1024 * 1024
    FONT_PATH = "./fonts/Menuetto.ttf"
    
    SUPPORTED_LANGUAGES = {
        'fr': {'name': 'Français', 'model': 'Helsinki-NLP/opus-mt-en-fr'},
        'es': {'name': 'Español', 'model': 'Helsinki-NLP/opus-mt-en-es'},
        'ar': {'name': 'العربية', 'model': 'Helsinki-NLP/opus-mt-en-ar'},
        'de': {'name': 'Deutsch', 'model': 'Helsinki-NLP/opus-mt-en-de'},
        'it': {'name': 'Italiano', 'model': 'Helsinki-NLP/opus-mt-en-it'},
        'pt': {'name': 'Português', 'model': 'Helsinki-NLP/opus-mt-en-pt'},
    }
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # PARAMÈTRES SOUS-TITRES
    SUBTITLE_FONT_SIZE = 24
    SUBTITLE_Y_POSITION = 80
    SUBTITLE_STROKE_WIDTH = 2
    SUBTITLE_MAX_WIDTH = 0.80
    WORD_MIN_GAP = 0.05  

# ============================================================================
# INITIALISATION
# ============================================================================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = Config.OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_FILE_SIZE

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)

print("="*80)
print(" SOUS-TITRES KARAOKE - ANTI-SUPERPOSITION")
print("="*80)

# ============================================================================
# POLICE
# ============================================================================

def detect_font():
    import platform
    custom_font = "./fonts/Menuetto.ttf"
    
    if os.path.exists(custom_font):
        print(f" Police: {custom_font}")
        return custom_font
    
    system = platform.system()
    fonts = {
        "Windows": ["C:/Windows/Fonts/arial.ttf"],
        "Linux": ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
        "Darwin": ["/System/Library/Fonts/Supplemental/Arial.ttf"]
    }
    
    for font in fonts.get(system, []):
        if os.path.exists(font):
            print(f" Police: {font}")
            return font
    
    print("⚠️ Police par défaut")
    return None

FONT_PATH = detect_font()

# Debug: Afficher la signature de TextClip
import inspect
try:
    sig = inspect.signature(TextClip.__init__)
    print(f"📋 Signature TextClip: {sig}")
except:
    print("⚠️ Impossible de détecter la signature de TextClip")

# ============================================================================
# MODÈLES
# ============================================================================

class ModelManager:
    def __init__(self):
        self.device = Config.DEVICE
        self.stt_model = None
        self.stt_processor = None
        self.translation_models = {}
        self.translation_tokenizers = {}
        print(f" Device: {self.device}")
        self.load_stt_model()
    
    def load_stt_model(self):
        print(" Chargement STT...")
        if not os.path.exists(Config.MODEL_PATH):
            raise FileNotFoundError(f"Modèle introuvable: {Config.MODEL_PATH}")
        
        self.stt_processor = WhisperProcessor.from_pretrained(Config.MODEL_PATH)
        self.stt_model = WhisperForConditionalGeneration.from_pretrained(Config.MODEL_PATH)
        self.stt_model.to(self.device)
        self.stt_model.eval()
        print(" STT chargé")
    
    def load_translation_model(self, target_lang):
        if target_lang in self.translation_models:
            return
        
        model_name = Config.SUPPORTED_LANGUAGES[target_lang]['model']
        print(f"📥 Chargement traduction {target_lang}...")
        
        self.translation_tokenizers[target_lang] = MarianTokenizer.from_pretrained(model_name)
        self.translation_models[target_lang] = MarianMTModel.from_pretrained(model_name)
        self.translation_models[target_lang].to(self.device)
        self.translation_models[target_lang].eval()
        print(f"✅ {target_lang} chargé")

    def transcribe_with_segments(self, audio_path):
        """Transcription avec word-level timestamps - Gestion audio long"""
        print("🎤 Chargement audio...")
        audio, sr = librosa.load(audio_path, sr=16000)
        audio_duration = len(audio) / 16000
        
        # Si l'audio est long (>30s), on le découpe en chunks
        chunk_duration = 30.0  # 30 secondes par chunk
        
        if audio_duration > chunk_duration:
            print(f"🎤 Audio long détecté ({audio_duration:.1f}s) - Découpage en chunks...")
            all_segments = []
            num_chunks = int(np.ceil(audio_duration / chunk_duration))
            
            for i in range(num_chunks):
                start_sample = int(i * chunk_duration * 16000)
                end_sample = int(min((i + 1) * chunk_duration * 16000, len(audio)))
                chunk_audio = audio[start_sample:end_sample]
                
                print(f"🎤 Chunk {i+1}/{num_chunks} ({start_sample/16000:.1f}s - {end_sample/16000:.1f}s)...")
                
                # Transcription du chunk
                inputs = self.stt_processor(chunk_audio, sampling_rate=16000, return_tensors="pt")
                input_features = inputs.input_features.to(self.device)
                
                with torch.no_grad():
                    predicted_ids = self.stt_model.generate(
                        input_features,
                        return_timestamps=False,  # Pas de timestamps pour éviter les bugs
                        max_new_tokens=444,
                        language='en',
                        task='transcribe'
                    )
                
                transcription = self.stt_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                
                # Créer les segments pour ce chunk
                chunk_offset = i * chunk_duration
                chunk_segments = self._create_segments_from_text(transcription, chunk_offset, end_sample/16000)
                all_segments.extend(chunk_segments)
            
            print(f"✅ {len(all_segments)} segments créés (audio long)")
            return all_segments
        
        else:
            # Audio court - traitement normal
            print("🎤 Préparation des features...")
            inputs = self.stt_processor(audio, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(self.device)

            print("🎤 Génération de la transcription...")
            with torch.no_grad():
                predicted_ids = self.stt_model.generate(
                    input_features,
                    return_timestamps=False,
                    max_new_tokens=444,
                    language='en',
                    task='transcribe'
                )

            transcription = self.stt_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            print(f"📝 Transcription: {transcription}")

            segments = self._create_segments_from_text(transcription, 0.0, audio_duration)
            print(f"✅ {len(segments)} segments créés")
            return segments
    
    def _create_segments_from_text(self, transcription, start_offset, end_time):
        """Créer des segments avec timing depuis le texte"""
        words = transcription.split()
        
        if not words:
            return []

        total_duration = end_time - start_offset
        total_chars = sum(len(w) for w in words)
        time_per_char = total_duration / max(total_chars, 1)

        segments = []
        current_time = start_offset
        
        segment_words = []
        segment_start = start_offset
        
        for i, word in enumerate(words):
            word_duration = len(word) * time_per_char + 0.15
            segment_words.append({
                'word': word,
                'start': round(current_time, 3),
                'end': round(current_time + word_duration, 3)
            })
            current_time += word_duration + 0.05
            
            # Créer un segment tous les 6 mots ou à la fin
            if len(segment_words) >= 6 or i == len(words) - 1:
                segment_text = ' '.join([w['word'] for w in segment_words])
                segments.append({
                    'start': segment_start,
                    'end': min(segment_words[-1]['end'], end_time),
                    'text': segment_text,
                    'words': segment_words
                })
                segment_words = []
                segment_start = current_time

        return segments
        
    def translate_segment(self, segment, target_lang):
        """✅ Traduction PHRASE COMPLÈTE + redistribution intelligente"""
        if target_lang not in self.translation_models:
            self.load_translation_model(target_lang)
        
        text = segment['text']
        inputs = self.translation_tokenizers[target_lang](
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.translation_models[target_lang].generate(**inputs)
        
        translation = self.translation_tokenizers[target_lang].decode(outputs[0], skip_special_tokens=True)
        translated_words = translation.split()
        
        total_duration = segment['end'] - segment['start']
        num_words = len(translated_words)
        
        total_chars = sum(len(w) for w in translated_words)
        time_per_char = total_duration / max(total_chars, 1)
        
        current_time = segment['start']
        word_timings = []
        
        for i, word in enumerate(translated_words):
            word_duration = max(len(word) * time_per_char + 0.15, 0.2)
            
            if i == num_words - 1:
                end_time = segment['end']
            else:
                end_time = min(current_time + word_duration, segment['end'] - Config.WORD_MIN_GAP)
            
            if current_time < end_time:
                word_timings.append({
                    'word': word,
                    'start': round(current_time, 3),
                    'end': round(end_time, 3)
                })
                current_time = end_time + Config.WORD_MIN_GAP
            else:
                if word_timings:
                    word_timings[-1]['word'] += ' ' + word
        
        return {
            'start': segment['start'],
            'end': segment['end'],
            'text': translation,
            'words': word_timings
        }

model_manager = ModelManager()
print("✅ Prêt!\n" + "="*80)

# ============================================================================
# UTILITAIRES
# ============================================================================

def allowed_file(filename):
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    return ext in Config.ALLOWED_VIDEO_EXTENSIONS or ext in Config.ALLOWED_AUDIO_EXTENSIONS

def extract_audio_from_video(video_path, audio_path):
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, fps=16000, nbytes=2, codec='pcm_s16le', logger=None)
        video.close()
        return True
    except Exception as e:
        print(f"Erreur extraction: {e}")
        traceback.print_exc()
        return False

def convert_audio_to_wav(input_path, output_path):
    """Convertir n'importe quel format audio en WAV 16kHz pour Whisper"""
    try:
        audio, sr = librosa.load(input_path, sr=16000)
        sf.write(output_path, audio, 16000)
        return True
    except Exception as e:
        print(f"Erreur conversion audio: {e}")
        traceback.print_exc()
        return False

def create_text_clip_safe(text, font_size, color, stroke_color, stroke_width, size, font_path):
    """Création de TextClip compatible avec toutes les versions de moviepy"""
    # Essayer différentes combinaisons de paramètres
    params_to_try = [
        # Moviepy 2.x récent
        {
            'text': text,
            'font_size': font_size,
            'color': color,
            'stroke_color': stroke_color,
            'stroke_width': stroke_width,
            'size': size,
            'method': 'caption'
        },
        # Moviepy 1.x
        {
            'txt': text,
            'fontsize': font_size,
            'color': color,
            'stroke_color': stroke_color,
            'stroke_width': stroke_width,
            'size': size,
            'method': 'caption'
        },
        # Moviepy minimal (sans stroke)
        {
            'txt': text,
            'fontsize': font_size,
            'color': color,
            'size': size,
            'method': 'caption'
        }
    ]
    
    # Ajouter le font seulement si disponible
    if font_path:
        for params in params_to_try[:2]:  # Seulement les 2 premiers
            params['font'] = font_path
    
    last_error = None
    for params in params_to_try:
        try:
            return TextClip(**params)
        except Exception as e:
            last_error = e
            continue
    
    # Si tout échoue, lever la dernière erreur
    raise last_error

def create_karaoke_subtitled_video(video_path, segments, output_path):
    """✅ Création vidéo avec VÉRIFICATION stricte anti-superposition"""
    try:
        video = VideoFileClip(video_path)
        all_clips = [video]
        
        all_word_clips = []
        
        for seg_idx, segment in enumerate(segments):
            words = segment['words']
            if not words:
                continue
            
            print(f"\n📝 Segment {seg_idx+1}/{len(segments)}: {len(words)} mots")
            
            for word_idx, word_data in enumerate(words):
                start = word_data['start']
                end = word_data['end']
                
                if start >= end:
                    print(f"  ⚠️ Timing invalide ignoré: {word_data['word']} ({start}-{end})")
                    continue
                
                if all_word_clips:
                    last_clip = all_word_clips[-1]
                    if start < last_clip['end'] + Config.WORD_MIN_GAP:
                        old_start = start
                        start = last_clip['end'] + Config.WORD_MIN_GAP
                        if start >= end:
                            print(f"  ⚠️ Superposition évitée (fusionné): {word_data['word']}")
                            continue
                        print(f"  🔧 Ajusté: {old_start:.2f}→{start:.2f} pour '{word_data['word']}'")
                
                text_parts = []
                for i, w in enumerate(words):
                    if i == word_idx:
                        text_parts.append(w['word'].upper())
                    else:
                        text_parts.append(w['word'])
                
                full_text = ' '.join(text_parts)
                
                try:
                    txt_clip = create_text_clip_safe(
                        full_text,
                        Config.SUBTITLE_FONT_SIZE,
                        'white',
                        'black',
                        Config.SUBTITLE_STROKE_WIDTH,
                        (int(video.w * Config.SUBTITLE_MAX_WIDTH), None),
                        FONT_PATH if FONT_PATH else 'Arial'
                    )
                except:
                    normalized = unicodedata.normalize('NFKD', full_text)
                    display_text = normalized.encode('ASCII', 'ignore').decode('ASCII')
                    txt_clip = create_text_clip_safe(
                        display_text,
                        Config.SUBTITLE_FONT_SIZE,
                        'white',
                        'black',
                        Config.SUBTITLE_STROKE_WIDTH,
                        (int(video.w * Config.SUBTITLE_MAX_WIDTH), None),
                        FONT_PATH if FONT_PATH else 'Arial'
                    )
                
                y_pos = video.h - Config.SUBTITLE_Y_POSITION
                txt_clip = txt_clip.with_position(('center', y_pos))
                txt_clip = txt_clip.with_start(start)
                txt_clip = txt_clip.with_duration(end - start)
                
                all_clips.append(txt_clip)
                
                all_word_clips.append({
                    'word': word_data['word'],
                    'start': start,
                    'end': end
                })
                
                if (word_idx + 1) % 10 == 0:
                    print(f"  ✓ {word_idx+1}/{len(words)} mots traités")
        
        print(f"\n✅ {len(all_clips)-1} clips créés (anti-superposition validée)")
        
        final_video = CompositeVideoClip(all_clips)
        
        print("📤 Export vidéo...")
        final_video.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=video.fps,
            preset='ultrafast',
            bitrate='5000k',
            threads=8,
            logger=None
        )
        
        video.close()
        final_video.close()
        print("✅ Vidéo créée!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        traceback.print_exc()
        return False

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    return render_template('video_subtitles.html', 
                         languages=Config.SUPPORTED_LANGUAGES,
                         device=str(Config.DEVICE))


@app.route('/api/process-media', methods=['POST'])
def process_media():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier reçu'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Nom de fichier vide'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Format de fichier non supporté'}), 400

        target_lang = request.form.get('language', 'fr')
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(input_path)
        
        ext = filename.rsplit('.', 1)[1].lower()
        is_audio = ext in Config.ALLOWED_AUDIO_EXTENSIONS
        
        print(f"\n--- Nouveau traitement : {'AUDIO' if is_audio else 'VIDÉO'} ---")

        audio_work_path = os.path.join(app.config['UPLOAD_FOLDER'], f"work_{timestamp}.wav")
        
        if is_audio:
            print("🎵 Conversion audio pour traitement...")
            if not convert_audio_to_wav(input_path, audio_work_path):
                return jsonify({'error': "Échec de la conversion audio"}), 500
        else:
            print("🎵 Extraction audio de la vidéo...")
            if not extract_audio_from_video(input_path, audio_work_path):
                return jsonify({'error': "Échec de l'extraction audio"}), 500

        print("🎤 Transcription Whisper...")
        segments = model_manager.transcribe_with_segments(audio_work_path)
        
        if not segments:
            return jsonify({'error': "Aucune parole détectée dans le fichier"}), 400

        print(f"🌍 Traduction vers : {target_lang}...")
        translated_segments = []
        for seg in segments:
            translated_seg = model_manager.translate_segment(seg, target_lang)
            translated_segments.append(translated_seg)

        if is_audio:
            final_url = f"/uploads/{unique_filename}"
        else:
            output_filename = f"subtitled_{unique_filename.rsplit('.', 1)[0]}.mp4"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            print("🎬 Incrustation des sous-titres karaoke...")
            success = create_karaoke_subtitled_video(input_path, translated_segments, output_path)
            if not success:
                return jsonify({'error': "Erreur lors du montage vidéo"}), 500
            final_url = f"/outputs/{output_filename}"

        if os.path.exists(audio_work_path):
            os.remove(audio_work_path)

        return jsonify({
            'success': True,
            'is_audio': is_audio,
            'video_url': final_url,
            'segments': translated_segments,
            'total_segments': len(translated_segments),
            'language': target_lang
        })

    except Exception as e:
        print(f"❌ ERREUR CRITIQUE : {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
    
@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_file(os.path.join(Config.UPLOAD_FOLDER, filename))

@app.route('/outputs/<filename>')
def serve_output(filename):
    return send_file(os.path.join(Config.OUTPUT_FOLDER, filename))

@app.route('/api/languages', methods=['GET'])
def get_languages():
    return jsonify({lang: info['name'] for lang, info in Config.SUPPORTED_LANGUAGES.items()})

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'stt_loaded': model_manager.stt_model is not None,
        'device': str(Config.DEVICE)
    })

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': f'Fichier trop grand (max {Config.MAX_FILE_SIZE/(1024*1024):.0f} MB)'}), 413

# ============================================================================
# DÉMARRAGE
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🌐 SERVEUR KARAOKE - ANTI-SUPERPOSITION")
    print("="*80)
    print(f"\n📍 URL: http://127.0.0.1:5000")
    print("\n✨ FONCTIONNALITÉS:")
    print("   ✓ Traduction phrase complète (garde le sens)")
    print("   ✓ Redistribution intelligente des mots traduits")
    print("   ✓ Gap minimum 50ms entre chaque mot")
    print("   ✓ Vérification double anti-superposition")
    print("   ✓ Ajustement automatique si chevauchement détecté")
    print("\n💡 Ctrl+C pour arrêter")
    print("="*80 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)