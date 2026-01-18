#  STT Translation App - Speech-to-Text & Traduction Multilingue

Application Flask complète pour la **transcription audio (Speech-to-Text)**, la **traduction multilingue** et l'**incrustation de sous-titres karaoke** sur vidéos.

##  Fonctionnalités

### Transcription Audio
- ✅ Transcription précise basée sur **Whisper fine-tuné**
- ✅ Support de formats multiples : WAV, MP3, FLAC, OGG, M4A, AAC
- ✅ Gestion des fichiers audio longs (découpage automatique en chunks)
- ✅ Détection automatique du langage

### Traduction Multilingue
- 🌍 Support de **6 langues** : 
  - 🇫🇷 Français (fr)
  - 🇪🇸 Espagnol (es)
  - 🇸🇦 Arabe (ar)
  - 🇩🇪 Allemand (de)
  - 🇮🇹 Italien (it)
  - 🇵🇹 Portugais (pt)
- ✅ Traduction phrase complète (préserve le sens)
- ✅ Utilise les modèles **MarianMT** de Helsinki-NLP

### Sous-titres Karaoke
- ✅ Incrustation de sous-titres sur vidéo
- ✅ **Anti-superposition GARANTIE**
- ✅ Redistribution intelligente des mots traduits
- ✅ Gap minimum de 50ms entre chaque mot
- ✅ Vérification double pour éviter les chevauchements
- ✅ Mise en surbrillance du mot actuel (majuscules)

### Support Vidéo
- Formats supportés : MP4, AVI, MOV, MKV, WEBM
- Extraction audio automatique
- Traitement parallèle audio + traduction
- Export vidéo optimisé avec sous-titres

##  Installation

### Prérequis
- **Python 3.9+**
- CUDA (optionnel, pour GPU accéléré)
- Git

### 1. Cloner le repository
```bash
git clone https://github.com/HLG-1/Projet_traitement_audio.git
cd stt-translation-app
```

### 2. Créer un environnement virtuel
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Télécharger le modèle Whisper
Le modèle fine-tuné doit être placé dans : `./models/whisper-trained/`

Vous pouvez télécharger le modèle depuis le projet Colab associé.

### 5. Structure des dossiers
```
stt-translation-app/
├── models/
│   └── whisper-trained/        # Modèle Whisper fine-tuné
├── uploads/                     # Fichiers temporaires
├── outputs/                     # Vidéos générées
├── fonts/                       # Polices personnalisées
├── static/
│   ├── css/
│   └── js/
├── templates/                   # Pages HTML
├── app.py                       # Application principale (vidéo)
├── app2.py                      # Variante
├── app3.py                      # Version simplifiée
└── requirements.txt
```

## 💻 Utilisation

### Démarrer l'application

#### Version Karaoke (avec sous-titres vidéo)
```bash
python app.py
```
URL : http://localhost:5000

#### Version Audio Simple
```bash
python app3.py
```
URL : http://localhost:5000

### Utilisation Web

1. **Accédez** à http://localhost:5000
2. **Téléchargez** un fichier audio ou vidéo
3. **Sélectionnez** la langue cible
4. **Cliquez** sur "Traiter"
5. **Téléchargez** les résultats

### Utilisation API

#### Transcription + Traduction
```bash
curl -X POST http://localhost:5000/api/transcribe \
  -F "file=@audio.wav" \
  -F "languages[]=fr" \
  -F "languages[]=es"
```

**Réponse :**
```json
{
  "success": true,
  "transcription": "Hello world",
  "translations": {
    "fr": "Bonjour le monde",
    "es": "Hola mundo"
  },
  "processing_time": 12.34,
  "target_languages": ["fr", "es"]
}
```

#### Vérification de l'état
```bash
curl http://localhost:5000/api/health
```

#### Langues supportées
```bash
curl http://localhost:5000/api/languages
```

## ⚙️ Configuration

Modifiez les paramètres dans la classe `Config` dans `app.py`:

```python
class Config:
    MODEL_PATH = "./models/whisper-trained"
    UPLOAD_FOLDER = "./uploads"
    OUTPUT_FOLDER = "./outputs"
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
    
    # Paramètres sous-titres
    SUBTITLE_FONT_SIZE = 24
    SUBTITLE_Y_POSITION = 80
    SUBTITLE_STROKE_WIDTH = 2
    WORD_MIN_GAP = 0.05  # Gap minimum entre mots (secondes)
```

## 📊 Architecture

### Flux de traitement
```
Fichier d'entrée
    ↓
[Validation] → Vérifier format et taille
    ↓
[Extraction audio] → Convertir en WAV 16kHz
    ↓
[Transcription] → Whisper fine-tuné
    ↓
[Traduction] → MarianMT (langue cible)
    ↓
[Timing] → Créer segments avec timings
    ↓
[Sous-titres] → Incruster dans vidéo (si vidéo)
    ↓
Résultat final
```

### Modèles utilisés

| Composant | Modèle | Souces |
|-----------|--------|--------|
| STT | Whisper fine-tuné | OpenAI + Fine-tuning |
| Traduction | MarianMT | Helsinki-NLP (Hugging Face) |
| Détection police | Système | Windows/Linux/MacOS |

##  Cas d'usage

-  **Créateur vidéo** : Ajouter des sous-titres traduits automatiquement
-  **Éducation** : Transcrire et traduire des cours
-  **Podcasters** : Générer sous-titres multilingues
-  **Contenu multilingue** : Servir une audience globale
-  **Accessibilité** : Rendre le contenu accessible

##  Performance

| Opération | Temps (secondes) |
|-----------|-----------------|
| Chargement modèles | ~10-15s |
| Transcription (1 min audio) | ~2-5s |
| Traduction | ~1-2s |
| Incrustation sous-titres | ~30-60s (selon durée vidéo) |

*Temps avec GPU CUDA. Sans GPU, compter 2-3x plus.*

##  Limitations connues

- Taille maximale fichier : 100 MB
- Durée audio maximale : Pas de limite technique (traitement par chunks)
- Polices : Dépend du système (Arial par défaut)
- Format vidéo : Nécessite FFmpeg installé

##  Dépannage

### Erreur : "Modèle introuvable"
``` Modèle introuvable : ./models/whisper-trained/
```
**Solution :** Téléchargez et placez le modèle dans `models/whisper-trained/`

### Erreur : "moviepy non installé"
```bash
pip install moviepy==1.0.3
pip install --upgrade imageio-ffmpeg
```

### Erreur : "Fichier trop grand"
Augmentez `MAX_FILE_SIZE` dans Config ou utilisez un fichier plus petit.

### GPU non reconnu
```bash
pip install torch torchcuda
```
Vérifiez avec : `python -c "import torch; print(torch.cuda.is_available())"`

##  Routes disponibles

| Route | Méthode | Description |
|-------|---------|-------------|
| `/` | GET | Page d'accueil |
| `/api/transcribe` | POST | Transcription + traduction |
| `/api/languages` | GET | Liste des langues |
| `/api/health` | GET | État de l'API |
| `/api/info` | GET | Informations API |
| `/uploads/<filename>` | GET | Télécharger fichier |
| `/outputs/<filename>` | GET | Télécharger résultat |

## 🔧 Développement

### Structure du code

- **`app.py`** : Version complète avec sous-titres karaoke
- **`app2.py`** : Variante (même fonctionnalités)
- **`app3.py`** : Version simplifiée (audio uniquement)
- **`check_textclip.py`** : Utilitaire debug moviepy

### Dépendances principales

```
Flask 3.0.0           # Framework web
torch / torchaudio    # Deep learning
transformers 4.40+    # Modèles NLP
librosa 0.10.1        # Traitement audio
moviepy 1.0.3         # Édition vidéo
soundfile 0.12.1      # Fichiers audio
```

## Licence

Ce projet est sous licence MIT. Voir `LICENSE` pour plus de détails.

##  Auteurs

- **Hajar EL HALLAGUE** - Développement principal



### Traiter une vidéo avec sous-titres
```bash
# Via l'interface web : 
# 1. Aller à http://localhost:5000
# 2. Uploader ma_video.mp4
# 3. Choisir langue
# 4. Attendre le traitement
# 5. Télécharger la vidéo avec sous-titres
```
