@echo off
chcp 65001 >nul
echo ================================================================================
echo 🎬 INSTALLATION - VIDÉO + SOUS-TITRES TRADUITS
echo ================================================================================
echo.

:: Vérifier Python
echo 📌 Vérification de Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python n'est pas installé ou pas dans le PATH
    echo 💡 Installez Python depuis https://www.python.org/downloads/
    pause
    exit /b 1
)
python --version
echo ✅ Python détecté
echo.

:: Vérifier ImageMagick
echo 📌 Vérification de ImageMagick...
magick --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ImageMagick n'est pas installé
    echo.
    echo 💡 INSTALLATION REQUISE:
    echo    1. Téléchargez: https://imagemagick.org/script/download.php#windows
    echo    2. Installez avec "Add to PATH" + "Install legacy utilities"
    echo    3. Redémarrez votre ordinateur
    echo    4. Relancez ce script
    echo.
    pause
    exit /b 1
)
echo ✅ ImageMagick détecté
echo.

:: Créer les dossiers
echo 📁 Création de la structure des dossiers...
if not exist "templates" mkdir templates
if not exist "models" mkdir models
if not exist "models\whisper-trained" mkdir models\whisper-trained
if not exist "uploads" mkdir uploads
if not exist "outputs" mkdir outputs
echo ✅ Dossiers créés
echo.

:: Installer les dépendances
echo 📦 Installation des dépendances Python...
echo    Cela peut prendre plusieurs minutes...
echo.

pip install --upgrade pip
pip install flask
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install librosa
pip install soundfile
pip install werkzeug
pip install moviepy
pip install imageio-ffmpeg

if %errorlevel% neq 0 (
    echo.
    echo ⚠️ Certaines dépendances ont échoué
    echo 💡 Essayez d'installer manuellement:
    echo    pip install flask torch transformers librosa soundfile werkzeug moviepy
    pause
    exit /b 1
)

echo.
echo ✅ Toutes les dépendances sont installées
echo.

:: Vérifier le modèle
echo 📌 Vérification du modèle Whisper...
if not exist "models\whisper-trained\config.json" (
    echo.
    echo ⚠️ Modèle Whisper non trouvé
    echo 💡 Placez votre modèle fine-tuné dans: models\whisper-trained\
    echo.
) else (
    echo ✅ Modèle détecté
)

echo.
echo ================================================================================
echo ✨ INSTALLATION TERMINÉE
echo ================================================================================
echo.
echo 📝 Prochaines étapes:
echo    1. Placez video_subtitles.html dans le dossier templates\
echo    2. Assurez-vous que le modèle est dans models\whisper-trained\
echo    3. Lancez: python app_video_subtitles.py
echo    4. Ouvrez: http://127.0.0.1:5000
echo.
echo 💡 Astuce: Créez un raccourci avec start.bat pour lancer facilement
echo.
pause