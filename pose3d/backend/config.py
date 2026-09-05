"""
Configurações do projeto.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
TMP_IMAGES_DIR = DATA_DIR / "tmp_images"
DB_PATH = DATA_DIR / "poses.db"

# Pares (imagem + pose bruta do MoveNet + pose corrigida por você)
# usados como dataset de treino da rede corretora. Diferente de
# TMP_IMAGES_DIR, essa pasta nunca é limpa automaticamente.
TRAINING_PAIRS_DIR = DATA_DIR / "training_pairs"

FRONTEND_DIR = BASE_DIR / "frontend"

# Onde o modelo corretor treinado (já convertido pra TFJS) é
# servido pro navegador. app.js carrega de /js/models/corrector/.
CORRECTOR_MODEL_DIR = FRONTEND_DIR / "js" / "models" / "corrector"

# Cria as pastas necessárias
DATA_DIR.mkdir(parents=True, exist_ok=True)
TMP_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_PAIRS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# DANBOORU
# ============================================================

DANBOORU_BASE_URL = "https://safebooru.donmai.us"

DANBOORU_LOGIN = os.getenv("DANBOORU_LOGIN", "")
DANBOORU_API_KEY = os.getenv("DANBOORU_API_KEY", "")

DANBOORU_REQUEST_INTERVAL_SECONDS = 1.0
DANBOORU_PAGE_LIMIT = 100

DEFAULT_SEARCH_TAGS = [
    "1girl",
    "solo",
    "standing",
    "1man"
]

# ============================================================
# POSE
# ============================================================

POSE_CONFIDENCE_THRESHOLD = 0.6
MIN_KEYPOINTS_DETECTED = 10

# Modelo YOLO Pose
POSE_MODEL = "yolo11n-pose.pt"

# ============================================================
# REDE CORRETORA (active learning)
# ============================================================
# O MoveNet (rodando no navegador) foi treinado em fotos reais, não
# em desenhos — erra sistematicamente em poses de personagens. Em vez
# de reentreinar o MoveNet (modelo grande, pouco dado disponível),
# treinamos uma rede pequena que aprende o "delta" entre o que o
# MoveNet chuta e o que você corrige na revisão manual. Essa ordem
# tem que ser idêntica à lista COCO_KEYPOINTS em frontend/js/app.js —
# é o índice nessa lista que liga uma posição do vetor de entrada/
# saída ao nome da articulação.
POSE_KEYPOINT_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# Abaixo desse número de pares no dataset, nem tenta treinar —
# a rede só ia decorar ruído.
CORRECTOR_MIN_TRAINING_SAMPLES = 20

# ============================================================
# SERVIDOR
# ============================================================

HOST = "127.0.0.1"
PORT = 5000
DEBUG = True
