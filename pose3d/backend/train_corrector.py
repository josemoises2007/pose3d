"""
train_corrector.py

Treina a rede corretora do pipeline de active learning e exporta
pro formato que o navegador consegue carregar (TensorFlow.js).

Roda separado do Flask — não é chamado pela API, é pra você
executar manualmente sempre que quiser retreinar com as correções
novas:

    python train_corrector.py

O que a rede faz: recebe a pose que o MoveNet chutou (17 pontos,
cada um com x, y, score) e aprende o delta (dx, dy) que leva até a
pose que você aprovou/corrigiu na revisão. Ou seja, ela não aprende
pose do zero — aprende a "puxar" o chute do MoveNet pra mais perto
do que costuma estar certo em desenhos/personagens.

Dependências (não fazem parte do requirements do backend Flask,
são só pra treino):

    pip install tensorflow tensorflowjs pillow numpy --break-system-packages

Se dessa instalação sobrar um erro tipo
"AttributeError: 'MessageFactory' object has no attribute
'GetPrototype'" na hora de exportar, é conflito de versão do
protobuf entre tensorflow e tensorflowjs (dependência transitiva
de ambos). Resolve com:

    pip install "protobuf>=6.31.1" --break-system-packages
"""

import json
import sys

import numpy as np
from PIL import Image

import config

try:
    import tensorflow as tf
except ImportError:
    sys.exit(
        "TensorFlow não instalado. Rode:\n"
        "  pip install tensorflow tensorflowjs pillow numpy --break-system-packages"
    )

KEYPOINT_NAMES = config.POSE_KEYPOINT_NAMES
N_KEYPOINTS = len(KEYPOINT_NAMES)
INPUT_DIM = N_KEYPOINTS * 3  # (x_norm, y_norm, score) por ponto
OUTPUT_DIM = N_KEYPOINTS * 2  # (dx_norm, dy_norm) por ponto

EPOCHS = 300
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.15

# ============================================================
# DATASET
# ============================================================


def _load_pairs():
    """
    Lê todos os *.json em TRAINING_PAIRS_DIR e monta os arrays de
    treino. Cada par vira um exemplo (X, Y, mask):

      X    = pose bruta normalizada, achatada
      Y    = delta normalizado (final - bruto), achatado
      mask = 1 onde dá pra comparar bruto x final (mesmo ponto
             presente nos dois), 0 caso contrário
    """

    pairs_dir = config.TRAINING_PAIRS_DIR

    X, Y, M = [], [], []

    for sidecar in sorted(pairs_dir.glob("*.json")):
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        keypoints_raw = data.get("keypoints_raw") or {}
        keypoints_final = data.get("keypoints_final") or {}

        if not keypoints_raw:
            continue

        image_path = pairs_dir / data["image_file"]

        if not image_path.exists():
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        x_vec = np.zeros(INPUT_DIM, dtype=np.float32)
        y_vec = np.zeros(OUTPUT_DIM, dtype=np.float32)
        m_vec = np.zeros(OUTPUT_DIM, dtype=np.float32)

        for i, name in enumerate(KEYPOINT_NAMES):
            p_raw = keypoints_raw.get(name)

            if p_raw is None:
                continue

            x_norm = p_raw["x"] / width
            y_norm = p_raw["y"] / height
            score = p_raw.get("score", 0.0)

            x_vec[i * 3: i * 3 + 3] = [x_norm, y_norm, score]

            p_final = keypoints_final.get(name)

            if p_final is None:
                # Você removeu esse ponto na revisão — não dá pra
                # treinar "pra onde ele deveria ir", então fica de
                # fora do cálculo do delta.
                continue

            fx_norm = p_final["x"] / width
            fy_norm = p_final["y"] / height

            y_vec[i * 2: i * 2 + 2] = [
                fx_norm - x_norm,
                fy_norm - y_norm,
            ]
            m_vec[i * 2: i * 2 + 2] = [1.0, 1.0]

        # Sem nenhum ponto comparável, esse par não ensina nada
        if m_vec.sum() == 0:
            continue

        X.append(x_vec)
        Y.append(y_vec)
        M.append(m_vec)

    return (
        np.array(X, dtype=np.float32),
        np.array(Y, dtype=np.float32),
        np.array(M, dtype=np.float32),
    )

# ============================================================
# MODELO
# ============================================================


def _build_model():
    inputs = tf.keras.Input(shape=(INPUT_DIM,), name="raw_pose")

    x = tf.keras.layers.Dense(128, activation="relu")(inputs)
    x = tf.keras.layers.Dense(128, activation="relu")(x)

    # Inicialização quase-zero na última camada: no começo do
    # treino a rede prevê delta ~0 (ou seja, "não mexe em nada"),
    # e só se afasta disso onde os dados mostram que vale a pena
    # corrigir. Evita que ela invente correções malucas com pouco
    # dado.
    outputs = tf.keras.layers.Dense(
        OUTPUT_DIM,
        activation="linear",
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-3),
        bias_initializer="zeros",
        name="delta",
    )(x)

    return tf.keras.Model(inputs, outputs, name="pose_corrector")


def _masked_mse(y_true, y_pred, mask):
    diff = (y_pred - y_true) * mask
    denom = tf.maximum(tf.reduce_sum(mask), 1.0)
    return tf.reduce_sum(tf.square(diff)) / denom


def _train(model, X_train, Y_train, M_train, X_val, Y_val, M_val):
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        with tf.GradientTape() as tape:
            pred = model(X_train, training=True)
            loss = _masked_mse(Y_train, pred, M_train)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if epoch % 50 == 0 or epoch == EPOCHS:
            val_pred = model(X_val, training=False)
            val_loss = _masked_mse(Y_val, val_pred, M_val)
            print(
                f"epoch {epoch:4d}  "
                f"train_loss={float(loss):.5f}  "
                f"val_loss={float(val_loss):.5f}"
            )

# ============================================================
# MAIN
# ============================================================


def main():
    X, Y, M = _load_pairs()
    n_samples = len(X)

    print(f"{n_samples} pares de treino encontrados em {config.TRAINING_PAIRS_DIR}")

    if n_samples < config.CORRECTOR_MIN_TRAINING_SAMPLES:
        print(
            f"Menos que {config.CORRECTOR_MIN_TRAINING_SAMPLES} pares — "
            "ainda não compensa treinar (a rede só ia decorar ruído). "
            "Continue revisando poses em /review.html e rode de novo depois."
        )
        return

    # Split treino/validação
    rng = np.random.default_rng(seed=42)
    idx = rng.permutation(n_samples)

    n_val = max(1, int(n_samples * VAL_FRACTION))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train, Y_train, M_train = X[train_idx], Y[train_idx], M[train_idx]
    X_val, Y_val, M_val = X[val_idx], Y[val_idx], M[val_idx]

    print(f"treino: {len(X_train)}  validação: {len(X_val)}")

    model = _build_model()
    _train(model, X_train, Y_train, M_train, X_val, Y_val, M_val)

    # ------------------------------------------------------------
    # Exporta pro formato que tf.js consegue carregar no navegador
    # (frontend/js/app.js faz tf.loadLayersModel nesse caminho)
    # ------------------------------------------------------------
    try:
        import tensorflowjs as tfjs
    except ImportError:
        sys.exit(
            "Treino ok, mas falta 'tensorflowjs' pra exportar. Rode:\n"
            "  pip install tensorflowjs --break-system-packages\n"
            "e execute o script de novo."
        )

    output_dir = config.CORRECTOR_MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    tfjs.converters.save_keras_model(model, str(output_dir))

    print(f"Modelo exportado pra {output_dir}")
    print("Recarregue /review.html — o app.js carrega o modelo automaticamente.")


if __name__ == "__main__":
    main()
