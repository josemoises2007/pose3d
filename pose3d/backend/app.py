import json
import shutil
import uuid
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    request,
    send_file,
    send_from_directory
)

import config
import db

from collector import CollectionManager

try:
    from flask_cors import CORS
except ImportError:
    CORS = None

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = Flask(
    __name__,
    static_folder=None
)

collection_manager = CollectionManager()

if CORS is not None:
    CORS(app)


@app.route("/")
def serve_index():
    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


@app.route("/review.html")
def serve_review():
    return send_from_directory(
        FRONTEND_DIR,
        "review.html"
    )


@app.route("/js/<path:filename>")
def serve_js(filename):
    return send_from_directory(
        FRONTEND_DIR / "js",
        filename
    )


@app.route("/css/<path:filename>")
def serve_css(filename):
    return send_from_directory(
        FRONTEND_DIR / "css",
        filename
    )


@app.route("/api/test")
def api_test():
    return jsonify({
        "success": True,
        "message": "Backend funcionando"
    })


@app.route("/api/collect/start", methods=["POST"])
def collect_start():
    data = request.get_json(silent=True) or {}

    tags_raw = data.get("tags", "")

    if isinstance(tags_raw, str):
        tags = tags_raw.split()
    else:
        tags = list(tags_raw or [])

    if not tags:
        return jsonify({
            "success": False,
            "error": "Informe ao menos uma tag."
        }), 400

    try:
        limit = int(data.get("limit", 0) or 0)
    except (TypeError, ValueError):
        limit = 0

    started = collection_manager.start(tags, limit=limit)

    if not started:
        return jsonify({
            "success": False,
            "error": "Coleta já está em andamento."
        }), 409

    return jsonify({
        "success": True,
        "status": collection_manager.get_status()
    })


@app.route("/api/collect/pause", methods=["POST"])
def collect_pause():
    ok = collection_manager.pause()

    return jsonify({
        "success": ok,
        "status": collection_manager.get_status()
    })


@app.route("/api/collect/resume", methods=["POST"])
def collect_resume():
    ok = collection_manager.resume()

    return jsonify({
        "success": ok,
        "status": collection_manager.get_status()
    })


@app.route("/api/collect/stop", methods=["POST"])
def collect_stop():
    ok = collection_manager.stop()

    return jsonify({
        "success": ok,
        "status": collection_manager.get_status()
    })


@app.route("/api/collect/status")
def collect_status():
    return jsonify(collection_manager.get_status())


@app.route("/api/stats")
def api_stats():
    return jsonify(db.get_stats())


@app.route("/api/corrector/status")
def corrector_status():
    """
    Quantos pares de treino já foram acumulados em
    TRAINING_PAIRS_DIR, e se já existe um modelo corretor
    exportado pro frontend (frontend/js/models/corrector/).
    """

    pairs_dir = Path(config.TRAINING_PAIRS_DIR)
    n_pairs = len(list(pairs_dir.glob("*.json")))

    model_path = Path(config.CORRECTOR_MODEL_DIR) / "model.json"

    return jsonify({
        "training_pairs": n_pairs,
        "min_required": config.CORRECTOR_MIN_TRAINING_SAMPLES,
        "ready_to_train":
            n_pairs >= config.CORRECTOR_MIN_TRAINING_SAMPLES,
        "model_exported": model_path.exists(),
    })


@app.route(
    "/api/maintenance/cleanup-missing-images",
    methods=["POST"]
)
def cleanup_missing_images():
    """
    Varre os posts com status='downloaded' e marca como 'error' os
    que não têm mais arquivo de imagem em disco (ex: pasta
    tmp_images foi limpa manualmente pra liberar espaço). Sem isso,
    /api/review/next fica devolvendo pra sempre os mesmos posts
    quebrados.
    """

    downloaded_posts = db.get_posts_by_status("downloaded")

    fixed = 0

    for post in downloaded_posts:
        post_id = post["post_id"]

        if _find_local_image(post_id) is None:
            db.update_post_status(
                post_id,
                status="error",
                error_message=(
                    "arquivo de imagem não encontrado em disco (limpeza)"
                ),
            )
            fixed += 1

    return jsonify({
        "success": True,
        "checked": len(downloaded_posts),
        "fixed": fixed
    })


@app.route("/api/manual3d/save", methods=["POST"])
def manual3d_save():
    data = request.get_json(silent=True) or {}

    keypoints_3d = data.get("keypoints_3d")
    tags = data.get("tags") or []

    if not keypoints_3d or not isinstance(keypoints_3d, dict):
        return jsonify({
            "success": False,
            "error": "keypoints_3d obrigatório."
        }), 400

    pose_id = f"manual_{uuid.uuid4().hex[:12]}"

    try:
        db.insert_skeleton(
            pose_id=pose_id,
            source_post_id=None,
            keypoints_3d=keypoints_3d,
            confidence=1.0,
            review_status="manual_approved",
            tags=tags,
            verified=True,
        )

        return jsonify({
            "success": True,
            "pose_id": pose_id
        })

    except Exception as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 500


@app.route("/api/review/next")
def review_next():
    post = db.get_next_downloaded_post()

    if post is None:
        return jsonify({
            "finished": True,
            "post": None
        })

    post_id = post["post_id"]

    return jsonify({
        "finished": False,
        "post": {
            "post_id": post_id,
            "image_url":
                f"/api/review/image/{post_id}"
        }
    })


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".bmp"
}


def _find_local_image(post_id):
    """
    Procura o arquivo de imagem baixado pra um post_id em
    TMP_IMAGES_DIR. Retorna o Path ou None se não existir.
    """

    image_dir = Path(config.TMP_IMAGES_DIR)

    files = [
        path
        for path in image_dir.glob(f"{post_id}.*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return files[0] if files else None


def _save_training_pair(post_id, keypoints_raw, keypoints_final):
    """
    Guarda o par (imagem, pose bruta do MoveNet, pose corrigida por
    você) em config.TRAINING_PAIRS_DIR, antes da imagem ser apagada
    em _delete_local_image. Esse par é o dataset que
    train_corrector.py usa pra treinar a rede corretora.

    Só vale a pena guardar quando existe uma pose bruta pra comparar
    (keypoints_raw) — sem ela não tem "correção" pra aprender, só a
    pose final sozinha (esqueleto todo manual).

    Best-effort: falha aqui não deve derrubar a revisão.
    """

    if not keypoints_raw:
        return

    try:
        image_path = _find_local_image(post_id)

        if image_path is None:
            return

        pairs_dir = Path(config.TRAINING_PAIRS_DIR)

        dest_image = pairs_dir / f"{post_id}{image_path.suffix.lower()}"
        shutil.copyfile(image_path, dest_image)

        sidecar = pairs_dir / f"{post_id}.json"
        sidecar.write_text(
            json.dumps({
                "post_id": post_id,
                "image_file": dest_image.name,
                "keypoints_raw": keypoints_raw,
                "keypoints_final": keypoints_final,
            }),
            encoding="utf-8",
        )

    except OSError:
        pass


def _delete_local_image(post_id):
    """
    Apaga o arquivo local do post depois que ele já foi revisado
    (aprovado ou rejeitado). Uma vez extraídos os keypoints, a
    imagem crua não é mais necessária — e mantê-la pra sempre é o
    que costuma lotar a pasta data/tmp_images.

    Best-effort: falha aqui não deve derrubar a revisão.
    """

    try:
        image_path = _find_local_image(post_id)

        if image_path is not None:
            image_path.unlink()

    except OSError:
        pass


@app.route(
    "/api/review/image/<int:post_id>"
)
def review_image(post_id):
    image_path = _find_local_image(post_id)

    if image_path is None:
        # Post marcado como "downloaded" mas o arquivo sumiu do
        # disco (apagado manualmente, disco limpo, etc). Sem isso,
        # /api/review/next fica devolvendo pra sempre o mesmo post
        # quebrado, já que ele nunca sai do status "downloaded".
        db.update_post_status(
            post_id,
            status="error",
            error_message="arquivo de imagem não encontrado em disco",
        )

        return jsonify({
            "error":
                "Imagem não encontrada."
        }), 404

    return send_file(image_path)


def _naive_lift_to_3d(keypoints_2d):
    """
    Conversão mínima de 2D -> 3D (z=0), usada apenas como
    fallback quando o cliente não manda keypoints_3d prontos.

    Substitui o antigo build_skeleton_3d() (módulo skeleton.py,
    removido do projeto). Se no futuro vocês quiserem uma
    estimativa de profundidade de verdade, é aqui que entra.
    """

    skeleton_3d = {}

    if isinstance(keypoints_2d, dict):
        items = keypoints_2d.items()

    elif isinstance(keypoints_2d, list):
        items = [
            (point.get("name"), point)
            for point in keypoints_2d
            if isinstance(point, dict) and point.get("name")
        ]

    else:
        items = []

    for name, point in items:
        skeleton_3d[name] = {
            "x": float(point.get("x", 0)),
            "y": float(point.get("y", 0)),
            "z": float(point.get("z", 0)),
        }

    return skeleton_3d


@app.route(
    "/api/review/submit",
    methods=["POST"]
)
def review_submit():
    data = request.get_json(
        silent=True
    ) or {}

    post_id = data.get(
        "post_id"
    )

    review_status = data.get(
        "review_status"
    )

    keypoints_2d = data.get(
        "keypoints_2d"
    )

    keypoints_3d = data.get(
        "keypoints_3d"
    )

    keypoints_2d_raw = data.get(
        "keypoints_2d_raw"
    )

    if not post_id:
        return jsonify({
            "error":
                "post_id obrigatório."
        }), 400

    if not keypoints_2d and not keypoints_3d:
        return jsonify({
            "error":
                "keypoints_2d ou keypoints_3d obrigatório."
        }), 400

    # O TensorFlow.js já roda no navegador (review.html) e
    # normalmente vai mandar keypoints_3d prontos (via editor
    # manual). Se só vier o 2D, fazemos um lift simples com z=0.
    if not keypoints_3d:
        keypoints_3d = _naive_lift_to_3d(keypoints_2d)

    pose_id = f"pose_{uuid.uuid4().hex[:12]}"

    try:
        db.insert_skeleton(
            pose_id=pose_id,
            source_post_id=post_id,
            keypoints_2d=keypoints_2d,
            keypoints_2d_raw=keypoints_2d_raw,
            keypoints_3d=keypoints_3d,
            confidence=1.0,
            review_status=review_status,
            tags=[],
            verified=(
                review_status == 
                "manual_approved"
            )
        )

        db.mark_post_processed(
            post_id
        )

        # Se você aprovou/corrigiu a pose, esse par (imagem + pose
        # bruta + pose corrigida) vira dado de treino da rede
        # corretora — precisa ser guardado ANTES de apagar a imagem.
        if review_status == "manual_approved":
            _save_training_pair(
                post_id,
                keypoints_2d_raw,
                keypoints_2d,
            )

        # Já extraímos os keypoints — a imagem crua não precisa
        # mais ocupar espaço em disco.
        _delete_local_image(post_id)

        return jsonify({
            "success": True,
            "pose_id": pose_id
        })

    except Exception as exc:
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 500


if __name__ == "__main__":
    db.init_db()

    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )
