"""
db.py
Acesso ao SQLite. Schema pensado para permitir, no futuro, busca por
poses semelhantes (sem implementar a busca em si nesta versão).

Tabelas:

  danbooru_posts
    - post_id (PK)
    - status: 'pending' | 'downloaded' | 'processed' | 'error'
    - error_message
    - collected_at

  skeletons
    - pose_id (PK)
    - source_post_id (FK -> danbooru_posts.post_id)
    - keypoints_3d (JSON)
    - keypoints_2d (JSON) — pose final, já revisada/corrigida
    - keypoints_2d_raw (JSON) — pose que o MoveNet chutou antes da
      correção manual; usada como entrada de treino da rede
      corretora (active learning). None quando a detecção
      automática falhou ou o esqueleto foi criado manualmente.
    - confidence
    - verified (bool)
    - review_status: 'auto_approved' | 'manual_approved' | 'rejected' | 'pending_review'
    - tags (JSON list)
    - created_at

CRUD implementado nesta etapa. A busca por poses semelhantes fica para
uma etapa futura (o schema já comporta isso).
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import config

# ============================================================
# SCHEMA
# ============================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS danbooru_posts (
    post_id INTEGER PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    collected_at TEXT
);

CREATE TABLE IF NOT EXISTS skeletons (
    pose_id TEXT PRIMARY KEY,
    source_post_id INTEGER,
    keypoints_3d TEXT NOT NULL,
    keypoints_2d TEXT,
    keypoints_2d_raw TEXT,
    confidence REAL,
    verified INTEGER NOT NULL DEFAULT 0,
    review_status TEXT NOT NULL DEFAULT 'pending_review',
    tags TEXT,
    created_at TEXT,
    FOREIGN KEY (source_post_id) REFERENCES danbooru_posts (post_id)
);
"""

# ============================================================
# STATUS VÁLIDOS
# ============================================================

VALID_POST_STATUSES = {
    "pending",
    "downloaded",
    "processed",
    "error",
}

VALID_REVIEW_STATUSES = {
    "auto_approved",
    "manual_approved",
    "rejected",
    "pending_review",
}

# ============================================================
# UTILITÁRIOS
# ============================================================


def _now() -> str:
    """
    Retorna a data/hora atual em UTC no formato ISO.
    """
    return datetime.now(timezone.utc).isoformat()


def get_connection():
    """
    Abre uma conexão com o banco SQLite.
    """

    conn = sqlite3.connect(
        config.DB_PATH
    )

    conn.row_factory = sqlite3.Row

    conn.execute(
        "PRAGMA foreign_keys = ON"
    )

    return conn


@contextmanager
def _cursor():
    """
    Abre conexão, entrega um cursor, e cuida de
    commit/close/erro.
    """

    conn = get_connection()

    try:

        cur = conn.cursor()

        yield cur

        conn.commit()

    except Exception:

        conn.rollback()

        raise

    finally:

        conn.close()

# ============================================================
# INICIALIZAÇÃO DO BANCO
# ============================================================


def init_db():

    conn = get_connection()

    conn.executescript(
        SCHEMA
    )

    conn.commit()

    conn.close()

# ============================================================
# DANBOORU POSTS
# ============================================================


def upsert_post(
    post_id: int,
    status: str="pending",
    error_message: str | None=None
):
    """
    Cria o post se não existir, ou atualiza status/erro
    se já existir.

    Usado tanto pelo coletor (registrar post visto pela
    primeira vez) quanto para permitir retomar uma coleta
    interrompida sem duplicar.
    """

    if status not in VALID_POST_STATUSES:

        raise ValueError(
            f"status inválido: {status!r}"
        )

    with _cursor() as cur:

        cur.execute(
            """
            INSERT INTO danbooru_posts (
                post_id,
                status,
                error_message,
                collected_at
            )
            VALUES (?, ?, ?, ?)

            ON CONFLICT(post_id) DO UPDATE SET
                status = excluded.status,
                error_message = excluded.error_message
            """,
            (
                post_id,
                status,
                error_message,
                _now()
            ),
        )


def update_post_status(
    post_id: int,
    status: str,
    error_message: str | None=None
):
    """
    Atualiza o status de um post já existente.
    """

    if status not in VALID_POST_STATUSES:

        raise ValueError(
            f"status inválido: {status!r}"
        )

    with _cursor() as cur:

        cur.execute(
            """
            UPDATE danbooru_posts

            SET
                status = ?,
                error_message = ?

            WHERE post_id = ?
            """,
            (
                status,
                error_message,
                post_id
            ),
        )

        if cur.rowcount == 0:

            raise KeyError(
                f"post_id {post_id} não encontrado"
            )


def post_exists(
    post_id: int
) -> bool:
    """
    Usado pelo coletor para pular posts já
    processados/registrados.
    """

    with _cursor() as cur:

        cur.execute(
            """
            SELECT 1
            FROM danbooru_posts
            WHERE post_id = ?
            """,
            (post_id,)
        )

        return cur.fetchone() is not None


def get_post(
    post_id: int
) -> dict | None:

    with _cursor() as cur:

        cur.execute(
            """
            SELECT *
            FROM danbooru_posts
            WHERE post_id = ?
            """,
            (post_id,)
        )

        row = cur.fetchone()

        return dict(row) if row else None


def get_posts_by_status(
    status: str,
    limit: int | None=None
) -> list[dict]:

    if status not in VALID_POST_STATUSES:

        raise ValueError(
            f"status inválido: {status!r}"
        )

    query = """
        SELECT *
        FROM danbooru_posts
        WHERE status = ?
        ORDER BY collected_at ASC
    """

    params: list = [
        status
    ]

    if limit is not None:

        query += " LIMIT ?"

        params.append(
            limit
        )

    with _cursor() as cur:

        cur.execute(
            query,
            params
        )

        return [
            dict(row)
            for row in cur.fetchall()
        ]


def get_next_downloaded_post() -> dict | None:
    """
    Usado por GET /api/review/next:
    pega o post baixado mais antigo que ainda não foi
    revisado (status 'downloaded').
    """

    posts = get_posts_by_status(
        "downloaded",
        limit=1
    )

    return posts[0] if posts else None


def mark_post_processed(post_id: int):
    """
    Marca um post como 'processed' depois que a revisão
    (manual ou automática) foi concluída para ele.
    """

    update_post_status(
        post_id,
        status="processed"
    )


def get_last_collected_post_id() -> int | None:
    """
    Maior post_id já registrado — útil para checar
    posts novos desde a última coleta.
    """

    with _cursor() as cur:

        cur.execute(
            """
            SELECT MAX(post_id) AS max_id
            FROM danbooru_posts
            """
        )

        row = cur.fetchone()

        if (
            row
            and row["max_id"] is not None
        ):

            return row["max_id"]

        return None


def get_min_collected_post_id() -> int | None:
    """
    Menor post_id já registrado — checkpoint para retomar
    a coleta histórica (paginação 'before') sem re-buscar
    páginas já feitas.
    """

    with _cursor() as cur:

        cur.execute(
            """
            SELECT MIN(post_id) AS min_id
            FROM danbooru_posts
            """
        )

        row = cur.fetchone()

        if (
            row
            and row["min_id"] is not None
        ):

            return row["min_id"]

        return None

# ============================================================
# SKELETONS
# ============================================================


def insert_skeleton(
    pose_id: str,
    source_post_id: int | None,
    keypoints_3d: dict,
    confidence: float,
    keypoints_2d: dict | list | None=None,
    keypoints_2d_raw: dict | list | None=None,
    review_status: str="pending_review",
    tags: list | None=None,
    verified: bool=False,
):
    """
    Insere um esqueleto 3D no banco.

    source_post_id pode ser None quando o esqueleto
    foi criado manualmente.

    keypoints_2d é opcional (None quando o esqueleto
    foi criado manualmente, sem imagem de origem).
    """

    if review_status not in VALID_REVIEW_STATUSES:

        raise ValueError(
            f"review_status inválido: {review_status!r}"
        )

    if not isinstance(
        keypoints_3d,
        dict
    ):

        raise ValueError(
            "keypoints_3d precisa ser um dicionário."
        )

    with _cursor() as cur:

        cur.execute(
            """
            INSERT INTO skeletons (
                pose_id,
                source_post_id,
                keypoints_3d,
                keypoints_2d,
                keypoints_2d_raw,
                confidence,
                verified,
                review_status,
                tags,
                created_at
            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pose_id,

                # Pode ser None para esqueleto manual
                source_post_id,

                json.dumps(
                    keypoints_3d
                ),

                # Pode ser None para esqueleto manual
                json.dumps(keypoints_2d)
                if keypoints_2d is not None
                else None,

                # Pose antes da correção — None quando a detecção
                # automática falhou ou o esqueleto é manual
                json.dumps(keypoints_2d_raw)
                if keypoints_2d_raw is not None
                else None,

                float(confidence),

                int(verified),

                review_status,

                json.dumps(
                    tags or []
                ),

                _now(),
            ),
        )


def migrate_db():
    conn = get_connection()

    try:
        columns = [
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(skeletons)"
            ).fetchall()
        ]

        if "keypoints_2d" not in columns:
            conn.execute(
                "ALTER TABLE skeletons ADD COLUMN keypoints_2d TEXT"
            )

        if "keypoints_2d_raw" not in columns:
            conn.execute(
                "ALTER TABLE skeletons ADD COLUMN keypoints_2d_raw TEXT"
            )

        conn.commit()

    finally:
        conn.close()

        
def _row_to_skeleton(
    row: sqlite3.Row
) -> dict:

    d = dict(row)

    if d.get("keypoints_2d"):
        d["keypoints_2d"] = json.loads(
            d["keypoints_2d"]
        )
    else:
        d["keypoints_2d"] = {}

    if d.get("keypoints_2d_raw"):
        d["keypoints_2d_raw"] = json.loads(
            d["keypoints_2d_raw"]
        )
    else:
        d["keypoints_2d_raw"] = None

    d["keypoints_3d"] = json.loads(
        d["keypoints_3d"]
    )

    d["tags"] = (
        json.loads(d["tags"])
        if d["tags"]
        else []
    )

    d["verified"] = bool(
        d["verified"]
    )

    return d


def get_skeleton(
    pose_id: str
) -> dict | None:

    with _cursor() as cur:

        cur.execute(
            """
            SELECT *
            FROM skeletons
            WHERE pose_id = ?
            """,
            (pose_id,)
        )

        row = cur.fetchone()

        return (
            _row_to_skeleton(row)
            if row
            else None
        )


def list_skeletons(
    review_status: str | None=None,
    limit: int | None=None
) -> list[dict]:

    query = """
        SELECT *
        FROM skeletons
    """

    params: list = []

    if review_status is not None:

        if review_status not in VALID_REVIEW_STATUSES:

            raise ValueError(
                f"review_status inválido: {review_status!r}"
            )

        query += """
            WHERE review_status = ?
        """

        params.append(
            review_status
        )

    query += """
        ORDER BY created_at DESC
    """

    if limit is not None:

        query += """
            LIMIT ?
        """

        params.append(
            limit
        )

    with _cursor() as cur:

        cur.execute(
            query,
            params
        )

        return [
            _row_to_skeleton(row)
            for row in cur.fetchall()
        ]


def get_next_for_review() -> dict | None:
    """
    Usado por GET /api/review/next:
    pega o esqueleto pendente mais antigo.
    """

    with _cursor() as cur:

        cur.execute(
            """
            SELECT *
            FROM skeletons

            WHERE review_status =
                'pending_review'

            ORDER BY created_at ASC

            LIMIT 1
            """
        )

        row = cur.fetchone()

        return (
            _row_to_skeleton(row)
            if row
            else None
        )


def submit_review(
    pose_id: str,
    review_status: str,
    keypoints_3d: dict | None=None,
):
    """
    Usado por POST /api/review/submit:

    - aprova/rejeita
    - opcionalmente substitui os keypoints
      por uma versão editada manualmente.
    """

    if review_status not in VALID_REVIEW_STATUSES:

        raise ValueError(
            f"review_status inválido: {review_status!r}"
        )

    verified = (
        review_status
        in (
            "auto_approved",
            "manual_approved"
        )
    )

    with _cursor() as cur:

        if keypoints_3d is not None:

            cur.execute(
                """
                UPDATE skeletons

                SET
                    review_status = ?,
                    verified = ?,
                    keypoints_3d = ?

                WHERE pose_id = ?
                """,
                (
                    review_status,

                    int(
                        verified
                    ),

                    json.dumps(
                        keypoints_3d
                    ),

                    pose_id
                ),
            )

        else:

            cur.execute(
                """
                UPDATE skeletons

                SET
                    review_status = ?,
                    verified = ?

                WHERE pose_id = ?
                """,
                (
                    review_status,

                    int(
                        verified
                    ),

                    pose_id
                ),
            )

        if cur.rowcount == 0:

            raise KeyError(
                f"pose_id {pose_id} não encontrado"
            )

# ============================================================
# DASHBOARD
# GET /api/stats
# ============================================================


def get_stats() -> dict:

    with _cursor() as cur:

        # ----------------------------------------------------
        # Posts totais
        # ----------------------------------------------------

        cur.execute(
            """
            SELECT COUNT(*) AS n
            FROM danbooru_posts
            """
        )

        total_posts = cur.fetchone()["n"]

        # ----------------------------------------------------
        # Posts por status
        # ----------------------------------------------------

        cur.execute(
            """
            SELECT
                status,
                COUNT(*) AS n

            FROM danbooru_posts

            GROUP BY status
            """
        )

        posts_by_status = {
            row["status"]: row["n"]
            for row in cur.fetchall()
        }

        # ----------------------------------------------------
        # Esqueletos totais
        # ----------------------------------------------------

        cur.execute(
            """
            SELECT COUNT(*) AS n
            FROM skeletons
            """
        )

        total_skeletons = (
            cur.fetchone()["n"]
        )

        # ----------------------------------------------------
        # Esqueletos por status de revisão
        # ----------------------------------------------------

        cur.execute(
            """
            SELECT
                review_status,
                COUNT(*) AS n

            FROM skeletons

            GROUP BY review_status
            """
        )

        skeletons_by_review_status = {
            row["review_status"]: row["n"]
            for row in cur.fetchall()
        }

        # ----------------------------------------------------
        # Esqueletos verificados
        # ----------------------------------------------------

        cur.execute(
            """
            SELECT COUNT(*) AS n

            FROM skeletons

            WHERE verified = 1
            """
        )

        verified_count = (
            cur.fetchone()["n"]
        )

    return {

        "posts": {

            "total":
                total_posts,

            "by_status":
                posts_by_status,
        },

        "skeletons": {

            "total":
                total_skeletons,

            "verified":
                verified_count,

            "by_review_status":
                skeletons_by_review_status,
        },
    }


init_db()
migrate_db()
# ============================================================
# EXECUÇÃO DIRETA
# ============================================================

if __name__ == "__main__":

    init_db()

    print(
        f"Banco inicializado em "
        f"{config.DB_PATH}"
    )
