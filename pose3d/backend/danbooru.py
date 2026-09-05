"""
danbooru.py

Responsável por toda comunicação com a API do Danbooru/Safebooru
e pela coleta incremental.

Estratégia de paginação:

    Usamos o cursor "before_id" da API do Danbooru
    (page=b<id>), que busca posts com id menor que <id>.

    Isso é estável mesmo com posts novos sendo publicados durante
    a coleta.

Responsabilidades:

    - consultar a API respeitando paginação
    - respeitar rate limit
    - autenticar usando login + API key
    - registrar posts no SQLite
    - continuar uma coleta interrompida
    - retry com backoff
    - baixar imagens
    - filtros/tags configuráveis
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import requests

import config
import db

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 30


@dataclass
class DanbooruPost:
    post_id: int
    image_url: str
    tags: str
    width: int
    height: int


class DanbooruCollector:
    """
    Coletor incremental de posts do Danbooru/Safebooru.

    Uso:

        collector = DanbooruCollector(
            tags=["1girl", "standing"]
        )

        for post in collector.iter_posts():
            path = collector.download_image(post)
            ...

    """

    def __init__(self, tags: list[str]):
        self.tags = tags

        # ---------------------------------------------------------
        # Sessão HTTP
        # ---------------------------------------------------------

        self.session = requests.Session()

        # Identifica o programa para o servidor.
        self.session.headers.update({
            "User-Agent": "Pose3DCollector/1.0",
            "Accept": "application/json",
        })

        # ---------------------------------------------------------
        # Checkpoint
        # ---------------------------------------------------------

        self._before_id = db.get_min_collected_post_id()

        # Controle do rate limit.
        self._last_request_time = 0.0

    # =============================================================
    # INFRAESTRUTURA
    # =============================================================

    def _respect_rate_limit(self):
        """
        Garante o intervalo mínimo entre requisições.
        """

        elapsed = time.monotonic() - self._last_request_time

        wait = (
            config.DANBOORU_REQUEST_INTERVAL_SECONDS
            -elapsed
        )

        if wait > 0:
            time.sleep(wait)

        self._last_request_time = time.monotonic()

    def _auth_params(self) -> dict:
        """
        Retorna os parâmetros de autenticação.

        A autenticação é feita com:

            login
            api_key
        """

        login = config.DANBOORU_LOGIN
        api_key = config.DANBOORU_API_KEY

        if login and api_key:
            return {
                "login": login,
                "api_key": api_key,
            }

        logger.warning(
            "Danbooru/Safebooru sem autenticação configurada."
        )

        return {}

    def _get_with_retry(
        self,
        url: str,
        params: dict,
    ) -> requests.Response | None:
        """
        Executa GET com retry e backoff.

        Retorna:
            requests.Response em caso de sucesso
            None se todas as tentativas falharem
        """

        for attempt in range(1, MAX_RETRIES + 1):

            self._respect_rate_limit()

            try:
                logger.debug(
                    "GET %s | parâmetros: %s",
                    url,
                    self._safe_params_for_log(params),
                )

                response = self.session.get(
                    url,
                    params=params,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )

                # -------------------------------------------------
                # Tratamento específico de HTTP
                # -------------------------------------------------

                if response.status_code == 403:
                    logger.error(
                        "Danbooru/Safebooru recusou a requisição "
                        "(HTTP 403). URL: %s",
                        url,
                    )

                elif response.status_code == 429:
                    logger.warning(
                        "Rate limit atingido pelo servidor "
                        "(HTTP 429)."
                    )

                response.raise_for_status()

                return response

            except requests.RequestException as exc:

                logger.warning(
                    "Falha na requisição ao Danbooru "
                    "(tentativa %d/%d): %s",
                    attempt,
                    MAX_RETRIES,
                    exc,
                )

                if attempt < MAX_RETRIES:

                    backoff = (
                        BACKOFF_BASE_SECONDS * attempt
                    )

                    logger.info(
                        "Aguardando %.1f segundos antes "
                        "da próxima tentativa...",
                        backoff,
                    )

                    time.sleep(backoff)

        logger.error(
            "Desistindo após %d tentativas: %s",
            MAX_RETRIES,
            url,
        )

        return None

    @staticmethod
    def _safe_params_for_log(params: dict) -> dict:
        """
        Remove a API key dos logs.

        Assim a chave nunca aparece no terminal.
        """

        safe_params = dict(params)

        if "api_key" in safe_params:
            safe_params["api_key"] = "***"

        return safe_params

    # =============================================================
    # API PÚBLICA
    # =============================================================

    def iter_posts(self):
        """
        Gera posts um por um.

        Respeita:
            - rate limit
            - autenticação
            - paginação
            - checkpoint
            - posts já existentes no banco
        """

        url = (
            f"{config.DANBOORU_BASE_URL.rstrip('/')}"
            "/posts.json"
        )

        logger.info(
            "Iniciando coleta no endpoint: %s",
            url,
        )

        logger.info(
            "Tags: %s",
            self.tags,
        )

        while True:

            params = {
                "tags": " ".join(self.tags),
                "limit": config.DANBOORU_PAGE_LIMIT,
                **self._auth_params(),
            }

            # -----------------------------------------------------
            # Paginação por cursor
            # -----------------------------------------------------

            if self._before_id is not None:
                params["page"] = (
                    f"b{self._before_id}"
                )

            response = self._get_with_retry(
                url,
                params,
            )

            if response is None:
                logger.error(
                    "Não foi possível consultar os posts. "
                    "Coleta interrompida."
                )
                return

            # -----------------------------------------------------
            # JSON
            # -----------------------------------------------------

            try:
                raw_posts = response.json()

            except ValueError:
                logger.error(
                    "Resposta inválida do servidor "
                    "(não é JSON)."
                )
                return

            if not isinstance(raw_posts, list):

                logger.error(
                    "Resposta inesperada da API: %s",
                    type(raw_posts).__name__,
                )

                return

            # -----------------------------------------------------
            # Fim da coleta
            # -----------------------------------------------------

            if not raw_posts:

                logger.info(
                    "Nenhum post encontrado. "
                    "Fim da coleta."
                )

                return

            logger.info(
                "Página recebida: %d posts.",
                len(raw_posts),
            )

            page_min_id = None

            # =====================================================
            # PROCESSAMENTO DOS POSTS
            # =====================================================

            for raw in raw_posts:

                if not isinstance(raw, dict):
                    continue

                post_id = raw.get("id")

                if post_id is None:
                    continue

                try:
                    post_id = int(post_id)

                except (TypeError, ValueError):
                    continue

                # -------------------------------------------------
                # Atualiza cursor
                # -------------------------------------------------

                if (
                    page_min_id is None
                    or post_id < page_min_id
                ):
                    page_min_id = post_id

                # -------------------------------------------------
                # Post já existente
                # -------------------------------------------------

                if db.post_exists(post_id):
                    logger.debug(
                        "Post %d já existe no banco. Pulando.",
                        post_id,
                    )
                    continue

                # -------------------------------------------------
                # URL da imagem
                # -------------------------------------------------

                image_url = raw.get("file_url")

                if not image_url:

                    db.upsert_post(
                        post_id,
                        status="error",
                        error_message="sem file_url",
                    )

                    logger.warning(
                        "Post %d sem file_url. Pulando.",
                        post_id,
                    )

                    continue

                # -------------------------------------------------
                # Dados do post
                # -------------------------------------------------

                tags = raw.get(
                    "tag_string",
                    "",
                )

                width = raw.get(
                    "image_width",
                    0,
                )

                height = raw.get(
                    "image_height",
                    0,
                )

                # -------------------------------------------------
                # Registra como pendente
                # -------------------------------------------------

                db.upsert_post(
                    post_id,
                    status="pending",
                )

                yield DanbooruPost(
                    post_id=post_id,
                    image_url=image_url,
                    tags=tags,
                    width=width,
                    height=height,
                )

            # -----------------------------------------------------
            # Proteção contra loop infinito
            # -----------------------------------------------------

            if page_min_id is None:

                logger.warning(
                    "Nenhum ID válido encontrado na página. "
                    "Coleta interrompida."
                )

                return

            # -----------------------------------------------------
            # Próximo cursor
            # -----------------------------------------------------

            self._before_id = page_min_id

            logger.debug(
                "Próximo checkpoint: before_id=%s",
                self._before_id,
            )

    # =============================================================
    # DOWNLOAD
    # =============================================================

    def download_image(
        self,
        post: DanbooruPost,
    ) -> str | None:
        """
        Baixa a imagem para:

            data/tmp_images/

        Retorna o caminho local ou None em caso de erro.
        """

        # ---------------------------------------------------------
        # Extensão
        # ---------------------------------------------------------

        suffix = Path(
            post.image_url
        ).suffix.lower()

        if not suffix or len(suffix) > 10:
            suffix = ".jpg"

        dest_path = (
            config.TMP_IMAGES_DIR
            / f"{post.post_id}{suffix}"
        )

        # ---------------------------------------------------------
        # Já existe?
        # ---------------------------------------------------------

        if dest_path.exists():

            logger.debug(
                "Imagem do post %d já existe: %s",
                post.post_id,
                dest_path,
            )

            db.update_post_status(
                post.post_id,
                status="downloaded",
            )

            return str(dest_path)

        # ---------------------------------------------------------
        # Download
        # ---------------------------------------------------------

        logger.info(
            "Baixando imagem do post %d...",
            post.post_id,
        )

        response = self._get_with_retry(
            post.image_url,
            params={},
        )

        if response is None:

            db.update_post_status(
                post.post_id,
                status="error",
                error_message=(
                    "falha ao baixar imagem"
                ),
            )

            return None

        # ---------------------------------------------------------
        # Verifica conteúdo
        # ---------------------------------------------------------

        if not response.content:

            logger.error(
                "Imagem vazia para o post %d.",
                post.post_id,
            )

            db.update_post_status(
                post.post_id,
                status="error",
                error_message="imagem vazia",
            )

            return None

        # ---------------------------------------------------------
        # Salva arquivo
        # ---------------------------------------------------------

        try:

            dest_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            dest_path.write_bytes(
                response.content
            )

        except OSError as exc:

            logger.error(
                "Falha ao salvar imagem "
                "do post %d: %s",
                post.post_id,
                exc,
            )

            db.update_post_status(
                post.post_id,
                status="error",
                error_message=str(exc),
            )

            return None

        # ---------------------------------------------------------
        # Sucesso
        # ---------------------------------------------------------

        db.update_post_status(
            post.post_id,
            status="downloaded",
        )

        logger.info(
            "Imagem do post %d salva em: %s",
            post.post_id,
            dest_path,
        )

        return str(dest_path)
