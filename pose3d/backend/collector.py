import threading
import logging

import db
from danbooru import DanbooruCollector

logger = logging.getLogger(__name__)
# Detecção de pose e construção do esqueleto 3D agora acontecem
# no navegador (TensorFlow.js), na tela de revisão. O coletor só
# baixa as imagens e registra o status no banco.


class CollectionManager:

    def __init__(self):
        self.status = "stopped"
        self.collected = 0
        self.downloaded = 0
        self.processed = 0
        self.errors = 0
        self.tags = []
        self.limit = 0

        self._pause_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread = None

    def start(self, tags, limit=0):
        if self.status == "running":
            return False

        self.tags = tags
        self.limit = limit

        self.collected = 0
        self.downloaded = 0
        self.processed = 0
        self.errors = 0

        self._pause_event.clear()
        self._stop_event.clear()

        self.status = "running"

        self._thread = threading.Thread(
            target=self._run,
            daemon=True
        )

        self._thread.start()

        return True

    def pause(self):
        if self.status != "running":
            return False

        self._pause_event.set()
        self.status = "paused"

        return True

    def resume(self):
        if self.status != "paused":
            return False

        self._pause_event.clear()
        self.status = "running"

        return True

    def stop(self):
        if self.status not in ("running", "paused"):
            return False

        self._stop_event.set()
        self._pause_event.clear()

        self.status = "stopped"

        return True

    def get_status(self):
        return {
            "status": self.status,
            "tags": self.tags,
            "limit": self.limit,
            "collected": self.collected,
            "downloaded": self.downloaded,
            "processed": self.processed,
            "errors": self.errors
        }

    def _run(self):
        try:
            collector = DanbooruCollector(
                tags=self.tags
            )

            for post in collector.iter_posts():

                # Parar
                if self._stop_event.is_set():
                    break

                # Pausar
                while self._pause_event.is_set():
                    if self._stop_event.is_set():
                        return

                    self._pause_event.wait(0.5)

                self.collected += 1

                # Baixar imagem
                image_path = collector.download_image(post)

                if image_path is None:
                    self.errors += 1
                    continue

                self.downloaded += 1

                # A partir daqui o post fica com status
                # "downloaded" (feito dentro de download_image)
                # e aguarda revisão manual em /review.html, onde
                # o TensorFlow.js roda a detecção de pose.

                # Limite opcional
                if self.limit > 0:
                    if self.collected >= self.limit:
                        break

            if self._stop_event.is_set():
                self.status = "stopped"
            else:
                self.status = "finished"

        except Exception as exc:
            logger.exception(
                "Erro durante coleta: %s",
                exc
            )

            self.errors += 1
            self.status = "error"
