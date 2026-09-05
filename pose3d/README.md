# Pose3D

Coleta poses de personagens a partir de imagens do Danbooru, detecta o
esqueleto, reconstrói em 3D e armazena apenas os keypoints 3D verificados
(não as imagens).

## Status

Estrutura inicial do projeto (v0). Módulos ainda são stubs com
`NotImplementedError` — serão implementados um de cada vez:

1. `backend/db.py` — schema SQLite
2. `backend/danbooru.py` — coleta incremental
3. `backend/pose.py` — detecção de pose (biblioteca a definir)
4. `backend/skeleton.py` — normalização para o formato 3D
5. `backend/app.py` — rotas da API ligando tudo
6. `frontend/js/app.js` — chamadas reais à API

## Como rodar (quando implementado)

```bash
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows
pip install -r requirements.txt
cp .env.example .env
python backend/db.py       # inicializa o banco
python backend/app.py      # sobe o servidor em http://127.0.0.1:5000
```

## Estrutura

```
pose3d/
├── backend/
│   ├── app.py        # Flask + rotas
│   ├── config.py      # configs não-secretas (tags, thresholds, paths)
│   ├── danbooru.py    # client + coletor incremental
│   ├── pose.py         # detecção de pose 2D + confiança
│   ├── skeleton.py     # conversão para esqueleto 3D normalizado
│   └── db.py            # SQLite: schema + queries
├── frontend/
│   ├── index.html   # dashboard + coleta
│   ├── review.html   # revisão manual
│   ├── css/style.css
│   └── js/app.js
├── data/
│   └── tmp_images/     # imagens temporárias, descartadas após verificação
├── tests/
├── .env.example
└── requirements.txt
```

## Princípios do projeto

- Imagens do Danbooru são temporárias; só o esqueleto 3D é permanente.
- Baixa confiança na detecção → vai para revisão manual, nunca inventa pontos.
- Coleta é incremental e retomável (sem re-baixar posts já processados).
- Comparação/recomendação de poses por similaridade fica para uma versão futura;
  o schema do banco já é pensado com isso em mente, mas não é implementado agora.
