const API_BASE =
  window.location.port === "5500" ||
    window.location.port === "5501"
    ? `${window.location.protocol}//${window.location.hostname}:5000/api`
    : "/api";

const CANVAS_SIZE = 512;

let detector = null;
let currentReviewItem = null;
let reviewImage = null;
let reviewKeypoints2D = {};
// Pose crua do MoveNet, congelada assim que a detecção termina —
// nunca é editada. É o que vira dado de treino da rede corretora
// (junto com reviewKeypoints2D final, no momento do submit).
let reviewKeypoints2DRaw = null;
let selectedPoint2D = -1;

// Rede corretora (active learning): pequena rede treinada com suas
// correções passadas, que ajusta a pose do MoveNet antes de você
// nem precisar mexer. Se ainda não existe modelo exportado (dataset
// pequeno demais), fica null e o fluxo funciona como antes.
let correctorModel = null;
let correctorLoadAttempted = false;

// Ordem tem que ser IDÊNTICA a config.POSE_KEYPOINT_NAMES no
// backend — é o índice nessa lista que liga uma posição do vetor
// de entrada/saída da rede corretora ao nome da articulação.
const COCO_KEYPOINTS = [
  "nose",
  "left_eye", "right_eye",
  "left_ear", "right_ear",
  "left_shoulder", "right_shoulder",
  "left_elbow", "right_elbow",
  "left_wrist", "right_wrist",
  "left_hip", "right_hip",
  "left_knee", "right_knee",
  "left_ankle", "right_ankle"
];

let imageTransform = { scale: 1, offsetX: 0, offsetY: 0, imgWidth: 0, imgHeight: 0 };
let interactionMode = "view"; // view | add | remove | connect
let connectFirstPoint = null;
let draggingPoint = null;
let hoverPos = null; // última posição do mouse sobre o canvas — usado pelo atalho "X"
let customEdges = []; // [[nameA, nameB], ...]

let skeleton3D = {}; // name -> {x, y, z}
let threeState = null; // {scene, camera, renderer, group, markers, lines}
let THREE = null; // módulo three.js, carregado sob demanda (ver initSkeleton3DViewer)

const EDGES = [
  ["nose", "left_eye"],
  ["nose", "right_eye"],
  ["left_eye", "left_ear"],
  ["right_eye", "right_ear"],

  ["left_shoulder", "right_shoulder"],

  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],

  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],

  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],

  ["left_hip", "right_hip"],

  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],

  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"]
];


async function initPoseDetector() {
  if (detector) {
    return detector;
  }

  await tf.ready();

  detector = await poseDetection.createDetector(
    poseDetection.SupportedModels.MoveNet,
    {
      modelType:
        poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING
    }
  );

  return detector;
}


// ------------------------------------------------------------
// Rede corretora (active learning)
// ------------------------------------------------------------

async function loadCorrectorModel() {
  if (correctorModel || correctorLoadAttempted) return correctorModel;
  correctorLoadAttempted = true;

  try {
    correctorModel = await tf.loadLayersModel(
      "js/models/corrector/model.json"
    );
  } catch (error) {
    // Normal enquanto não existe modelo treinado ainda (dataset
    // pequeno demais) — segue sem corretor, só com o MoveNet cru.
    // Não tenta de novo nas próximas imagens (correctorLoadAttempted
    // fica true) pra não spammar 404 a cada review_next.
    console.warn("Modelo corretor não encontrado, usando MoveNet puro:", error.message);
    correctorModel = null;
  }

  return correctorModel;
}


// Monta o vetor de entrada da rede: [x_norm, y_norm, score] por
// articulação, na ordem de COCO_KEYPOINTS, normalizado pelo
// tamanho da imagem. Pontos ausentes viram [0, 0, 0].
function keypointsToInputVector(keypoints, imgWidth, imgHeight) {
  const vec = [];

  for (const name of COCO_KEYPOINTS) {
    const p = keypoints[name];

    if (p) {
      vec.push(
        p.x / imgWidth,
        p.y / imgHeight,
        p.score ?? 0
      );
    } else {
      vec.push(0, 0, 0);
    }
  }

  return vec;
}


// Aplica a rede corretora em cima da pose crua do MoveNet. A rede
// prevê um delta (dx, dy) normalizado por articulação; somamos ao
// x/y original. Pontos que o MoveNet não detectou continuam
// ausentes (a rede só corrige o que já existe, não inventa pontos).
async function applyCorrector(keypointsRaw, imgWidth, imgHeight) {
  if (!correctorModel || !keypointsRaw) return keypointsRaw;

  const inputVec = keypointsToInputVector(keypointsRaw, imgWidth, imgHeight);

  const deltas = tf.tidy(() => {
    const input = tf.tensor2d([inputVec]);
    const output = correctorModel.predict(input);
    return output.dataSync();
  });

  const corrected = {};

  COCO_KEYPOINTS.forEach((name, i) => {
    const p = keypointsRaw[name];
    if (!p) return;

    corrected[name] = {
      x: p.x + deltas[i * 2] * imgWidth,
      y: p.y + deltas[i * 2 + 1] * imgHeight,
      score: p.score
    };
  });

  return corrected;
}


function getPoseCanvas() {
  return document.getElementById("pose-canvas");
}


function setReviewMode(text) {
  const element = document.getElementById("mode-label");
  if (element) {
    element.textContent = text;
  }
}


function showReviewError(message) {
  const element = document.getElementById("review-error");
  if (element) {
    element.textContent = message;
    element.style.display = "block";
  } else {
    console.error(message);
  }
}


function clearReviewError() {
  const element = document.getElementById("review-error");
  if (element) {
    element.textContent = "";
    element.style.display = "none";
  }
}


// ------------------------------------------------------------
// Transformação imagem <-> canvas (tamanho padrão CANVAS_SIZE)
// ------------------------------------------------------------

function computeImageTransform(img) {
  const scale = Math.min(
    CANVAS_SIZE / img.naturalWidth,
    CANVAS_SIZE / img.naturalHeight
  );

  const imgWidth = img.naturalWidth;
  const imgHeight = img.naturalHeight;

  return {
    scale,
    imgWidth,
    imgHeight,
    offsetX: (CANVAS_SIZE - imgWidth * scale) / 2,
    offsetY: (CANVAS_SIZE - imgHeight * scale) / 2
  };
}

function toCanvasCoords(p) {
  return {
    x: p.x * imageTransform.scale + imageTransform.offsetX,
    y: p.y * imageTransform.scale + imageTransform.offsetY
  };
}

function toImageCoords(p) {
  return {
    x: (p.x - imageTransform.offsetX) / imageTransform.scale,
    y: (p.y - imageTransform.offsetY) / imageTransform.scale
  };
}


function clearCanvas() {
  const canvas = getPoseCanvas();
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  reviewImage = null;
  reviewKeypoints2D = {};
  reviewKeypoints2DRaw = null;
  selectedPoint2D = -1;
  customEdges = [];
  draggingPoint = null;
  hoverPos = null;
  connectFirstPoint = null;
  interactionMode = "view";
  setModeButtonsUI();

  skeleton3D = {};
  if (threeState) {
    for (const name of Object.keys(threeState.markers)) {
      removeJoint3DObject(name);
    }
    rebuildSkeleton3DLines();
  }
  populateJointSelect3D();
}


async function detectPoseOnImage(image) {
  await initPoseDetector();

  const poses = await detector.estimatePoses(image);

  if (!poses.length) {
    throw new Error("Nenhuma pessoa foi detectada.");
  }

  return poses[0];
}


function normalizeKeypoints(keypoints) {
  const result = {};

  for (const point of keypoints) {
    if (!point.name) continue;

    result[point.name] = {
      x: Number(point.x),
      y: Number(point.y),
      score: Number(point.score ?? 0)
    };
  }

  return result;
}


function drawSkeleton(ctx, points, edges) {
  ctx.lineWidth = 3;

  for (const [a, b] of edges) {
    const p1 = points[a];
    const p2 = points[b];

    if (!p1 || !p2) continue;
    if (p1.score < 0.25 || p2.score < 0.25) continue;

    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
  }
}


function redrawCanvas() {
  const canvas = getPoseCanvas();
  if (!canvas || !reviewImage) return;

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.drawImage(
    reviewImage,
    imageTransform.offsetX,
    imageTransform.offsetY,
    imageTransform.imgWidth * imageTransform.scale,
    imageTransform.imgHeight * imageTransform.scale
  );

  const canvasPoints = {};
  for (const [name, p] of Object.entries(reviewKeypoints2D)) {
    canvasPoints[name] = { ...toCanvasCoords(p), score: p.score };
  }

  ctx.strokeStyle = "#4fc3f7";
  drawSkeleton(ctx, canvasPoints, EDGES);

  ctx.strokeStyle = "#ffb300";
  drawSkeleton(ctx, canvasPoints, customEdges);

  for (const [name, p] of Object.entries(canvasPoints)) {
    const isSelected = name === selectedPoint2D;
    const isConnectFirst = name === connectFirstPoint;
    const isLowConfidence = p.score < 0.25;

    // Antes, pontos de baixa confiança (score < 0.25) simplesmente
    // não eram desenhados — existiam no dado, mas ficavam invisíveis
    // e impossíveis de localizar/apagar na tela. Agora desenham
    // igual, só que "fantasma" (mais transparente, com contorno),
    // pra dar pra ver e clicar/deletar neles.
    ctx.globalAlpha = isLowConfidence ? 0.45 : 1;

    ctx.fillStyle = isSelected || isConnectFirst
      ? "#ff5252"
      : isLowConfidence ? "#9e9e9e" : "#4fc3f7";

    ctx.beginPath();
    ctx.arc(p.x, p.y, isSelected || isConnectFirst ? 7 : 5, 0, Math.PI * 2);
    ctx.fill();

    if (isLowConfidence) {
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    ctx.globalAlpha = 1;

    ctx.fillStyle = "#ffffff";
    ctx.font = "11px Arial";
    ctx.fillText(name, p.x + 7, p.y - 7);
  }
}


async function drawReviewImage(item) {
  const canvas = getPoseCanvas();
  if (!canvas) return;

  canvas.width = CANVAS_SIZE;
  canvas.height = CANVAS_SIZE;

  // item.image_url vem do backend como caminho relativo
  // ("/api/review/image/123"). Isso funciona quando o front é
  // servido pelo próprio Flask, mas quebra (404) quando o
  // review.html é aberto por outra porta (ex: Live Server em
  // 5500/5501) — nesse caso o navegador tentava buscar a
  // imagem no servidor errado. Resolve com o mesmo API_BASE
  // já usado nos fetch() abaixo.
  const imageUrl = item.image_url
    ? item.image_url.replace(/^\/api/, API_BASE)
    : null;

  if (!imageUrl) {
    throw new Error("Imagem não encontrada.");
  }

  const img = new Image();

  await new Promise((resolve, reject) => {
    img.onload = resolve;
    img.onerror = () => reject(new Error("Não foi possível carregar a imagem."));
    img.src = imageUrl;
  });

  reviewImage = img;
  imageTransform = computeImageTransform(img);

  reviewKeypoints2D = {};
  reviewKeypoints2DRaw = null;
  customEdges = [];
  selectedPoint2D = -1;
  connectFirstPoint = null;

  redrawCanvas();

  setReviewMode("detectando pose...");
  clearReviewError();

  try {
    const pose = await detectPoseOnImage(img);
    reviewKeypoints2DRaw = normalizeKeypoints(pose.keypoints);

    // Se já existe uma rede corretora treinada, aplica em cima do
    // MoveNet antes de te mostrar a pose. reviewKeypoints2DRaw fica
    // intocado — é ele que vai pro backend como "pose bruta" pra
    // virar dado de treino na próxima rodada.
    await loadCorrectorModel();
    reviewKeypoints2D = await applyCorrector(
      reviewKeypoints2DRaw,
      imageTransform.imgWidth,
      imageTransform.imgHeight
    );

    setReviewMode(
      correctorModel ? "pose detectada (corrigida)" : "pose detectada"
    );
  } catch (error) {
    // Comum em ilustrações/anime: o MoveNet é treinado em fotos
    // reais e pode não reconhecer ninguém. Isso não trava mais o
    // fluxo — a imagem fica em branco e dá pra montar o esqueleto
    // manualmente com "Adicionar ponto".
    console.warn("Detecção automática falhou:", error);
    reviewKeypoints2D = {};
    reviewKeypoints2DRaw = null;
    setReviewMode("pose não detectada — adicione os pontos manualmente");
    showReviewError(
      "Não foi possível detectar a pose automaticamente nesta imagem. " +
      "Use \"Adicionar ponto\" para montar o esqueleto manualmente."
    );
  }

  redrawCanvas();
  syncSkeleton3DJoints();
}


async function loadNextReviewItem() {
  const MAX_AUTO_SKIP = 25;

  for (let attempt = 0; attempt <= MAX_AUTO_SKIP; attempt++) {
    try {
      clearReviewError();
      setReviewMode("carregando imagem...");

      const response = await fetch(`${API_BASE}/review/next`, { cache: "no-store" });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();

      if (data.finished || !data.post) {
        currentReviewItem = null;
        clearCanvas();
        setReviewMode("nenhuma imagem pendente");
        return;
      }

      currentReviewItem = data.post;
      await drawReviewImage(currentReviewItem);
      return; // sucesso, encerra o loop

    } catch (error) {
      console.error(error);
      setReviewMode("erro");
      showReviewError(error.message);
      // Post com imagem ausente/corrompida: o backend já marca
      // esse post como "error", então a próxima volta do loop já
      // busca outro post. Continua até achar um válido ou bater
      // o limite de tentativas.
    }
  }

  showReviewError(
    `${MAX_AUTO_SKIP + 1} posts seguidos falharam ao carregar a imagem. ` +
    "Rode o script cleanup_missing_images.py pra corrigir o banco de uma vez, " +
    "ou verifique a pasta data/tmp_images."
  );
}


async function submitReview(decision) {
  if (!currentReviewItem) return;

  try {
    setReviewMode("salvando...");

    const response = await fetch(`${API_BASE}/review/submit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        post_id: currentReviewItem.post_id,
        review_status: decision === "approve" ? "manual_approved" : "rejected",
        keypoints_2d: reviewKeypoints2D,
        keypoints_2d_raw: reviewKeypoints2DRaw,
        custom_edges: customEdges
      })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || `HTTP ${response.status}`);
    }

    await loadNextReviewItem();

  } catch (error) {
    console.error(error);
    showReviewError("Erro ao salvar: " + error.message);
  }
}


// ------------------------------------------------------------
// Edição interativa dos pontos 2D (arrastar, adicionar, remover, conectar)
// ------------------------------------------------------------

function getCanvasMousePos(evt) {
  const canvas = getPoseCanvas();
  const rect = canvas.getBoundingClientRect();
  return {
    x: (evt.clientX - rect.left) * (canvas.width / rect.width),
    y: (evt.clientY - rect.top) * (canvas.height / rect.height)
  };
}

function findNearestPoint(cx, cy, threshold = 12) {
  let closest = null;
  let closestDist = threshold;

  for (const [name, p] of Object.entries(reviewKeypoints2D)) {
    const cp = toCanvasCoords(p);
    const d = Math.hypot(cp.x - cx, cp.y - cy);
    if (d <= closestDist) {
      closestDist = d;
      closest = name;
    }
  }

  return closest;
}

// Diferente de findNearestPoint (só o mais próximo), essa pega
// TODOS os pontos dentro do raio — usada pelo atalho de deletar
// com X, que precisa apagar todos que estiverem empilhados sob o
// mouse de uma vez, não só um.
function findPointsNear(cx, cy, threshold = 12) {
  const hits = [];

  for (const [name, p] of Object.entries(reviewKeypoints2D)) {
    const cp = toCanvasCoords(p);
    if (Math.hypot(cp.x - cx, cp.y - cy) <= threshold) {
      hits.push(name);
    }
  }

  return hits;
}

function toggleCustomEdge(a, b) {
  const idx = customEdges.findIndex(
    ([x, y]) => (x === a && y === b) || (x === b && y === a)
  );

  if (idx >= 0) {
    customEdges.splice(idx, 1);
  } else {
    customEdges.push([a, b]);
  }
}

function onCanvasMouseDown(evt) {
  if (!reviewImage) return;

  const pos = getCanvasMousePos(evt);
  const hit = findNearestPoint(pos.x, pos.y);

  if (interactionMode === "add") {
    if (hit) {
      showReviewError("Já existe um ponto próximo. Escolha outro local.");
      return;
    }

    const name = window.prompt("Nome do novo ponto (ex: left_finger_tip):");
    if (!name) return;

    if (reviewKeypoints2D[name]) {
      showReviewError("Já existe um ponto com esse nome.");
      return;
    }

    clearReviewError();
    const imgPos = toImageCoords(pos);
    reviewKeypoints2D[name] = { x: imgPos.x, y: imgPos.y, score: 1 };
    selectedPoint2D = name;

    // Volta pro modo padrão (arrastar) assim que o ponto é criado.
    // Sem isso, o próximo clique nesse mesmo ponto era interpretado
    // como "tentar adicionar outro ponto aqui" em vez de arrastá-lo.
    interactionMode = "view";
    setModeButtonsUI();

    syncSkeleton3DJoints();
    redrawCanvas();
    return;
  }

  if (interactionMode === "remove") {
    if (hit) {
      delete reviewKeypoints2D[hit];
      customEdges = customEdges.filter(([a, b]) => a !== hit && b !== hit);
      if (selectedPoint2D === hit) selectedPoint2D = -1;

      // Mesma lógica: depois de remover, volta a poder arrastar
      // normalmente sem precisar desligar o modo manualmente.
      interactionMode = "view";
      setModeButtonsUI();

      syncSkeleton3DJoints();
      redrawCanvas();
    }
    return;
  }

  if (interactionMode === "connect") {
    if (!hit) return;

    if (!connectFirstPoint) {
      connectFirstPoint = hit;
      redrawCanvas();
      return;
    }

    if (connectFirstPoint !== hit) {
      toggleCustomEdge(connectFirstPoint, hit);
      syncSkeleton3DJoints();
    }

    connectFirstPoint = null;

    // Conexão concluída: volta pro modo padrão (arrastar) em vez
    // de ficar esperando outro par de pontos pra conectar.
    interactionMode = "view";
    setModeButtonsUI();

    redrawCanvas();
    return;
  }

  // modo padrão: Shift + clique em dois pontos conecta os dois,
  // sem precisar sair do modo de arrastar (tudo junto).
  if (hit && evt.shiftKey) {
    if (!connectFirstPoint) {
      connectFirstPoint = hit;
      redrawCanvas();
      return;
    }

    if (connectFirstPoint !== hit) {
      toggleCustomEdge(connectFirstPoint, hit);
      syncSkeleton3DJoints();
    }

    connectFirstPoint = null;
    redrawCanvas();
    return;
  }

  // clique/arraste normal: mover um ponto existente
  if (hit) {
    draggingPoint = hit;
    selectedPoint2D = hit;
    connectFirstPoint = null;
    redrawCanvas();
  } else {
    selectedPoint2D = -1;
    connectFirstPoint = null;
    redrawCanvas();
  }
}

function onCanvasMouseMove(evt) {
  const pos = getCanvasMousePos(evt);
  hoverPos = pos; // usado pelo atalho de deletar com a tecla X

  if (!draggingPoint) return;

  const imgPos = toImageCoords(pos);

  reviewKeypoints2D[draggingPoint].x = imgPos.x;
  reviewKeypoints2D[draggingPoint].y = imgPos.y;

  updateSkeleton3DFrom2D(draggingPoint);
  redrawCanvas();
}

function onCanvasMouseLeave() {
  hoverPos = null;
}

function onCanvasMouseUp() {
  draggingPoint = null;
}

// Passa o mouse em cima de um ou mais pontos e aperta X: apaga
// todos que estiverem sob o cursor, não só o mais próximo. Ignora
// quando o foco está num campo de texto (ex: nome do ponto no
// "Adicionar ponto", campos do painel manual 3D), pra não apagar
// nada sem querer enquanto você digita.
function onKeyDown(evt) {
  const tag = (evt.target?.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select") return;

  if (evt.key.toLowerCase() !== "x") return;
  if (!reviewImage || !hoverPos) return;

  const hits = findPointsNear(hoverPos.x, hoverPos.y);
  if (!hits.length) return;

  for (const name of hits) {
    delete reviewKeypoints2D[name];
    customEdges = customEdges.filter(([a, b]) => a !== name && b !== name);

    if (selectedPoint2D === name) selectedPoint2D = -1;
    if (connectFirstPoint === name) connectFirstPoint = null;
    if (draggingPoint === name) draggingPoint = null;
  }

  syncSkeleton3DJoints();
  redrawCanvas();
}

function setModeButtonsUI() {
  const map = {
    add: "btn-add-point",
    remove: "btn-remove-point",
    connect: "btn-connect"
  };

  for (const [mode, id] of Object.entries(map)) {
    const btn = document.getElementById(id);
    if (btn) btn.classList.toggle("mode-active", interactionMode === mode);
  }
}

function setInteractionMode(mode) {
  interactionMode = interactionMode === mode ? "view" : mode;
  connectFirstPoint = null;
  setModeButtonsUI();
  redrawCanvas();
}

function setupEditTools() {
  const canvas = getPoseCanvas();
  if (canvas) {
    canvas.addEventListener("mousedown", onCanvasMouseDown);
    canvas.addEventListener("mousemove", onCanvasMouseMove);
    canvas.addEventListener("mouseleave", onCanvasMouseLeave);
    window.addEventListener("mouseup", onCanvasMouseUp);
    window.addEventListener("keydown", onKeyDown);
  }

  const addBtn = document.getElementById("btn-add-point");
  if (addBtn) addBtn.addEventListener("click", () => setInteractionMode("add"));

  const removeBtn = document.getElementById("btn-remove-point");
  if (removeBtn) removeBtn.addEventListener("click", () => setInteractionMode("remove"));

  const connectBtn = document.getElementById("btn-connect");
  if (connectBtn) connectBtn.addEventListener("click", () => setInteractionMode("connect"));
}


// ------------------------------------------------------------
// Visualizador 3D (Three.js) — cria a cena de verdade
// ------------------------------------------------------------

async function initSkeleton3DViewer() {
  const container = document.getElementById("skeleton-3d-viewer");
  if (!container) return; // ex.: dashboard (index.html) não tem esse painel

  try {
    // Build "build/three.min.js" está deprecado (removido no r160).
    // Carregamos o módulo ES via import map declarado no review.html.
    THREE = await import("three");
  } catch (error) {
    console.error("Não foi possível carregar three.js:", error);
    showReviewError("Visualizador 3D indisponível (falha ao carregar three.js).");
    return;
  }

  const width = container.clientWidth || 400;
  const height = container.clientHeight || 400;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x161616);

  const camera = new THREE.PerspectiveCamera(50, width / height, 0.05, 100);
  camera.position.set(0, 0, 2.5);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  container.innerHTML = "";
  container.appendChild(renderer.domElement);

  scene.add(new THREE.AmbientLight(0xffffff, 1));

  const group = new THREE.Group();
  scene.add(group);

  threeState = { scene, camera, renderer, group, markers: {}, lines: {} };

  function animate() {
    requestAnimationFrame(animate);
    group.rotation.y += 0.006;
    renderer.render(scene, camera);
  }
  animate();

  window.addEventListener("resize", () => {
    const w = container.clientWidth || width;
    const h = container.clientHeight || height;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });

  syncSkeleton3DJoints();
}

function ensureJoint3DObject(name) {
  if (!threeState || threeState.markers[name]) return;

  const geometry = new THREE.SphereGeometry(0.035, 12, 12);
  const material = new THREE.MeshBasicMaterial({ color: 0x4fc3f7 });
  const marker = new THREE.Mesh(geometry, material);

  threeState.group.add(marker);
  threeState.markers[name] = marker;
}

function removeJoint3DObject(name) {
  if (!threeState || !threeState.markers[name]) return;
  threeState.group.remove(threeState.markers[name]);
  delete threeState.markers[name];
}

function rebuildSkeleton3DLines() {
  if (!threeState) return;

  for (const { line } of Object.values(threeState.lines)) {
    threeState.group.remove(line);
  }
  threeState.lines = {};

  const allEdges = [...EDGES, ...customEdges];
  const material = new THREE.LineBasicMaterial({ color: 0xffffff });

  for (const [a, b] of allEdges) {
    if (!skeleton3D[a] || !skeleton3D[b]) continue;

    const key = a + "|" + b;
    if (threeState.lines[key]) continue;

    const geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(),
      new THREE.Vector3()
    ]);
    const line = new THREE.Line(geometry, material);

    threeState.group.add(line);
    threeState.lines[key] = { line, a, b };
  }
}

function updateSkeleton3DPositions() {
  if (!threeState) return;

  for (const [name, marker] of Object.entries(threeState.markers)) {
    const p = skeleton3D[name];
    if (!p) continue;
    marker.position.set(p.x, p.y, p.z);
  }

  for (const { line, a, b } of Object.values(threeState.lines)) {
    const pa = skeleton3D[a];
    const pb = skeleton3D[b];
    if (!pa || !pb) continue;

    const positions = line.geometry.attributes.position;
    positions.setXYZ(0, pa.x, pa.y, pa.z);
    positions.setXYZ(1, pb.x, pb.y, pb.z);
    positions.needsUpdate = true;
  }
}

function updateSkeleton3DFrom2D(name) {
  const p2d = reviewKeypoints2D[name];
  if (!p2d || !skeleton3D[name]) return;

  skeleton3D[name].x = imageTransform.imgWidth
    ? ((p2d.x / imageTransform.imgWidth) - 0.5) * 2
    : 0;
  skeleton3D[name].y = imageTransform.imgHeight
    ? -((p2d.y / imageTransform.imgHeight) - 0.5) * 2
    : 0;

  updateSkeleton3DPositions();
  if (document.getElementById("joint3d")?.value === name) {
    loadSelectedJoint3D();
  }
}

function syncSkeleton3DJoints() {
  const names = Object.keys(reviewKeypoints2D);

  for (const name of Object.keys(skeleton3D)) {
    if (!names.includes(name)) {
      delete skeleton3D[name];
      removeJoint3DObject(name);
    }
  }

  for (const name of names) {
    const p2d = reviewKeypoints2D[name];

    if (!skeleton3D[name]) {
      skeleton3D[name] = {
        x: imageTransform.imgWidth
          ? ((p2d.x / imageTransform.imgWidth) - 0.5) * 2
          : 0,
        y: imageTransform.imgHeight
          ? -((p2d.y / imageTransform.imgHeight) - 0.5) * 2
          : 0,
        z: 0
      };
      ensureJoint3DObject(name);
    }
  }

  rebuildSkeleton3DLines();
  updateSkeleton3DPositions();
  populateJointSelect3D();
}

function populateJointSelect3D() {
  const select = document.getElementById("joint3d");
  if (!select) return;

  const current = select.value;
  select.innerHTML = "";

  for (const name of Object.keys(skeleton3D)) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    select.appendChild(option);
  }

  if (skeleton3D[current]) {
    select.value = current;
  }

  loadSelectedJoint3D();
}

function loadSelectedJoint3D() {
  const select = document.getElementById("joint3d");
  const posX = document.getElementById("posX");
  const posY = document.getElementById("posY");
  const posZ = document.getElementById("posZ");
  if (!select || !posX || !posY || !posZ) return;

  const point = skeleton3D[select.value];
  if (!point) return;

  posX.value = point.x.toFixed(3);
  posY.value = point.y.toFixed(3);
  posZ.value = point.z.toFixed(3);
}

function applyJoint3DPosition() {
  const select = document.getElementById("joint3d");
  if (!select || !select.value) return;

  skeleton3D[select.value] = {
    x: parseFloat(document.getElementById("posX").value) || 0,
    y: parseFloat(document.getElementById("posY").value) || 0,
    z: parseFloat(document.getElementById("posZ").value) || 0
  };

  updateSkeleton3DPositions();
  setManual3DStatus("Ponto " + select.value + " atualizado.");
}

function setManual3DStatus(message) {
  const el = document.getElementById("manual3dStatus");
  if (el) el.textContent = message;
}

async function saveSkeleton3D() {
  setManual3DStatus("Salvando esqueleto 3D...");

  try {
    const response = await fetch(`${API_BASE}/manual3d/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        post_id: currentReviewItem ? currentReviewItem.post_id : null,
        keypoints_3d: skeleton3D,
        tags: ["manual_3d"]
      })
    });

    const result = await response.json();

    if (!response.ok || !result.success) {
      throw new Error(result.error || "Erro ao salvar.");
    }

    setManual3DStatus("Esqueleto 3D salvo com sucesso! ID: " + result.pose_id);

  } catch (error) {
    console.error(error);
    setManual3DStatus("Erro ao salvar: " + error.message);
  }
}

function setupManual3DPanel() {
  const select = document.getElementById("joint3d");
  if (select) select.addEventListener("change", loadSelectedJoint3D);

  const applyBtn = document.getElementById("apply3d");
  if (applyBtn) applyBtn.addEventListener("click", applyJoint3DPosition);

  const saveBtn = document.getElementById("save3d");
  if (saveBtn) saveBtn.addEventListener("click", saveSkeleton3D);
}


function setupReviewButtons() {
  const approve = document.getElementById("btn-approve");
  if (approve) approve.addEventListener("click", () => submitReview("approve"));

  const reject = document.getElementById("btn-reject");
  if (reject) reject.addEventListener("click", () => submitReview("reject"));

  const next = document.getElementById("btn-next");
  if (next) next.addEventListener("click", loadNextReviewItem);
}


// ------------------------------------------------------------
// Dashboard (index.html): iniciar/pausar/retomar coleta e stats
//
// Faltava inteiramente — o botão "Iniciar" só dava submit no
// formulário sem nenhum JS ouvindo, então nada era enviado pra
// /api/collect/start e a tela nunca refletia o que a coleta
// estava fazendo no backend.
// ------------------------------------------------------------

const COLLECT_STATUS_LABELS = {
  stopped: "parado",
  running: "coletando...",
  paused: "pausado",
  finished: "concluído",
  error: "erro"
};

let dashboardPollHandle = null;

function getCollectElements() {
  return {
    form: document.getElementById("collect-form"),
    tagsInput: document.getElementById("tags"),
    pauseBtn: document.getElementById("btn-pause"),
    resumeBtn: document.getElementById("btn-resume"),
    statusEl: document.getElementById("collect-status"),
    barEl: document.getElementById("collect-bar"),
    errorsEl: document.getElementById("collect-errors")
  };
}

function renderCollectStatus(status) {
  if (!status) return;

  const { statusEl, barEl, errorsEl } = getCollectElements();

  if (statusEl) {
    const label = COLLECT_STATUS_LABELS[status.status] || status.status;
    const limitPart = status.limit > 0 ? ` de ${status.limit}` : "";
    statusEl.textContent =
      `${label} — ${status.downloaded}/${status.collected}${limitPart} baixadas`;
  }

  if (barEl) {
    if (status.limit > 0) {
      barEl.max = status.limit;
      barEl.value = status.collected;
    } else {
      // Sem limite definido não dá pra saber o total: barra
      // fica indeterminada (sem "value") em vez de travada em 0.
      barEl.removeAttribute("value");
    }
  }

  if (errorsEl) {
    errorsEl.innerHTML = "";
    if (status.errors > 0) {
      const li = document.createElement("li");
      li.textContent = `${status.errors} erro(s) durante a coleta.`;
      errorsEl.appendChild(li);
    }
  }
}

function renderDashboardStats(stats) {
  const values = {
    "stat-collected": stats?.posts?.total ?? 0,
    "stat-processed": stats?.posts?.by_status?.processed ?? 0,
    "stat-auto-approved": stats?.skeletons?.by_review_status?.auto_approved ?? 0,
    "stat-review": stats?.skeletons?.by_review_status?.pending_review ?? 0,
    "stat-rejected": stats?.skeletons?.by_review_status?.rejected ?? 0,
    "stat-skeletons": stats?.skeletons?.total ?? 0
  };

  for (const [id, value] of Object.entries(values)) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  }
}

async function refreshDashboard() {
  try {
    const [statusRes, statsRes] = await Promise.all([
      fetch(`${API_BASE}/collect/status`, { cache: "no-store" }),
      fetch(`${API_BASE}/stats`, { cache: "no-store" })
    ]);

    if (statusRes.ok) {
      renderCollectStatus(await statusRes.json());
    }

    if (statsRes.ok) {
      renderDashboardStats(await statsRes.json());
    }
  } catch (error) {
    console.error("Erro ao atualizar dashboard:", error);
  }
}

function setupCollectForm() {
  const { form, tagsInput } = getCollectElements();
  if (!form) return;

  form.addEventListener("submit", async (evt) => {
    evt.preventDefault();

    const tags = (tagsInput?.value || "").trim();
    if (!tags) {
      window.alert("Informe ao menos uma tag antes de iniciar.");
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/collect/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tags })
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        window.alert(data.error || "Não foi possível iniciar a coleta.");
        return;
      }

      renderCollectStatus(data.status);
    } catch (error) {
      console.error(error);
      window.alert("Erro ao iniciar a coleta: " + error.message);
    }
  });
}

function setupCollectButtons() {
  const { pauseBtn, resumeBtn } = getCollectElements();

  async function callAndRender(endpoint) {
    try {
      const response = await fetch(`${API_BASE}/collect/${endpoint}`, { method: "POST" });
      const data = await response.json();
      renderCollectStatus(data.status);
    } catch (error) {
      console.error(`Erro ao chamar /collect/${endpoint}:`, error);
    }
  }

  if (pauseBtn) pauseBtn.addEventListener("click", () => callAndRender("pause"));
  if (resumeBtn) resumeBtn.addEventListener("click", () => callAndRender("resume"));
}

function setupCleanupButton() {
  const btn = document.getElementById("btn-cleanup-images");
  const resultEl = document.getElementById("cleanup-result");
  if (!btn) return;

  btn.addEventListener("click", async () => {
    btn.disabled = true;
    if (resultEl) resultEl.textContent = "Verificando posts...";

    try {
      const response = await fetch(`${API_BASE}/maintenance/cleanup-missing-images`, {
        method: "POST"
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || "Falha ao corrigir imagens ausentes.");
      }

      if (resultEl) {
        resultEl.textContent =
          `${data.checked} post(s) verificados, ${data.fixed} corrigido(s).`;
      }

      await refreshDashboard();

    } catch (error) {
      console.error(error);
      if (resultEl) resultEl.textContent = "Erro: " + error.message;
    } finally {
      btn.disabled = false;
    }
  });
}


document.addEventListener("DOMContentLoaded", async () => {
  const isDashboardPage = !!document.getElementById("collect-form");
  const isReviewPage = !!getPoseCanvas();

  if (isDashboardPage) {
    setupCollectForm();
    setupCollectButtons();
    setupCleanupButton();

    await refreshDashboard();

    if (dashboardPollHandle) clearInterval(dashboardPollHandle);
    dashboardPollHandle = setInterval(refreshDashboard, 2000);
  }

  if (isReviewPage) {
    setupReviewButtons();
    setupEditTools();
    setupManual3DPanel();
    await initSkeleton3DViewer();

    try {
      await initPoseDetector();
      await loadNextReviewItem();
    } catch (error) {
      console.error(error);
      showReviewError(error.message);
    }
  }
});