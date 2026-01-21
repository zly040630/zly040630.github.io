// =====================
// Canvas / UI 参数
// =====================
const CANVAS_SIZE = 280;
const CANVAS_SCALE = 0.5;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearButton = document.getElementById("clear-button");

let isMouseDown = false;
let hasIntroText = true;
let lastX = 0;
let lastY = 0;

// 初始化画笔
ctx.lineWidth = 28;
ctx.lineJoin = "round";
ctx.font = "28px sans-serif";
ctx.textAlign = "center";
ctx.textBaseline = "middle";
ctx.fillStyle = "#212121";
ctx.strokeStyle = "#212121";
ctx.fillText("Loading...", CANVAS_SIZE / 2, CANVAS_SIZE / 2);

// =====================
// 多模型配置
// =====================
// prob=true  : 模型输出已经是概率（softmax 后）
// prob=false : 模型输出是 logits，需要前端 softmax
const MODELS = [
  { key: "100",   title: "100 samples",   path: "./cnn.onnx",              prob: true,  session: null },
  { key: "600",   title: "600 samples",   path: "./cnn_mnist_600.onnx",    prob: false, session: null },
  { key: "6000",  title: "6000 samples",  path: "./cnn_mnist_6000.onnx",   prob: false, session: null },
  { key: "60000", title: "60000 samples", path: "./cnn_mnist_60000.onnx",  prob: false, session: null },
];

// 加载全部模型
const loadingModelPromise = Promise.all(
  MODELS.map(async (m) => {
    m.session = await ort.InferenceSession.create(m.path);
  })
);

// =====================
// 工具函数：softmax（logits -> prob）
// 数值稳定版：先减 max
// =====================
function softmax(arr) {
  let maxVal = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > maxVal) maxVal = arr[i];
  }

  const exps = new Float32Array(arr.length);
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    const e = Math.exp(arr[i] - maxVal);
    exps[i] = e;
    sum += e;
  }

  const probs = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    probs[i] = exps[i] / sum;
  }
  return probs;
}

// =====================
// 清空：画布 + 四组预测条
// =====================
function clearCanvas() {
  ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

  for (const m of MODELS) {
    for (let i = 0; i < 10; i++) {
      const element = document.getElementById(`prediction-${m.key}-${i}`);
      if (!element) continue;
      element.className = "prediction-col";
      element.children[0].children[0].style.height = "0";
    }
  }
}

// =====================
// 画线 + 触发预测更新
// =====================
function drawLine(fromX, fromY, toX, toY) {
  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.closePath();
  ctx.stroke();
  updatePredictions();
}

// =====================
// 生成 ONNX 输入 Tensor
// =====================
function getInputTensor() {
  const targetSize = 28;
  const scaleCanvas = document.createElement("canvas");
  scaleCanvas.width = targetSize;
  scaleCanvas.height = targetSize;
  const scaleCtx = scaleCanvas.getContext("2d");

  // 白底
  scaleCtx.fillStyle = "#fff";
  scaleCtx.fillRect(0, 0, targetSize, targetSize);

  // 缩放绘制
  scaleCtx.drawImage(canvas, 0, 0, targetSize, targetSize);

  // 读取像素 -> 灰度 -> 反色 -> Normalize 到 [-1,1]
  const imgData = scaleCtx.getImageData(0, 0, targetSize, targetSize).data;
  const inputData = new Float32Array(targetSize * targetSize);

  for (let i = 0; i < targetSize * targetSize; i++) {
    const r = imgData[i * 4];
    const g = imgData[i * 4 + 1];
    const b = imgData[i * 4 + 2];
    const gray = (r + g + b) / 3;
    const x = (255 - gray) / 255;     // 黑字白底 -> 字越黑值越大
    inputData[i] = (x - 0.5) / 0.5;   // Normalize 到 [-1,1]
  }

  return new ort.Tensor("float32", inputData, [1, 1, targetSize, targetSize]);
}

// =====================
// 更新四组预测结果
// =====================
async function updatePredictions() {
  // 模型没加载完就不推理
  if (MODELS.some((m) => !m.session)) return;

  const inputTensor = getInputTensor();

  for (const m of MODELS) {
    const session = m.session;

    const feeds = {};
    feeds[session.inputNames[0]] = inputTensor;

    const outputMap = await session.run(feeds);
    const outputTensor = outputMap[session.outputNames[0]];
    let out = outputTensor.data; // 可能是 logits 或 prob

    // 如果输出是 logits，则 softmax
    const probs = m.prob ? out : softmax(out);

    // 找 top1（用于高亮）
    let maxVal = -Infinity;
    for (let i = 0; i < probs.length; i++) {
      if (probs[i] > maxVal) maxVal = probs[i];
    }

    // 写入对应预测区
    for (let i = 0; i < probs.length; i++) {
      const element = document.getElementById(`prediction-${m.key}-${i}`);
      if (!element) continue;

      element.children[0].children[0].style.height = `${probs[i] * 100}%`;
      element.className =
        probs[i] === maxVal
          ? "prediction-col top-prediction"
          : "prediction-col";
    }
  }
}

// =====================
// 鼠标事件
// =====================
function canvasMouseDown(event) {
  isMouseDown = true;
  if (hasIntroText) {
    clearCanvas();
    hasIntroText = false;
  }

  const x = event.offsetX / CANVAS_SCALE;
  const y = event.offsetY / CANVAS_SCALE;

  // 画一个点：微小偏移 + 触发 mousemove
  lastX = x + 0.001;
  lastY = y + 0.001;
  canvasMouseMove(event);
}

function canvasMouseMove(event) {
  const x = event.offsetX / CANVAS_SCALE;
  const y = event.offsetY / CANVAS_SCALE;

  if (isMouseDown) {
    drawLine(lastX, lastY, x, y);
  }

  lastX = x;
  lastY = y;
}

function bodyMouseUp() {
  isMouseDown = false;
}

function bodyMouseOut(event) {
  // 鼠标出窗口：强制停止绘制
  if (!event.relatedTarget || event.relatedTarget.nodeName === "HTML") {
    isMouseDown = false;
  }
}

// =====================
// 模型加载完成后：绑定事件 & 显示提示
// =====================
loadingModelPromise
  .then(() => {
    canvas.addEventListener("mousedown", canvasMouseDown);
    canvas.addEventListener("mousemove", canvasMouseMove);
    document.body.addEventListener("mouseup", bodyMouseUp);
    document.body.addEventListener("mouseout", bodyMouseOut);
    clearButton.addEventListener("mousedown", clearCanvas);

    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctx.fillText("Draw a number here!", CANVAS_SIZE / 2, CANVAS_SIZE / 2);
  })
  .catch((err) => {
    console.error("Model load failed:", err);
    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctx.fillText("Model load failed", CANVAS_SIZE / 2, CANVAS_SIZE / 2);
  });
