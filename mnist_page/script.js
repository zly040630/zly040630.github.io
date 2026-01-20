// 画布参数设置
const CANVAS_SIZE = 280;
const CANVAS_SCALE = 0.5;

// 获取画布元素和上下文（context）
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

// 获取清空按钮
const clearButton = document.getElementById("clear-button");

// 鼠标事件处理
let isMouseDown = false;
let hasIntroText = true;
let lastX = 0;
let lastY = 0;

// 加载模型
let session = null;
const loadingModelPromise = ort.InferenceSession.create("./cnn.onnx").then(
  (createdSession) => {
    session = createdSession;
  }
);

// 初始化画图设置
ctx.lineWidth = 28;
ctx.lineJoin = "round";
ctx.font = "28px sans-serif";
ctx.textAlign = "center";
ctx.textBaseline = "middle";
ctx.fillStyle = "#212121";
// ctx.fillText("Draw a number here!", CANVAS_SIZE / 2, CANVAS_SIZE / 2);
ctx.fillText("Loading...", CANVAS_SIZE / 2, CANVAS_SIZE / 2);

// Set the line color for the canvas.
ctx.strokeStyle = "#212121";

// 清空画图区和预测结果
function clearCanvas() {
  ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  for (let i = 0; i < 10; i++) {
    const element = document.getElementById(`prediction-${i}`);
    element.className = "prediction-col";
    element.children[0].children[0].style.height = "0";
  }
}


// 绘制线，更新预测结果
function drawLine(fromX, fromY, toX, toY) {
  // Draws a line from (fromX, fromY) to (toX, toY).
  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.closePath();
  ctx.stroke();
  updatePredictions();
}

// 获取输入张量，输出一个onnx模型需要的Tensor格式
function getInputTensor() {
  const targetSize = 28;
  const scaleCanvas = document.createElement("canvas");
  scaleCanvas.width = targetSize;
  scaleCanvas.height = targetSize;
  const scaleCtx = scaleCanvas.getContext("2d");
  scaleCtx.fillStyle = "#fff";
  scaleCtx.fillRect(0, 0, targetSize, targetSize);
  scaleCtx.drawImage(canvas, 0, 0, targetSize, targetSize);

  const imgData = scaleCtx.getImageData(0, 0, targetSize, targetSize).data;
  const inputData = new Float32Array(targetSize * targetSize);
  for (let i = 0; i < targetSize * targetSize; i++) {
    const r = imgData[i * 4];
    const g = imgData[i * 4 + 1];
    const b = imgData[i * 4 + 2];
    const gray = (r + g + b) / 3;
    const x = (255 - gray) / 255;
    inputData[i] = (x - 0.5) / 0.5;
  }

  return new ort.Tensor("float32", inputData, [1, 1, targetSize, targetSize]);
}

// 更新预测结果
async function updatePredictions() {
  //  获取输入数据的Tensor
  const inputTensor = getInputTensor();
  const feeds = {};
  feeds[session.inputNames[0]] = inputTensor;

  // 运行模型，获取输出结果
  const outputMap = await session.run(feeds);
  const outputTensor = outputMap[session.outputNames[0]];
  const predictions = outputTensor.data;
  const maxPrediction = Math.max(...predictions);

  // 更新预测结果的显示
  for (let i = 0; i < predictions.length; i++) {
    const element = document.getElementById(`prediction-${i}`);
    element.children[0].children[0].style.height = `${predictions[i] * 100}%`;
    element.className =
      predictions[i] === maxPrediction
        ? "prediction-col top-prediction"
        : "prediction-col";
  }
}


// 鼠标按下事件：开始绘图
function canvasMouseDown(event) {
  isMouseDown = true;
  if (hasIntroText) {
    clearCanvas();
    hasIntroText = false;
  }
  const x = event.offsetX / CANVAS_SCALE;
  const y = event.offsetY / CANVAS_SCALE;

  // To draw a dot on the mouse down event, we set laxtX and lastY to be
  // slightly offset from x and y, and then we call `canvasMouseMove(event)`,
  // which draws a line from (laxtX, lastY) to (x, y) that shows up as a
  // dot because the difference between those points is so small. However,
  // if the points were the same, nothing would be drawn, which is why the
  // 0.001 offset is added.
  lastX = x + 0.001;
  lastY = y + 0.001;
  canvasMouseMove(event);
}

// 鼠标移动事件：绘制线条
function canvasMouseMove(event) {
  const x = event.offsetX / CANVAS_SCALE;
  const y = event.offsetY / CANVAS_SCALE;
  if (isMouseDown) {
    drawLine(lastX, lastY, x, y);
  }
  lastX = x;
  lastY = y;
}

// 鼠标抬起事件：停止绘制
function bodyMouseUp() {
  isMouseDown = false;
}

// 鼠标离开画布事件：停止绘制
function bodyMouseOut(event) {
  // We won't be able to detect a MouseUp event if the mouse has moved
  // ouside the window, so when the mouse leaves the window, we set
  // `isMouseDown` to false automatically. This prevents lines from
  // continuing to be drawn when the mouse returns to the canvas after
  // having been released outside the window.
  // 当鼠标移出浏览器窗口时，强制认为“鼠标已经松开”，避免回到 canvas 后继续画线。
  if (!event.relatedTarget || event.relatedTarget.nodeName === "HTML") {
    isMouseDown = false;
  }
}

// 加载模型后，绑定事件
loadingModelPromise.then(() => {
  canvas.addEventListener("mousedown", canvasMouseDown);
  canvas.addEventListener("mousemove", canvasMouseMove);
  document.body.addEventListener("mouseup", bodyMouseUp);
  document.body.addEventListener("mouseout", bodyMouseOut);
  clearButton.addEventListener("mousedown", clearCanvas);

  ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  ctx.fillText("Draw a number here!", CANVAS_SIZE / 2, CANVAS_SIZE / 2);
})
