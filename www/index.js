// Setup canvas element and context variables
const canvas = document.getElementById("draw");
const drawCtx = canvas.getContext("2d", { willReadFrequently: true });

const scale = document.createElement('canvas');
scale.width = scale.height = 28;
const scaleCtx = scale.getContext("2d", { willReadFrequently: true });
//document.body.appendChild(scale);

// Canvas mouse move and touch controls
let isDown = false;

function handleDown(evt) {
  evt.preventDefault();
  isDown = true;
  drawCtx.lineWidth = 20;
  drawCtx.lineJoin = "round";
  drawCtx.lineCap = "round";
  drawCtx.fillStyle = "black";
  drawCtx.beginPath();
}

function handleMouseMove(evt) {
  var x = evt.offsetX != null ? evt.offsetX : evt.originalEvent.layerX;
  var y = evt.offsetY != null ? evt.offsetY : evt.originalEvent.layerY;
  if (isDown) {
    drawCtx.lineTo(x, y);
    drawCtx.stroke();
    do_classify();
  }
}

function handleTouchMove(evt) {
  evt.preventDefault();
  var rect = canvas.getBoundingClientRect();
  var x = evt.touches[0].clientX - rect.left;
  var y = evt.touches[0].clientY - rect.top;
  if (isDown) {
    drawCtx.lineTo(x, y);
    drawCtx.stroke();
    do_classify();
  }
}

function handleUp() {
  isDown = false;
  drawCtx.stroke();
}

function reset() {
  drawHist(new Array(10).fill(-Infinity));
  clearImage();
}

// Canvas manipulation routines
function clearImage() {
  drawCtx.fillStyle = "white";
  drawCtx.fillRect(0, 0, 280, 280);
}

function processImage() {
  const w = scaleCtx.canvas.width
  const h = scaleCtx.canvas.height;

  // Downscale for inference
  scaleCtx.drawImage(canvas, 0, 0, w, h);

  // Center and scale image data
  const imageData = scaleCtx.getImageData(0, 0, w, h);
  scaleCtx.fillStyle = "white";
  scaleCtx.fillRect(0, 0, w, h);

  let minx = Infinity;
  let maxx = 0;
  let miny = Infinity;
  let maxy = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (imageData.data[4 * (y * w + x)] < 255) {
        minx = Math.min(x - 5, minx);
        maxx = Math.max(x + 5, maxx);
        miny = Math.min(y - 5, miny);
        maxy = Math.max(y + 5, maxy);
      }
    }
  }

  // Redraw scaled image and return pixel data
  scaleCtx.drawImage(
    canvas,
    canvas.width * minx / w,
    canvas.height * miny / h,
    canvas.width * (maxx - minx) / w,
    canvas.height * (maxy - miny) / h,
    0, 0, w, h);
  return scaleCtx.getImageData(0, 0, w, h)
}

// Draw classifier result with Observable Plot
function drawHist(classify) {
  const t = 5.0;
  const maxC = classify.reduce((a, b) => a > b ? a : b);
  const plot = Plot.plot({
    marginTop: 50,
    width: 280,
    height: 200,
    axis: null,
    x: { axis: "bottom" },
    y: { domain: [0, 1] },
    marks: [
      Plot.text(['Digit Classification'], { frameAnchor: "Top", dy: -25 }),
      Plot.barY(classify.map((v) => Math.exp(v / t) / Math.exp(maxC / t)), { fill: 'SteelBlue' })
    ],
  });

  const div = document.querySelector("#hist");
  while (div.firstChild) {
    div.removeChild(div.lastChild);
  }
  div.append(plot);
}

// Download and initialise MNIST trained MLP weights
let weights_ptr = -1;
(async () => {
  const response = await fetch("./mnist.data");
  if (!response.ok) {
    throw new Error(`Problem downloading model weights (${response.status})`);
  }
  const bytes = await response.arrayBuffer();
  const floats = new Float64Array(bytes);
  weights_ptr = Module._malloc(floats.byteLength);
  Module.HEAPF64.set(floats, weights_ptr / 8);

  // Ready to start
  reset();
  canvas.addEventListener("mousedown", handleDown);
  canvas.addEventListener('touchstart', handleDown);
  canvas.addEventListener("mousemove", handleMouseMove);
  canvas.addEventListener('touchmove', handleTouchMove);
  document.addEventListener('touchend', handleUp);
  document.addEventListener("mouseup", handleUp);
  document.getElementById("clear").addEventListener('click', reset);
})().catch(err => {
  drawCtx.fillStyle = "red";
  drawCtx.font = "14px sans-serif bold";
  drawCtx.fillText("An error occurred:", 5, 100);
  drawCtx.fillText(err.message, 5, 120);
});

function do_classify() {
  if (weights_ptr < 0) {
    throw new Error("Can't classify image. Weights not downloaded yet.");
  }

  scaleData = processImage();

  // Copy image data to Wasm memory
  const image = new Float64Array(28 * 28);
  image.forEach((_, i) => {
    image[i] = 1. - scaleData.data[i * 4] / 255.;
  });
  const image_ptr = Module._malloc(28 * 28 * 8);
  Module.HEAPF64.set(image, image_ptr / 8);

  // Call Wasm classifier and store results
  const out_ptr = Module._malloc(10);
  Module._classifier_(weights_ptr, image_ptr, out_ptr);

  // Draw results as a histogram
  const out_bytes = Module.HEAPF64.subarray(out_ptr / 8, out_ptr / 8 + 10);
  drawHist(out_bytes);

  // Wasm memory cleanup
  Module._free(image_ptr);
  Module._free(out_ptr);
}

// Canvas inital display
reset();
drawCtx.fillStyle = "#EEE";
drawCtx.fillRect(0, 0, 280, 280);
drawCtx.fillStyle = "black";
drawCtx.font = "14px sans-serif";
drawCtx.fillText("Downloading Wasm and weights.", 5, 30);
drawCtx.fillText("Please wait...", 5, 50);
