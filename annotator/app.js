const state = {
  activeBox: "left",
  boxes: {
    left: null,
    right: null,
  },
  draft: null,
  imageWidth: 0,
  imageHeight: 0,
};

const elements = {
  screenshot: document.querySelector("#screenshot"),
  overlay: document.querySelector("#overlay"),
  stage: document.querySelector("#stage"),
  leftButton: document.querySelector("#left-button"),
  rightButton: document.querySelector("#right-button"),
  clearLeft: document.querySelector("#clear-left"),
  clearRight: document.querySelector("#clear-right"),
  saveButton: document.querySelector("#save-button"),
  imageName: document.querySelector("#image-name"),
  configPath: document.querySelector("#config-path"),
  leftCoords: document.querySelector("#left-coords"),
  rightCoords: document.querySelector("#right-coords"),
  message: document.querySelector("#message"),
};

let context = null;

function setMessage(text, tone = "normal") {
  elements.message.textContent = text;
  elements.message.dataset.tone = tone;
}

function roundedBox(box) {
  return {
    x: Math.round(box.x),
    y: Math.round(box.y),
    width: Math.round(box.width),
    height: Math.round(box.height),
  };
}

function boxLabel(box) {
  if (!box) {
    return "Not set";
  }
  const clean = roundedBox(box);
  return `x=${clean.x}, y=${clean.y}, w=${clean.width}, h=${clean.height}`;
}

function updateCoords() {
  elements.leftCoords.textContent = boxLabel(state.boxes.left);
  elements.rightCoords.textContent = boxLabel(state.boxes.right);
}

function setActiveBox(next) {
  state.activeBox = next;
  elements.leftButton.classList.toggle("active", next === "left");
  elements.rightButton.classList.toggle("active", next === "right");
  setMessage(`Drawing ${next} crop box. Drag on the image.`, "normal");
  draw();
}

function syncCanvas() {
  const rect = elements.stage.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  elements.overlay.width = Math.round(rect.width * ratio);
  elements.overlay.height = Math.round(rect.height * ratio);
  const ctx = elements.overlay.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  draw();
}

function naturalToDisplay(box) {
  const imageRect = elements.screenshot.getBoundingClientRect();
  const scaleX = imageRect.width / state.imageWidth;
  const scaleY = imageRect.height / state.imageHeight;
  return {
    x: box.x * scaleX,
    y: box.y * scaleY,
    width: box.width * scaleX,
    height: box.height * scaleY,
  };
}

function displayToNatural(box) {
  const imageRect = elements.screenshot.getBoundingClientRect();
  const scaleX = state.imageWidth / imageRect.width;
  const scaleY = state.imageHeight / imageRect.height;
  return roundedBox({
    x: box.x * scaleX,
    y: box.y * scaleY,
    width: box.width * scaleX,
    height: box.height * scaleY,
  });
}

function pointerToImagePoint(event) {
  const imageRect = elements.screenshot.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(event.clientX - imageRect.left, imageRect.width)),
    y: Math.max(0, Math.min(event.clientY - imageRect.top, imageRect.height)),
  };
}

function normalizeDisplayBox(start, end) {
  return {
    x: Math.min(start.x, end.x),
    y: Math.min(start.y, end.y),
    width: Math.abs(end.x - start.x),
    height: Math.abs(end.y - start.y),
  };
}

function drawBox(ctx, box, color, label, dashed = false) {
  if (!box) {
    return;
  }
  const display = naturalToDisplay(box);
  ctx.save();
  ctx.lineWidth = 3;
  ctx.strokeStyle = color;
  ctx.fillStyle = `${color}20`;
  if (dashed) {
    ctx.setLineDash([8, 6]);
  }
  ctx.fillRect(display.x, display.y, display.width, display.height);
  ctx.strokeRect(display.x, display.y, display.width, display.height);
  ctx.fillStyle = color;
  ctx.font = "600 15px Iowan Old Style, Georgia, serif";
  ctx.fillText(label, display.x + 10, display.y + 22);
  ctx.restore();
}

function draw() {
  const ctx = elements.overlay.getContext("2d");
  if (!ctx) {
    return;
  }
  ctx.clearRect(0, 0, elements.overlay.width, elements.overlay.height);
  drawBox(ctx, state.boxes.left, "#1f6b84", "LEFT");
  drawBox(ctx, state.boxes.right, "#af5d36", "RIGHT");
  if (state.draft) {
    drawBox(ctx, displayToNatural(state.draft), "#111111", state.activeBox.toUpperCase(), true);
  }
}

function installPointerHandlers() {
  let dragStart = null;

  elements.overlay.addEventListener("pointerdown", (event) => {
    if (!state.imageWidth || !state.imageHeight) {
      return;
    }
    dragStart = pointerToImagePoint(event);
    state.draft = { x: dragStart.x, y: dragStart.y, width: 0, height: 0 };
    elements.overlay.setPointerCapture(event.pointerId);
    draw();
  });

  elements.overlay.addEventListener("pointermove", (event) => {
    if (!dragStart) {
      return;
    }
    const current = pointerToImagePoint(event);
    state.draft = normalizeDisplayBox(dragStart, current);
    draw();
  });

  elements.overlay.addEventListener("pointerup", (event) => {
    if (!dragStart || !state.draft) {
      return;
    }
    const current = pointerToImagePoint(event);
    const draft = normalizeDisplayBox(dragStart, current);
    if (draft.width >= 8 && draft.height >= 8) {
      state.boxes[state.activeBox] = displayToNatural(draft);
      updateCoords();
      setMessage(`Saved ${state.activeBox} box.`, "normal");
    }
    dragStart = null;
    state.draft = null;
    draw();
  });
}

async function saveConfig() {
  if (!state.boxes.left || !state.boxes.right) {
    setMessage("Both left and right boxes are required before saving.", "error");
    return;
  }

  const payload = {
    source_image: context.first_image.path,
    image_size: {
      width: state.imageWidth,
      height: state.imageHeight,
    },
    left: roundedBox(state.boxes.left),
    right: roundedBox(state.boxes.right),
    created_at: new Date().toISOString(),
  };

  const response = await fetch("/api/crop-config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json();
  if (!response.ok) {
    setMessage(body.error || "Failed to save crop config.", "error");
    return;
  }
  setMessage(`Saved crop config to ${body.path}.`, "success");
}

function loadSavedBoxes(savedConfig) {
  if (!savedConfig) {
    return;
  }
  state.boxes.left = savedConfig.left || null;
  state.boxes.right = savedConfig.right || null;
  updateCoords();
}

async function loadContext() {
  const response = await fetch("/api/context");
  context = await response.json();
  elements.imageName.textContent = context.first_image.name;
  elements.configPath.textContent = context.crop_config_path;
  elements.screenshot.src = context.first_image.url;
  loadSavedBoxes(context.crop_config);
}

async function main() {
  installPointerHandlers();
  elements.leftButton.addEventListener("click", () => setActiveBox("left"));
  elements.rightButton.addEventListener("click", () => setActiveBox("right"));
  elements.clearLeft.addEventListener("click", () => {
    state.boxes.left = null;
    updateCoords();
    draw();
  });
  elements.clearRight.addEventListener("click", () => {
    state.boxes.right = null;
    updateCoords();
    draw();
  });
  elements.saveButton.addEventListener("click", () => saveConfig());

  elements.screenshot.addEventListener("load", () => {
    state.imageWidth = elements.screenshot.naturalWidth;
    state.imageHeight = elements.screenshot.naturalHeight;
    syncCanvas();
  });

  new ResizeObserver(() => syncCanvas()).observe(elements.stage);

  await loadContext();
  updateCoords();
  setActiveBox("left");
}

main().catch((error) => {
  console.error(error);
  setMessage(error.message || "Failed to load annotator.", "error");
});
