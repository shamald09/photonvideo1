let fileInput = document.getElementById("fileElem");
let dropArea = document.getElementById("drop-area");
let preview = document.getElementById("preview");
let previewVideo = document.getElementById("preview-video");
let dropEmpty = document.getElementById("drop-empty");
let dropText = document.getElementById("drop-text");
let dropHint = document.getElementById("drop-hint");
let scanLine = document.getElementById("scan-line");
let analyzeBtn = document.getElementById("analyze-btn");
let btnLabel = document.getElementById("btn-label");
let modelLabel = document.getElementById("model-label");

let modeImageBtn = document.getElementById("mode-image-btn");
let modeVideoBtn = document.getElementById("mode-video-btn");

let idleState = document.getElementById("idle-state");
let resultState = document.getElementById("result-state");

let currentMode = "image"; // "image" | "video"

dropArea.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", handleFiles);

dropArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropArea.classList.add("drag-active");
});

dropArea.addEventListener("dragleave", () => {
  dropArea.classList.remove("drag-active");
});

dropArea.addEventListener("drop", (e) => {
  e.preventDefault();
  dropArea.classList.remove("drag-active");
  fileInput.files = e.dataTransfer.files;
  handleFiles();
});

function setMode(mode) {
  currentMode = mode;

  // reset any preview/result state when switching modes
  fileInput.value = "";
  preview.style.display = "none";
  previewVideo.classList.add("hidden");
  previewVideo.pause();
  previewVideo.removeAttribute("src");
  dropEmpty.style.display = "flex";
  resultState.classList.add("hidden");
  idleState.classList.remove("hidden");

  if (mode === "image") {
    modeImageBtn.classList.add("active");
    modeVideoBtn.classList.remove("active");
    fileInput.setAttribute("accept", "image/*");
    dropText.innerText = "Drop image or click to browse";
    dropHint.innerText = "JPG · PNG · WEBP";
    btnLabel.innerText = "Analyze image";
    modelLabel.innerText = "Model: EfficientNet-B4";
  } else {
    modeVideoBtn.classList.add("active");
    modeImageBtn.classList.remove("active");
    fileInput.setAttribute("accept", "video/*");
    dropText.innerText = "Drop video or click to browse";
    dropHint.innerText = "MP4 · MOV · WEBM";
    btnLabel.innerText = "Analyze video";
    modelLabel.innerText = "Model: Xception + Wav2Vec2";
  }
}

function handleFiles() {
  let file = fileInput.files[0];
  if (!file) return;

  let url = URL.createObjectURL(file);

  if (currentMode === "video") {
    previewVideo.src = url;
    previewVideo.classList.remove("hidden");
    preview.style.display = "none";
  } else {
    preview.src = url;
    preview.style.display = "block";
    previewVideo.classList.add("hidden");
  }

  dropEmpty.style.display = "none";

  // reset previous result when a new file is chosen
  resultState.classList.add("hidden");
  idleState.classList.remove("hidden");
}

function handleAnalyzeClick() {
  if (currentMode === "video") {
    uploadVideo();
  } else {
    uploadImage();
  }
}

function uploadImage() {
  let file = fileInput.files[0];
  if (!file) {
    alert("Please select an image first.");
    return;
  }

  let formData = new FormData();
  formData.append("file", file);

  analyzeBtn.disabled = true;
  btnLabel.innerText = "Analyzing...";
  scanLine.classList.remove("hidden");
  idleState.classList.add("hidden");
  resultState.classList.add("hidden");

  fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    body: formData
  })
  .then(res => res.json())
  .then(data => {
    scanLine.classList.add("hidden");
    analyzeBtn.disabled = false;
    btnLabel.innerText = "Analyze image";

    if (data.error) {
      alert(data.error);
      idleState.classList.remove("hidden");
      return;
    }

    renderResult(data.result, data.confidence);
  })
  .catch(err => {
    scanLine.classList.add("hidden");
    analyzeBtn.disabled = false;
    btnLabel.innerText = "Analyze image";
    idleState.classList.remove("hidden");
    alert("Error connecting to server");
    console.error(err);
  });
}

function uploadVideo() {
  let file = fileInput.files[0];
  if (!file) {
    alert("Please select a video first.");
    return;
  }

  let formData = new FormData();
  formData.append("file", file);

  analyzeBtn.disabled = true;
  btnLabel.innerText = "Analyzing...";
  scanLine.classList.remove("hidden");
  idleState.classList.add("hidden");
  resultState.classList.add("hidden");

  fetch("http://127.0.0.1:5000/predict_video", {
    method: "POST",
    body: formData
  })
  .then(res => res.json())
  .then(data => {
    scanLine.classList.add("hidden");
    analyzeBtn.disabled = false;
    btnLabel.innerText = "Analyze video";

    if (data.error) {
      alert(data.error);
      idleState.classList.remove("hidden");
      return;
    }

    renderResult(data.result, data.confidence);
  })
  .catch(err => {
    scanLine.classList.add("hidden");
    analyzeBtn.disabled = false;
    btnLabel.innerText = "Analyze video";
    idleState.classList.remove("hidden");
    alert("Error connecting to server");
    console.error(err);
  });
}

function renderResult(resultText, confidence) {
  let pct = (confidence * 100).toFixed(1);
  let isReal = resultText.toLowerCase().includes("real");

  document.getElementById("result-text").innerText = resultText;
  document.getElementById("confidence-text").innerText = pct + "%";

  let dot = document.getElementById("verdict-dot");
  let bar = document.getElementById("bar-fill");
  let color = isReal ? "var(--verdict-real)" : "var(--verdict-ai)";

  dot.style.background = color;
  bar.style.background = color;
  bar.style.width = pct + "%";

  resultState.classList.remove("hidden");
}