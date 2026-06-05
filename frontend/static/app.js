const imageInput = document.querySelector("#imageInput");
const livePreview = document.querySelector("#livePreview");
const previewPlaceholder = document.querySelector("#previewPlaceholder");
const clearImageButton = document.querySelector("#clearImageButton");
const thresholdInput = document.querySelector("#thresholdInput");
const thresholdValue = document.querySelector("#thresholdValue");

function showPreview(src) {
  if (!livePreview) return;

  livePreview.src = src;
  livePreview.classList.add("visible");
  previewPlaceholder?.classList.add("hidden");
  clearImageButton?.classList.add("visible");
}

function clearPreview() {
  if (imageInput) {
    imageInput.value = "";
  }

  if (livePreview) {
    livePreview.removeAttribute("src");
    livePreview.classList.remove("visible");
  }

  previewPlaceholder?.classList.remove("hidden");
  clearImageButton?.classList.remove("visible");
}

if (imageInput && livePreview) {
  imageInput.addEventListener("change", () => {
    const file = imageInput.files?.[0];
    if (!file) return;

    showPreview(URL.createObjectURL(file));
  });
}

if (clearImageButton) {
  clearImageButton.addEventListener("click", () => {
    if (window.location.pathname === "/predict") {
      window.location.href = "/";
      return;
    }

    clearPreview();
  });
}

if (thresholdInput && thresholdValue) {
  thresholdInput.addEventListener("input", () => {
    thresholdValue.textContent = Math.round(Number(thresholdInput.value) * 100);
  });
}
