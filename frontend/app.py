from __future__ import annotations

import base64
import sys
import time
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import torch
from flask import Flask, render_template, request
from PIL import Image
from torchvision import transforms

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.models import create_resnet50_multilabel


CHECKPOINT_PATH = ROOT_DIR / "outputs" / "checkpoints" / "best_resnet50.pt"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ATTRIBUTE_TRANSLATIONS = {
    "print": "estampado",
    "printed": "com estampado",
    "graphic": "gráfico",
    "abstract": "abstrato",
    "floral": "floral",
    "floral print": "estampado floral",
    "striped": "às riscas",
    "stripe": "risca",
    "plaid": "xadrez",
    "polka dot": "bolinhas",
    "paisley": "paisley",
    "tribal": "tribal",
    "lace": "renda",
    "mesh": "rede",
    "sheer": "transparente",
    "crochet": "croché",
    "knit": "malha",
    "woven": "tecido",
    "cotton": "algodão",
    "denim": "ganga",
    "leather": "couro",
    "faux leather": "pele sintética",
    "linen": "linho",
    "chiffon": "chiffon",
    "sleeve": "manga",
    "sleeveless": "sem mangas",
    "long sleeve": "manga comprida",
    "v-neck": "gola em V",
    "crew neck": "gola redonda",
    "collar": "colarinho",
    "shirt": "camisa",
    "hooded": "com capuz",
    "crop": "curto",
    "mini": "mini",
    "midi": "midi",
    "maxi": "maxi",
    "bodycon": "justo ao corpo",
    "skater": "evasé",
    "shift": "corte direito",
    "a-line": "corte em A",
    "flare": "evasé",
    "strapless": "sem alças",
    "button": "botões",
    "pocket": "bolso",
    "drawstring": "cordão ajustável",
    "pleated": "plissado",
    "embroidered": "bordado",
    "wash": "lavagem",
    "red": "vermelho",
    "pink": "rosa",
}

PORTUGUESE_ACCENT_FIXES = {
    "grafico": "gráfico",
    "as riscas": "às riscas",
    "croche": "croché",
    "algodao": "algodão",
    "pele sintetica": "pele sintética",
    "evase": "evasé",
    "sem alcas": "sem alças",
    "botoes": "botões",
    "cordao ajustavel": "cordão ajustável",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


def allowed_file(filename: str) -> bool:
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return extension in ALLOWED_EXTENSIONS


def clean_attr_name(attr: str) -> str:
    attr = attr.replace("attr_", "").replace("_", " ")
    translated = ATTRIBUTE_TRANSLATIONS.get(attr, attr)
    return PORTUGUESE_ACCENT_FIXES.get(translated, translated)


def confidence_badge(probability: float) -> dict[str, str]:
    if probability > 85:
        return {"label": "Muito elevada", "class_name": "high"}
    if probability >= 60:
        return {"label": "Média", "class_name": "medium"}
    return {"label": "Baixa", "class_name": "low"}


def build_inference_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


@lru_cache(maxsize=1)
def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    attr_columns = checkpoint["attr_columns"]

    model = create_resnet50_multilabel(
        num_labels=len(attr_columns),
        pretrained=False,
        dropout=0.2,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, attr_columns


def image_to_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def predict(image: Image.Image, threshold: float, image_size: int = 224):
    transform = build_inference_transform(image_size)
    tensor = transform(image).unsqueeze(0)
    model, attr_columns = load_model()

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0)

    results = []
    for attr, prob in zip(attr_columns, probs):
        probability = float(prob)
        if probability >= threshold:
            percent = round(probability * 100, 2)
            results.append(
                {
                    "name": clean_attr_name(attr),
                    "probability": percent,
                    "confidence": confidence_badge(percent),
                }
            )

    return sorted(results, key=lambda item: item["probability"], reverse=True)


@app.get("/")
def index():
    return render_template(
        "index.html",
        checkpoint_exists=CHECKPOINT_PATH.exists(),
        threshold=0.5,
    )


@app.post("/predict")
def predict_route():
    threshold = float(request.form.get("threshold", 0.5))
    threshold = max(0.1, min(threshold, 0.9))
    image_file = request.files.get("image")

    if image_file is None or image_file.filename == "":
        return render_template(
            "index.html",
            error="Escolhe uma imagem para analisar.",
            checkpoint_exists=CHECKPOINT_PATH.exists(),
            threshold=threshold,
        )

    if not allowed_file(image_file.filename):
        return render_template(
            "index.html",
            error="Formato inválido. Usa JPG, PNG ou WEBP.",
            checkpoint_exists=CHECKPOINT_PATH.exists(),
            threshold=threshold,
        )

    try:
        image = Image.open(image_file).convert("RGB")
        image_width, image_height = image.size
        started_at = time.perf_counter()
        results = predict(image=image, threshold=threshold)
        elapsed_ms = round((time.perf_counter() - started_at) * 1000)
    except Exception as exc:
        return render_template(
            "index.html",
            error=str(exc),
            checkpoint_exists=CHECKPOINT_PATH.exists(),
            threshold=threshold,
        )

    return render_template(
        "index.html",
        checkpoint_exists=CHECKPOINT_PATH.exists(),
        elapsed_ms=elapsed_ms,
        file_info={
            "name": Path(image_file.filename).name,
            "resolution": f"{image_width}x{image_height}",
        },
        image_preview=image_to_data_url(image),
        result_count=len(results),
        results=results,
        threshold=threshold,
        top_results=results[:3],
    )


@app.get("/health")
def health():
    return {"checkpoint_exists": CHECKPOINT_PATH.exists(), "status": "ok"}


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
