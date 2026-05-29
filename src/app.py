from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models import create_resnet50_multilabel


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


def build_inference_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


@st.cache_resource
def load_model(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    attr_columns = checkpoint["attr_columns"]

    model = create_resnet50_multilabel(
        num_labels=len(attr_columns),
        pretrained=False,
        dropout=0.2,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, attr_columns


def clean_attr_name(attr: str) -> str:
    attr = attr.replace("attr_", "").replace("_", " ")
    return ATTRIBUTE_TRANSLATIONS.get(attr, attr)


def predict(
    image: Image.Image,
    model,
    attr_columns: list[str],
    threshold: float,
    image_size: int,
) -> list[dict[str, float | str]]:
    transform = build_inference_transform(image_size)
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0)

    results = []

    for attr, prob in zip(attr_columns, probs):
        probability = float(prob)

        if probability >= threshold:
            results.append(
                {
                    "Atributo": clean_attr_name(attr),
                    "Probabilidade (%)": round(probability * 100, 2),
                }
            )

    return sorted(results, key=lambda x: x["Probabilidade (%)"], reverse=True)


def main() -> None:
    st.set_page_config(
        page_title="Classificação de Atributos de Roupa",
        page_icon="👕",
        layout="centered",
    )

    st.title("Classificação de Atributos de Roupa")
    st.write(
        "Carrega uma imagem de uma peça de roupa para o modelo prever os seus atributos."
    )

    st.sidebar.header("Definições")

    checkpoint_path = st.sidebar.text_input(
        "Modelo treinado",
        value="outputs/checkpoints/best_resnet50.pt",
    )

    threshold = st.sidebar.slider(
        "Limite de confiança",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
    )

    image_size = st.sidebar.selectbox(
        "Tamanho da imagem",
        options=[224, 256, 384],
        index=0,
    )

    uploaded_file = st.file_uploader(
        "Escolhe uma imagem",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("Carrega uma imagem para começar.")
        return

    checkpoint = Path(checkpoint_path)

    if not checkpoint.exists():
        st.error(f"Modelo treinado não encontrado: {checkpoint_path}")
        return

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Imagem carregada",
        use_container_width=True,
    )

    with st.spinner("A analisar a imagem..."):
        model, attr_columns = load_model(str(checkpoint))

        results = predict(
            image=image,
            model=model,
            attr_columns=attr_columns,
            threshold=threshold,
            image_size=image_size,
        )

    st.subheader("Atributos previstos")

    if not results:
        st.warning("Nenhum atributo passou o limite de confiança escolhido.")
        return

    df = pd.DataFrame(results)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Top 10 atributos")

    top_df = df.head(10).set_index("Atributo")
    st.bar_chart(top_df)


if __name__ == "__main__":
    main()