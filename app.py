from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

from goddess_information import GODDESS_INFO, find_goddess_image


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Goddess_Split"
MODEL_PATH = DATASET_DIR / "goddess_material_resnet18_v3_best.pth"
IMAGE_SIZE = 224
RECOGNITION_THRESHOLD = 0.70
BEST_GODDESS_TEMP = 1.0
BEST_MATERIAL_TEMP = 1.2
CENTROID_THRESHOLD = 0.60

GODDESSES = [
    "Annapurna",
    "Durga",
    "Kali",
    "Lakshmi",
    "Lalitha",
    "Saraswati",
]
MATERIALS = ["Brass", "Bronze", "Copper", "Silver", "Stone", "Wood"]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

st.set_page_config(
    page_title="Divine Vision | Goddess Idol Recognition",
    page_icon="✦",
    layout="wide",
)
st.markdown(
    """
    <style>
    :root {
        --dv-white: #ffffff;
        --dv-soft: #fafafa;
        --dv-ink: #222222;
        --dv-muted: #707078;
        --dv-lavender: #8b7cf6;
        --dv-lavender-soft: #f3f1ff;
        --dv-line: #eaeaea;
    }
    html, body, [class*="css"] {
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: var(--dv-white);
        color: var(--dv-ink);
    }
    [data-testid="stHeader"] { background: rgba(255, 255, 255, .92); }
    [data-testid="stToolbar"] { right: 1rem; }
    .main .block-container {
        max-width: 1160px;
        padding: 1.5rem 2rem 1.25rem;
    }
    h1, h2, h3, h4 {
        color: var(--dv-ink);
        letter-spacing: -.035em;
        font-weight: 650;
    }
    h1 { font-size: clamp(2rem, 4vw, 3.25rem); line-height: 1.12; }
    h2 { font-size: clamp(1.45rem, 2.5vw, 1.9rem); }
    h3 { font-size: 1.2rem; }
    p, li { color: var(--dv-muted); line-height: 1.7; }
    [data-testid="stCaptionContainer"], [data-testid="stMetricLabel"] {
        color: var(--dv-muted);
    }
    [data-testid="stMarkdownContainer"] a {
        color: var(--dv-lavender);
        text-decoration: none;
        font-weight: 600;
    }
    [data-testid="stMarkdownContainer"] a:hover { text-decoration: underline; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--dv-white);
        border: 1px solid var(--dv-line);
        border-radius: 14px;
        box-shadow: none;
        transition: border-color 160ms ease, transform 160ms ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #d8d4f7;
    }
    [data-testid="stImage"] img {
        max-width: 100%;
        aspect-ratio: 4 / 3;
        object-fit: contain;
        background: var(--dv-soft);
        border: 1px solid var(--dv-line);
        border-radius: 10px;
    }
    [data-testid="stMetric"] {
        background: var(--dv-soft);
        border: 1px solid var(--dv-line);
        padding: .8rem 1rem;
        border-radius: 12px;
    }
    [data-testid="stMetricValue"] { color: var(--dv-ink); font-weight: 650; }
    .stButton > button {
        min-height: 2.8rem;
        border: 1px solid var(--dv-line);
        border-radius: 11px;
        color: var(--dv-ink);
        background: var(--dv-white);
        font-weight: 600;
        transition: background-color 160ms ease, border-color 160ms ease, transform 160ms ease;
    }
    .stButton > button:hover {
        background: var(--dv-soft);
        border-color: var(--dv-lavender);
        color: var(--dv-ink);
        transform: translateY(-1px);
    }
    .stButton > button[kind="primary"] {
        background: var(--dv-lavender);
        border-color: var(--dv-lavender);
        color: var(--dv-white);
    }
    .stButton > button[kind="primary"]:hover {
        background: #7969ed;
        border-color: #7969ed;
        color: var(--dv-white);
    }
    [data-testid="stFileUploader"] {
        background: var(--dv-soft);
        border: 1px dashed #c9c4f4;
        border-radius: 12px;
        padding: .45rem;
    }
    [data-testid="stFileUploader"] section {
        background: transparent;
        border: 0;
    }
    [data-testid="stFileUploader"] button {
        border-radius: 9px;
        border-color: var(--dv-line);
    }
    [data-testid="stRadio"] > div {
        gap: .35rem;
        padding: .25rem;
        background: var(--dv-soft);
        border: 1px solid var(--dv-line);
        border-radius: 12px;
        width: fit-content;
    }
    [data-testid="stRadio"] label {
        padding: .48rem .85rem;
        border-radius: 9px;
        color: var(--dv-muted);
        font-size: .92rem;
    }
    [data-testid="stRadio"] label:has(input:checked) {
        background: var(--dv-white);
        color: var(--dv-ink);
        box-shadow: 0 0 0 1px var(--dv-line);
    }
    [data-testid="stProgress"] > div > div {
        background: var(--dv-lavender);
        border-radius: 999px;
    }
    [data-testid="stProgress"] > div {
        background: #efeff2;
        border-radius: 999px;
    }
    [data-testid="stAlert"] {
        border-radius: 11px;
        border: 1px solid var(--dv-line);
    }
    .dv-brand {
        display: flex;
        align-items: center;
        gap: .75rem;
        padding: .25rem 0 1rem;
        border-bottom: 1px solid var(--dv-line);
        margin-bottom: 1rem;
    }
    .dv-mark {
        display: grid;
        place-items: center;
        width: 42px;
        height: 42px;
        border-radius: 13px;
        background: var(--dv-lavender-soft);
        color: var(--dv-lavender);
        font-size: 1.3rem;
        font-weight: 700;
    }
    .dv-brand-name { color: var(--dv-ink); font-size: 1.18rem; font-weight: 700; }
    .dv-brand-tagline {
        margin-top: .08rem;
        color: var(--dv-muted);
        font-size: .82rem;
        letter-spacing: .01em;
    }
    .dv-eyebrow {
        color: var(--dv-lavender);
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .dv-class-label {
        display: inline-block;
        margin-bottom: .15rem;
        color: var(--dv-lavender);
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .dv-footer {
        border-top: 1px solid var(--dv-line);
        margin-top: 2.5rem;
        padding: 1rem 0 .25rem;
        color: var(--dv-muted);
        font-size: .88rem;
        text-align: center;
    }
    .dv-footer strong { color: var(--dv-ink); }
    .dv-result-name { color: var(--dv-ink); font-size: 1.8rem; font-weight: 700; }
    .dv-material-card {
        border: 1px solid var(--dv-line);
        border-radius: 12px;
        padding: .85rem 1rem;
        background: var(--dv-soft);
    }
    .dv-material-label {
        color: var(--dv-muted);
        font-size: .75rem;
        font-weight: 700;
        letter-spacing: .1em;
        text-transform: uppercase;
    }
    .dv-material-value { margin-top: .2rem; color: var(--dv-ink); font-size: 1.15rem; font-weight: 650; }
    .dv-confidence-label {
        display: flex;
        justify-content: space-between;
        margin: .65rem 0 .25rem;
        color: var(--dv-muted);
        font-size: .85rem;
    }
    @media (max-width: 768px) {
        .main .block-container { padding: 1.1rem 1rem .75rem; }
        [data-testid="stRadio"] > div { width: 100%; justify-content: space-between; }
        [data-testid="stRadio"] label { padding: .45rem .55rem; font-size: .84rem; }
    }
    @media (max-width: 480px) {
        .main .block-container { padding: .9rem .8rem .65rem; }
        h1 { font-size: 2rem; }
        .dv-brand-tagline { font-size: .75rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    number_of_features = model.fc.in_features
    model.fc = nn.Identity()
    model.goddess_classifier = nn.Linear(number_of_features, len(GODDESSES))
    model.material_classifier = nn.Linear(number_of_features, len(MATERIALS))
    model = model.to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


@st.cache_resource
def compute_goddess_centroids(_model):
    """Build feature prototypes from validation images for the existing OOD gate."""
    import numpy as np
    from torchvision.datasets import ImageFolder

    validation_dir = DATASET_DIR / "validation"
    try:
        dataset = ImageFolder(str(validation_dir), transform=transform)
    except (FileNotFoundError, RuntimeError) as error:
        st.warning(f"Validation images unavailable; similarity check disabled ({error}).")
        return None

    if len(dataset) == 0:
        st.warning("Validation split is empty; similarity check disabled.")
        return None

    features_by_class = {name: [] for name in dataset.classes}
    with torch.no_grad():
        for image_tensor, class_index in dataset:
            feature = _model(image_tensor.unsqueeze(0).to(device))
            features_by_class[dataset.classes[class_index]].append(
                feature.cpu().numpy().reshape(-1)
            )

    populated_features = [
        np.mean(np.stack(class_features), axis=0)
        for class_features in features_by_class.values()
        if class_features
    ]
    if not populated_features:
        st.warning("No validation class features found; similarity check disabled.")
        return None

    centroids = np.stack(populated_features)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    return centroids / np.clip(norms, 1e-8, None)


def get_goddess_information(goddess_name):
    return GODDESS_INFO.get(goddess_name)


def set_active_page(page_name):
    st.session_state["active_page"] = page_name


def render_header():
    st.markdown(
        """
        <div class="dv-brand">
          <div>
            <div class="dv-brand-name">Divine Vision</div>
            <div class="dv-brand-tagline">AI-powered Goddess idol recognition</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.session_state.setdefault("active_page", "Home")
    st.radio(
        "Main navigation",
        ["Home", "Identify Idol", "Explore Goddesses"],
        horizontal=True,
        label_visibility="collapsed",
        key="active_page",
    )


def render_footer():
    st.markdown(
        """
        <div class="dv-footer">
          <strong>Divine Vision</strong>
          <span> · AI-powered exploration of Goddess idols and cultural heritage</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home():
    hero_copy, hero_art = st.columns([1.15, 0.85], gap="large", vertical_alignment="center")
    with hero_copy:
        st.markdown(
            '<div class="dv-eyebrow">A thoughtful way to discover</div>',
            unsafe_allow_html=True,
        )
        st.title("Recognize the art. Discover the story.")
        st.write(
            "Upload an idol image to receive an AI prediction for its Goddess and material, "
            "alongside concise cultural context."
        )
        action, explore = st.columns(2)
        with action:
            st.button(
                "Identify an Idol",
                type="primary",
                use_container_width=True,
                on_click=set_active_page,
                args=("Identify Idol",),
            )
        with explore:
            st.button(
                "Explore Goddesses",
                use_container_width=True,
                on_click=set_active_page,
                args=("Explore Goddesses",),
            )
    LOGO_PATH = Path(__file__).parent / "assets" / "divine_vision_logo.png"
    with hero_art:
        if LOGO_PATH.exists():
            # Crop a small amount from the top and bottom.
            with Image.open(LOGO_PATH) as original:
                logo = original.convert("RGB")
                width, height = logo.size

                crop_top = int(height * 0.06)
                crop_bottom = int(height * 0.94)

                logo = logo.crop((0, crop_top, width, crop_bottom))

                st.image(
                    logo,
                    use_container_width=True,
                )
        else:
            st.info(
                "Explore the Goddess library to learn about all six supported classes."
            )

    overview = st.columns(3, gap="medium")
    for column, value, label in zip(
        overview,
        ("06", "06", "JPG · PNG"),
        ("Goddess categories", "Material classes", "Image formats supported"),
    ):
        with column.container(border=True):
            st.metric(label, value)

    st.markdown("### A considered approach to idol recognition")
    feature_columns = st.columns(3, gap="medium")
    feature_cards = [
        (
            "Goddess Recognition",
            "Classifies uploaded idol imagery across the six Goddess classes in the trained model.",
        ),
        (
            "Material Classification",
            "Predicts whether the idol appears to be brass, bronze, copper, silver, stone, or wood.",
        ),
        (
            "Cultural Information",
            "Pairs recognized predictions with a concise introduction and cultural significance.",
        ),
    ]
    for column, (title, description) in zip(feature_columns, feature_cards):
        with column.container(border=True):
            st.markdown(f"#### {title}")
            st.write(description)

    st.markdown("### How it works")
    steps = st.columns(3, gap="medium")
    for column, number, title, description in zip(
        steps,
        ("01", "02", "03"),
        ("Upload", "Predict", "Explore"),
        (
            "Choose one or more JPG, JPEG, or PNG idol images.",
            "The existing image pipeline predicts a Goddess and material for each image.",
            "Review confidence, cultural context, and further reading.",
        ),
    ):
        with column:
            st.markdown(f"**{number} · {title}**")
            st.write(description)


def render_goddess_library():
    st.markdown('<div class="dv-eyebrow">The collection</div>', unsafe_allow_html=True)
    st.title("Explore Goddesses")
   
    st.write(
        "Discover the divine beauty, timeless stories, and rich cultural "
        "heritage of six revered Goddesses. Explore their significance, "
        "symbolism, and traditional attributes while experiencing how "
        "AI-powered image recognition brings the world of sacred idols "
        "to life. Each Goddess tells a unique story of strength, wisdom, "
        "compassion, and divine feminine energy."
    )

    for start in range(0, len(GODDESSES), 3):
        columns = st.columns(3)
        for index, (column, goddess_name) in enumerate(
            zip(columns, GODDESSES[start : start + 3]),
            start=start + 1,
        ):
            information = GODDESS_INFO[goddess_name]
            with column.container(border=True):
                image_path = find_goddess_image(
                    goddess_name,
                    dataset_root=str(DATASET_DIR),
                )
                if image_path:
                    st.image(image_path, caption=f"{goddess_name} idol", width="stretch")
                else:
                    st.caption("No local dataset image is available for this class.")
                st.markdown(
                    f'<div class="dv-class-label">Goddess class {index:02d}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"### {goddess_name}")
                st.write(information["introduction"])
                st.write(information["significance"])
                st.markdown(f"**Attributes:** {information['attributes']}")
                st.markdown(
                    f'<a href="{information["read_more"]}" target="_blank" '
                    f'rel="noopener noreferrer">Read more about {goddess_name} ↗</a>',
                    unsafe_allow_html=True,
                )


def predict_image(model, image, centroids):
    image_tensor = transform(image.copy()).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image_tensor)
        goddess_logits = model.goddess_classifier(features)
        material_logits = model.material_classifier(features)
        goddess_probabilities = torch.softmax(
            goddess_logits / BEST_GODDESS_TEMP,
            dim=1,
        )
        material_probabilities = torch.softmax(
            material_logits / BEST_MATERIAL_TEMP,
            dim=1,
        )
        goddess_confidence, goddess_index = goddess_probabilities.max(dim=1)
        material_confidence, material_index = material_probabilities.max(dim=1)

    confidence_value = goddess_confidence.item()
    material_confidence_value = material_confidence.item()
    similarity = None
    if centroids is not None:
        feature = features.cpu().numpy().reshape(-1)
        feature /= max(float(np.linalg.norm(feature)), 1e-8)
        similarity = float((centroids @ feature).max())

    recognized = (
        confidence_value >= RECOGNITION_THRESHOLD
        and material_confidence_value >= RECOGNITION_THRESHOLD
        and (similarity is None or similarity >= CENTROID_THRESHOLD)
    )
    return {
        "goddess": GODDESSES[goddess_index.item()] if recognized else None,
        "top_goddess": GODDESSES[goddess_index.item()],
        "goddess_confidence": confidence_value,
        "top_material": MATERIALS[material_index.item()],
        "material": MATERIALS[material_index.item()] if recognized else None,
        "material_confidence": material_confidence_value,
        "recognized": recognized,
    }


def render_prediction_page():
    st.markdown('<div class="dv-eyebrow">Image recognition</div>', unsafe_allow_html=True)
    st.title("Identify Your Goddess Idol")
    st.write(
        "Upload one or more idol photos. Each image is processed independently; "
        "predictions are estimates rather than certain identification."
    )
    with st.container(border=True):
        st.markdown("#### Add idol photos")
        st.caption("Select JPG, JPEG, or PNG files. Multiple images are supported.")
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Supported formats: JPG, JPEG, and PNG.",
            label_visibility="collapsed",
        )
    if not uploaded_files:
        st.info("Your individual prediction cards will appear here after you upload photos.")
        return
    st.caption(
        f"{len(uploaded_files)} image(s) selected: "
        + " · ".join(uploaded_file.name for uploaded_file in uploaded_files)
    )

    model = None
    centroids = None
    model_error = None
    with st.spinner("Loading the trained model..."):
        try:
            model = load_model()
        except Exception as error:
            model_error = error
        if model is not None:
            try:
                centroids = compute_goddess_centroids(model)
            except Exception as error:
                st.warning(f"Similarity check unavailable: {error}")

    processed_images = []
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        item = {
            "index": index,
            "name": uploaded_file.name,
            "image": None,
            "result": None,
            "error": None,
        }
        if Path(uploaded_file.name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            item["error"] = "Unsupported file type. Upload a JPG, JPEG, or PNG image."
            processed_images.append(item)
            continue

        try:
            uploaded_file.seek(0)
            with Image.open(uploaded_file) as source_image:
                source_image.verify()
            uploaded_file.seek(0)
            with Image.open(uploaded_file) as source_image:
                item["image"] = source_image.convert("RGB")
                item["image"].load()
        except (UnidentifiedImageError, OSError, ValueError) as error:
            item["error"] = f"Could not read this file as a valid image: {error}"
            processed_images.append(item)
            continue

        if model_error is not None or model is None:
            item["error"] = "The trained model could not be loaded; prediction is unavailable."
            processed_images.append(item)
            continue

        with st.spinner(f"Predicting image {index}..."):
            try:
                item["result"] = predict_image(model, item["image"], centroids)
            except Exception as error:
                item["error"] = f"Prediction failed: {error}"
        processed_images.append(item)


    for item in processed_images:
        with st.container(border=True):
            content_column, image_column = st.columns(
                [1.35, 1],
                gap="large",
                vertical_alignment="center",
            )
            with content_column:
                st.caption(f"IMAGE {item['index']}")
                st.markdown(f"### {item['name']}")
                if item["error"]:
                    st.error(item["error"])
                else:
                    result = item["result"]
                    if not result["recognized"]:
                        st.warning("Image not recognized")
                        st.write("Try another image.")
                    else:
                        goddess_name = result["goddess"]
                        information = get_goddess_information(goddess_name)
                        st.markdown(
                            f'<div class="dv-result-name" style="margin: .2rem 0 .55rem;">'
                            f'{goddess_name}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div class="dv-material-card">'
                            f'<div class="dv-material-label">Predicted material</div>'
                            f'<div class="dv-material-value">{result["material"]}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div class="dv-confidence-label"><span>Goddess confidence</span>'
                            f'<strong>{result["goddess_confidence"]:.1%}</strong></div>',
                            unsafe_allow_html=True,
                        )
                        st.progress(result["goddess_confidence"])
                        st.markdown(
                            f'<div class="dv-confidence-label"><span>Material confidence</span>'
                            f'<strong>{result["material_confidence"]:.1%}</strong></div>',
                            unsafe_allow_html=True,
                        )
                        st.progress(result["material_confidence"])
                        if information:
                            st.write(
                                f"{information['introduction']} "
                                f"{information['significance']}"
                            )
                            st.markdown(
                                f'<a href="{information["read_more"]}" target="_blank" '
                                f'rel="noopener noreferrer">Read more about {goddess_name} ↗</a>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.info("Additional Goddess information is unavailable.")

            with image_column:
                if item["image"] is not None:
                    st.image(item["image"], width="stretch")


render_header()
page = st.session_state["active_page"]
if page == "Home":
    render_home()
elif page == "Explore Goddesses":
    render_goddess_library()
else:
    render_prediction_page()
render_footer()
