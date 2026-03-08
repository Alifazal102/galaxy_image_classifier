"""
Galaxy Image Classifier — Upload a galaxy image to get morphological predictions.
"""
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from model import GalaxyNN
import streamlit as st


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths: model weights and label catalog (class names from Galaxy Zoo taxonomy)
CHECKPOINT_PATH = "models/galaxy_net.pth"
LABEL_CATALOG_PATH = os.path.join("data", "training_classifications.csv")

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@st.cache_resource
def load_label_names() -> List[str]:
    """Load class IDs from the label catalog."""
    df = pd.read_csv(LABEL_CATALOG_PATH)
    return df.columns[1:].astype(str).tolist()


@st.cache_resource
def load_label_descriptions() -> Dict[str, str]:
    """Map class IDs to human-readable descriptions."""
    label_names = load_label_names()
    return {
        label_names[0]: "Smooth and rounded (no obvious disk)",
        label_names[1]: "Features or a disk are visible",
        label_names[2]: "Looks like a star or image artifact rather than a galaxy",
        label_names[3]: "Disk viewed edge-on",
        label_names[4]: "Disk not edge-on (more face-on)",
        label_names[5]: "Central bar feature present",
        label_names[6]: "Not barred (no central bar feature)",
        label_names[7]: "Spiral arms present",
        label_names[8]: "Spiral arms not visible",
        label_names[9]: "Central bulge not visible",
        label_names[10]: "Central bulge just noticeable",
        label_names[11]: "Central bulge is obvious",
        label_names[12]: "Central bulge is dominant",
        label_names[13]: "Something odd or unusual is present",
        label_names[14]: "Not odd (nothing unusual seen)",
        label_names[15]: "Completely round shape",
        label_names[16]: "In-between round and elongated",
        label_names[17]: "Cigar-shaped (strongly elongated)",
        label_names[18]: "Ring-shaped feature",
        label_names[19]: "Lens or arc feature",
        label_names[20]: "Galaxy looks disturbed",
        label_names[21]: "Galaxy looks irregular",
        label_names[22]: "Other kind of odd feature",
        label_names[23]: "Merger (two galaxies interacting)",
        label_names[24]: "Dust lane visible",
        label_names[25]: "Rounded bulge shape",
        label_names[26]: "Boxy bulge shape",
        label_names[27]: "Central bulge not visible",
        label_names[28]: "Tightly wound spiral arms",
        label_names[29]: "Medium-wound spiral arms",
        label_names[30]: "Loosely wound spiral arms",
        label_names[31]: "1 spiral arm",
        label_names[32]: "2 spiral arms",
        label_names[33]: "3 spiral arms",
        label_names[34]: "4 spiral arms",
        label_names[35]: "More than four spiral arms",
        label_names[36]: "Can't tell how many spiral arms",
    }


@st.cache_resource
def load_model_and_labels() -> Tuple[torch.nn.Module, List[str]]:
    """Load the trained model and label catalog."""
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at '{CHECKPOINT_PATH}'. "
            f"Place galaxy_net.pth there or update CHECKPOINT_PATH."
        )

    label_names = load_label_names()
    model = GalaxyNN()
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, label_names


def downsize_pil_image(image: Image.Image, margin: int = 60, rescale_factor: float = 6.0) -> Image.Image:
    """Crop margins and rescale to reduce image size before model input."""
    w, h = image.size
    if w <= 2 * margin or h <= 2 * margin:
        cropped = image
    else:
        cropped = image.crop((margin, margin, w - margin, h - margin))

    scale = 1.0 / float(np.sqrt(rescale_factor))
    new_w = max(1, int(round(cropped.size[0] * scale)))
    new_h = max(1, int(round(cropped.size[1] * scale)))

    resized = cropped.resize((new_w, new_h), Image.BILINEAR)
    return resized


def prepare_tensor_from_image(image: Image.Image) -> torch.Tensor:
    """Convert image to tensor: RGB, downsized, 77×77, normalized to [0,1]."""
    rgb = image.convert("RGB")
    downsized = downsize_pil_image(rgb)
    downsized = downsized.resize((77, 77), Image.BILINEAR)

    arr = np.array(downsized).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(DEVICE)
    return tensor


def run_inference(pil_image: Image.Image) -> Tuple[List[Tuple[str, float]], np.ndarray]:
    """Run the model and return top predictions with probabilities."""
    model, label_names = load_model_and_labels()
    label_descriptions = load_label_descriptions()
    x = prepare_tensor_from_image(pil_image)

    with torch.no_grad():
        outputs = model(x)

    if isinstance(outputs, (list, tuple)):
        outputs = outputs[0]

    probs = outputs.squeeze().detach().cpu().numpy()

    indices = np.argsort(probs)[::-1]
    top_k = min(10, len(indices))
    top = [
        (label_descriptions.get(label_names[i], label_names[i]), float(probs[i]))
        for i in indices[:top_k]
    ]
    return top, probs


def main() -> None:
    st.set_page_config(page_title="Galaxy Image Classifier", layout="wide")

    st.title("Galaxy Image Classifier")
    st.write(
        "Upload a galaxy image to classify its morphology. The model predicts "
        "attributes such as shape, spiral arms, and central bulge."
    )

    uploaded_file = st.file_uploader(
        "Choose a galaxy image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
    )

    if uploaded_file is None:
        st.info("Upload an image to get started.")
        return

    image = Image.open(uploaded_file)
    downsized_preview = downsize_pil_image(image)

    st.subheader("Image")
    col_orig, col_down = st.columns(2)
    with col_orig:
        st.caption("Original")
        st.image(image, width=350, use_container_width=False)
    with col_down:
        st.caption("Preprocessed (used for prediction)")
        st.image(downsized_preview, width=350, use_container_width=False)

    try:
        top_predictions, probs = run_inference(image)
    except Exception as exc:
        st.error(f"Error during prediction: {exc}")
        return

    st.subheader("Predictions")
    if not top_predictions:
        st.write("No predictions available.")
    else:
        best_label, best_prob = top_predictions[0]
        st.markdown(f"**Top prediction:** {best_label} ({best_prob:.3f})")

        pred_df = pd.DataFrame(
            {
                "Attribute": [name for name, _ in top_predictions],
                "Probability": [p for _, p in top_predictions],
            }
        )
        st.bar_chart(pred_df.set_index("Attribute")["Probability"])

        st.dataframe(
            pred_df.style.format({"Probability": "{:.3f}"}),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
