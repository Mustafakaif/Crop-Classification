# """Crop Classification App"""

# import sys
# import subprocess
# import os

# # Install system dependencies programmatically
# def install_system_deps():
#     try:
#         subprocess.check_call(['apt-get', 'update'])
#         subprocess.check_call(['apt-get', 'install', '-y', 'libgl1', 'libglib2.0-0', 'libsm6', 'libxext6', 'libxrender1', 'libfontconfig1', 'libgomp1'])
#     except:
#         pass  # Fail silently if we can't install (might be permission issues)

# # Try to install system deps
# if os.path.exists('/.dockerenv'):  # Check if we're in a container
#     try:
#         install_system_deps()
#     except:
#         pass

# # Install PyTorch with specific version for CPU
# try:
#     import torch
# except ImportError:
#     subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.0.0", "torchvision==0.15.1", "--extra-index-url", "https://download.pytorch.org/whl/cpu"])

# # Now import everything
# import streamlit as st
# import torch
# import joblib
# import torchvision.transforms as T
# from PIL import Image
# import matplotlib.pyplot as plt
# from collections import defaultdict
# from sklearn.metrics import classification_report
# import pandas as pd
# import numpy as np

# # Rest of your code...



# import sys
# import subprocess
# import pkg_resources
# import os

# # Install requirements if missing
# required = ['torch==2.0.0', 'torchvision==0.15.1', 'streamlit==1.25.0']
# for req in required:
#     package = req.split('==')[0]
#     try:
#         pkg_resources.get_distribution(package)
#     except:
#         subprocess.check_call([sys.executable, "-m", "pip", "install", req, "--quiet"])

# # Now import
# import streamlit as st
# import torch
# import joblib
# import torchvision.transforms as T
# from PIL import Image
# import matplotlib.pyplot as plt
# from collections import defaultdict
# from sklearn.metrics import classification_report
# import pandas as pd
# import numpy as np

# # Your MODEL_PATH
# MODEL_PATH = "crop_model_22k.joblib"  # Make sure this file is in your repo

# # Rest of your code...

"""Crop Classification App"""

import sys
import subprocess
import os

# Simple import check without pkg_resources
try:
    import torch
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.0.0", "torchvision==0.15.1", "streamlit==1.25.0", "--quiet"])
    import torch

# Now import everything else
import streamlit as st
import torch
import joblib
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# Your MODEL_PATH
MODEL_PATH = "crop_model_22k.joblib"

# Rest of your code continues here...



'''import streamlit as st
import torch
import joblib
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import classification_report

# -----------------------------
# CHANGE MODEL PATH HERE
# -----------------------------

MODEL_PATH = r"C:\Job\v1\crop_model_22k.joblib"'''

# -----------------------------
# CLASSES
# -----------------------------

classes = [
    "Tomato",
    "Potato",
    "Chilli",
    "Cucumber",
    "Okra",
    "Sunflower",
    "Ridgegourd",
    "Coriander"
]

# -----------------------------
# LOAD MODEL
# -----------------------------

@st.cache_resource
def load_models():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dinov2 = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14"
    ).to(device)

    dinov2.eval()

    svm_model = joblib.load(MODEL_PATH)

    return dinov2, svm_model, device


dinov2, model, device = load_models()

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# -----------------------------
# PREDICT FUNCTION
# -----------------------------

def predict_image(image):

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = dinov2(img).cpu().numpy()

    prediction = model.predict(features)[0]

    return prediction


# -----------------------------
# STREAMLIT UI
# -----------------------------

st.title("🌱 Crop Classification Evaluation Tool")

uploaded_files = st.file_uploader(
    "Upload test images",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True
)

# -----------------------------
# RUN MODEL ONLY ONCE
# -----------------------------

if uploaded_files and "results" not in st.session_state:

    results = []

    with st.spinner("Running model predictions..."):

        for file in uploaded_files:

            image = Image.open(file).convert("RGB")

            prediction = predict_image(image)

            entry = {
                "image": image,
                "pred": prediction,
                "true": prediction
            }

            results.append(entry)

    st.session_state.results = results


# -----------------------------
# DISPLAY REVIEW UI
# -----------------------------

if "results" in st.session_state:

    results = st.session_state.results

    grouped = defaultdict(list)

    for r in results:
        grouped[r["pred"]].append(r)

    st.header("Model Predictions Review")

    with st.form("review_form"):

        for cls in grouped:

            st.subheader(f"{cls} Folder")

            items = grouped[cls]

            for idx, item in enumerate(items):

                col1, col2 = st.columns([1,1])

                col1.image(item["image"], width=200)

                default_index = classes.index(item["pred"]) if item["pred"] in classes else 0

                true_label = col2.selectbox(
                    "Actual Label",
                    classes,
                    index=default_index,
                    key=f"{cls}_{idx}"
                )

                item["true"] = true_label

                if item["pred"] == true_label:
                    col2.success("✔ Correct Prediction")
                else:
                    col2.error("✖ Wrong Prediction")

        submitted = st.form_submit_button("Generate Summary Report")

    # -----------------------------
    # GENERATE REPORT
    # -----------------------------

    if submitted:

        correct = 0
        wrong = 0

        wrong_images = []

        true_labels = []
        pred_labels = []

        for r in results:

            pred_labels.append(r["pred"])
            true_labels.append(r["true"])

            if r["pred"] == r["true"]:
                correct += 1
            else:
                wrong += 1
                wrong_images.append(r)

        total = len(results)

        accuracy = correct / total if total else 0

        st.header("📊 Model Evaluation Report")

        st.write(f"Total Images: {total}")
        st.write(f"Correct Predictions: {correct}")
        st.write(f"Wrong Predictions: {wrong}")
        st.write(f"Accuracy: {accuracy:.2f}")

        # -----------------------------
        # GRAPH 1 ACCURACY BAR
        # -----------------------------

        fig1, ax1 = plt.subplots()

        ax1.bar(["Correct","Wrong"], [correct,wrong])

        ax1.set_title("Prediction Accuracy")

        st.pyplot(fig1)

        # -----------------------------
        # GRAPH 2 PIE CHART
        # -----------------------------

        fig2, ax2 = plt.subplots()

        ax2.pie(
            [correct,wrong],
            labels=["Correct","Wrong"],
            autopct="%1.1f%%"
        )

        ax2.set_title("Prediction Distribution")

        st.pyplot(fig2)

        # -----------------------------
        # GRAPH 3 CLASS DISTRIBUTION
        # -----------------------------

        class_counts = defaultdict(int)

        for p in pred_labels:
            class_counts[p] += 1

        fig3, ax3 = plt.subplots()

        ax3.bar(class_counts.keys(), class_counts.values())

        ax3.set_title("Predicted Class Distribution")

        plt.xticks(rotation=45)

        st.pyplot(fig3)

        # -----------------------------
        # CLASSWISE F1 SCORE
        # -----------------------------

        from sklearn.metrics import classification_report
        import pandas as pd

        st.header("📊 Class-wise Metrics Table")

        # Generate report as dictionary
        report = classification_report(
            true_labels,
            pred_labels,
            output_dict=True
        )

        # Convert to DataFrame
        df_report = pd.DataFrame(report).transpose()

        # Keep only actual classes (remove accuracy/macro avg rows)
        df_report = df_report.reindex(classes, fill_value=0)

        # Round values for better display
        df_report = df_report.round(2)

        
        # Show table
        st.dataframe(df_report)
        # -----------------------------
        # WRONG PREDICTIONS
        # -----------------------------

        if wrong_images:

            st.header("❌ Wrong Predictions")

            for w in wrong_images:

                st.image(
                    w["image"],
                    caption=f"Predicted: {w['pred']} | Actual: {w['true']}",
                    width=200
                )

        else:

            st.success("No wrong predictions found 🎉")