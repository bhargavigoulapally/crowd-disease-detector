import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Crop Disease Detector",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f4fff5, #e8f5e9);
    }

    /* Hide Streamlit default elements */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 46px;
        font-weight: 800;
        color: #1b5e20;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #4f6351;
        font-size: 18px;
        margin-bottom: 30px;
    }

    /* Cards */
    .card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0px 5px 20px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    .result-card {
        background: linear-gradient(135deg, #e8f5e9, #ffffff);
        padding: 25px;
        border-radius: 18px;
        border-left: 6px solid #2e7d32;
        box-shadow: 0px 5px 20px rgba(0,0,0,0.08);
    }

    .disease-title {
        color: #1b5e20;
        font-size: 27px;
        font-weight: 700;
    }

    .confidence {
        font-size: 20px;
        font-weight: 600;
        color: #2e7d32;
    }

    .section-title {
        color: #1b5e20;
        font-size: 23px;
        font-weight: 700;
    }

    .info-box {
        background: #f1f8e9;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #c8e6c9;
        margin-top: 10px;
    }

    .footer {
        text-align: center;
        color: #607d64;
        margin-top: 40px;
        padding: 20px;
        font-size: 14px;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 50px;
        font-size: 18px;
        font-weight: 600;
    }

</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🌱 Crop Disease Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered crop leaf disease detection system</div>',
    unsafe_allow_html=True
)

# ============================================================
# MODEL CONFIGURATION
# ============================================================

# Public Hugging Face model
API_URL = "https://api-inference.huggingface.co/models/linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"

# No API key is required for this demo.
HEADERS = {}


# ============================================================
# DISEASE INFORMATION
# ============================================================

DISEASE_INFO = {

    "healthy": {
        "crop": "Crop",
        "description": "The uploaded leaf appears healthy.",
        "prevention": [
            "Maintain proper watering.",
            "Provide sufficient sunlight.",
            "Use balanced fertilizers.",
            "Regularly inspect plants for early symptoms."
        ]
    },

    "early blight": {
        "crop": "Tomato / Potato",
        "description": "Early blight commonly produces dark spots and concentric rings on leaves.",
        "prevention": [
            "Remove infected leaves.",
            "Avoid watering directly on leaves.",
            "Maintain good spacing between plants.",
            "Keep the field clean from infected plant material."
        ]
    },

    "late blight": {
        "crop": "Tomato / Potato",
        "description": "Late blight can cause dark irregular lesions on leaves and rapid plant damage.",
        "prevention": [
            "Remove severely infected plants.",
            "Avoid excess moisture around foliage.",
            "Improve air circulation.",
            "Monitor plants regularly."
        ]
    },

    "bacterial": {
        "crop": "Various crops",
        "description": "Bacterial diseases may appear as spots, lesions, or damaged areas on leaves.",
        "prevention": [
            "Remove infected plant material.",
            "Avoid unnecessary leaf wetness.",
            "Use clean gardening tools.",
            "Maintain good field hygiene."
        ]
    },

    "rust": {
        "crop": "Various crops",
        "description": "Rust diseases commonly produce orange, yellow, or brown powder-like spots.",
        "prevention": [
            "Remove heavily infected leaves.",
            "Improve air circulation.",
            "Avoid excessive humidity.",
            "Monitor plants frequently."
        ]
    }
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_label(label):
    """Convert model label into a readable name."""
    label = str(label).replace("_", " ")
    label = label.replace("  ", " ")
    return label.strip()


def get_disease_info(label):
    """Find suitable disease information."""

    lower_label = label.lower()

    if "healthy" in lower_label:
        return DISEASE_INFO["healthy"]

    if "early blight" in lower_label:
        return DISEASE_INFO["early blight"]

    if "late blight" in lower_label:
        return DISEASE_INFO["late blight"]

    if "bacterial" in lower_label:
        return DISEASE_INFO["bacterial"]

    if "rust" in lower_label:
        return DISEASE_INFO["rust"]

    return {
        "crop": "Detected Crop",
        "description": "The AI model identified a possible plant condition from the uploaded image.",
        "prevention": [
            "Inspect the plant carefully.",
            "Remove severely damaged leaves.",
            "Maintain proper watering.",
            "Avoid overcrowding.",
            "Consult a local agricultural expert for confirmation."
        ]
    }


def predict_disease(image):

    try:
        # Convert image to RGB
        image = image.convert("RGB")

        # Save image into memory
        image_bytes = BytesIO()
        image.save(image_bytes, format="JPEG")
        image_bytes.seek(0)

        # Send image to Hugging Face inference API
        response = requests.post(
            API_URL,
            headers=HEADERS,
            data=image_bytes.getvalue(),
            timeout=60
        )

        # Model loading
        if response.status_code == 503:

            try:
                data = response.json()

                wait_time = data.get(
                    "estimated_time",
                    20
                )

                return None, f"Model is loading. Please wait about {int(wait_time)} seconds and try again."

            except Exception:
                return None, "The AI model is currently loading. Please try again in a few seconds."

        # API error
        if response.status_code != 200:
            return None, f"AI service returned an error: {response.status_code}"

        result = response.json()

        # Validate response
        if not isinstance(result, list) or len(result) == 0:
            return None, "No prediction was returned by the AI model."

        # Sort predictions
        result = sorted(
            result,
            key=lambda x: float(x.get("score", 0)),
            reverse=True
        )

        best_prediction = result[0]

        label = clean_label(
            best_prediction.get("label", "Unknown")
        )

        score = float(
            best_prediction.get("score", 0)
        )

        return {
            "label": label,
            "score": score,
            "all_predictions": result[:5]
        }, None

    except requests.exceptions.Timeout:
        return None, "The AI service took too long to respond. Please try again."

    except requests.exceptions.RequestException:
        return None, "Unable to connect to the AI service. Check your internet connection."

    except Exception as e:
        return None, f"Unexpected error: {str(e)}"


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">📷 Upload Crop Leaf Image</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a clear image of a crop leaf. "
    "The AI model will analyze the image and predict a possible disease."
)

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload JPG, JPEG, PNG or WEBP image"
)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# IMAGE + ANALYSIS
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    left, right = st.columns(2)

    with left:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">🖼️ Uploaded Image</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            use_container_width=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-title">🔍 AI Analysis</div>',
            unsafe_allow_html=True
        )

        analyze = st.button(
            "🌱 Analyze Crop",
            type="primary"
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================
    # PREDICTION
    # ========================================================

    if analyze:

        with st.spinner("AI is analyzing the crop leaf..."):

            prediction, error = predict_disease(image)

        if error:

            st.error(error)

        else:

            label = prediction["label"]
            score = prediction["score"]

            info = get_disease_info(label)

            confidence = score * 100

            # =================================================
            # RESULT
            # =================================================

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="disease-title">🌿 Prediction Result</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"### 🦠 {label}"
            )

            st.markdown(
                f'<div class="confidence">Confidence: {confidence:.2f}%</div>',
                unsafe_allow_html=True
            )

            st.progress(
                min(max(score, 0.0), 1.0)
            )

            st.markdown("</div>", unsafe_allow_html=True)

            # =================================================
            # DETAILS
            # =================================================

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    '<div class="card">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="section-title">🌾 Crop Information</div>',
                    unsafe_allow_html=True
                )

                st.write(
                    f"**Possible crop:** {info['crop']}"
                )

                st.write(
                    f"**Detected condition:** {label}"
                )

                st.markdown("</div>", unsafe_allow_html=True)

            with col2:

                st.markdown(
                    '<div class="card">',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="section-title">📋 Description</div>',
                    unsafe_allow_html=True
                )

                st.write(
                    info["description"]
                )

                st.markdown("</div>", unsafe_allow_html=True)

            # =================================================
            # PREVENTION
            # =================================================

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-title">🛡️ Prevention / Management</div>',
                unsafe_allow_html=True
            )

            for item in info["prevention"]:
                st.write(f"• {item}")

            st.markdown("</div>", unsafe_allow_html=True)

            # =================================================
            # TOP PREDICTIONS
            # =================================================

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-title">📊 Top AI Predictions</div>',
                unsafe_allow_html=True
            )

            for item in prediction["all_predictions"]:

                prediction_label = clean_label(
                    item.get("label", "Unknown")
                )

                prediction_score = (
                    float(item.get("score", 0)) * 100
                )

                st.write(
                    f"**{prediction_label}** — "
                    f"{prediction_score:.2f}%"
                )

                st.progress(
                    min(
                        max(
                            float(item.get("score", 0)),
                            0.0
                        ),
                        1.0
                    )
                )

            st.markdown("</div>", unsafe_allow_html=True)

            # =================================================
            # DISCLAIMER
            # =================================================

            st.warning(
                "⚠️ This AI prediction is for educational and "
                "informational purposes only. For serious crop "
                "disease problems, consult an agricultural expert."
            )


# ============================================================
# INFORMATION SECTION
# ============================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">🌱 How It Works</div>',
    unsafe_allow_html=True
)

st.write(
    """
    1. Upload a crop leaf image.
    
    2. The image is sent to an AI image-classification model.
    
    3. The model identifies the most likely plant disease.
    
    4. The application displays the predicted disease and confidence score.
    
    5. General prevention and management information is displayed.
    """
)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🌱 Crop Disease Detector | Built with Python + Streamlit + AI
        <br>
        AI-powered agricultural assistance for educational purposes
    </div>
    """,
    unsafe_allow_html=True
)