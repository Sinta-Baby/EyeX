"""
==========================================================
EyeX

Main Application

Description:
AI-assisted retinal fundus image analysis with
Image Quality Assessment, EfficientNet-B3
classification, and Grad-CAM explainability.

==========================================================
"""

import os
import streamlit as st


def dedent(text):
    """Remove indentation from every line for Streamlit HTML/CSS blocks."""
    return "\n".join(
        line.strip()
        for line in text.splitlines()
    ).strip()

from PIL import Image, UnidentifiedImageError

from inference.predict import predict_image
from inference.lesion_detector import IDRiDLesionDetector


@st.cache_resource
def load_lesion_detector():
    """Load the trained IDRiD U-Net once and reuse it across reruns."""
    return IDRiDLesionDetector()


from processors.image_quality import (
    assess_image_quality
)


# ======================================================
# Page Configuration
# ======================================================

st.set_page_config(
    page_title="EyeX | Explainable Retinal AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ======================================================
# Class / Visual Accent Configuration
# ======================================================

CLASS_ACCENTS = {

    "Healthy": {
        "color": "#0F9D78",
        "bg": "#E7F7F1",
        "label": "Healthy"
    },

    "DR": {
        "color": "#B4790F",
        "bg": "#FBF1DF",
        "label": "Diabetic Retinopathy"
    },

    "Glaucoma": {
        "color": "#5B4FCF",
        "bg": "#EEECFB",
        "label": "Glaucoma"
    },

    "AMD": {
        "color": "#C0392B",
        "bg": "#FBEAE8",
        "label": "Age-Related Macular Degeneration"
    },
}


DEFAULT_ACCENT = {
    "color": "#1B4C8C",
    "bg": "#EAF1FB",
    "label": "Result"
}


def get_accent(disease_key: str) -> dict:

    """
    Return a cosmetic accent for the predicted class.
    """

    return CLASS_ACCENTS.get(
        str(disease_key).strip(),
        DEFAULT_ACCENT
    )


# ======================================================
# Session State
# ======================================================

if "quality_result" not in st.session_state:

    st.session_state.quality_result = None


if "quality_image_name" not in st.session_state:

    st.session_state.quality_image_name = None


if "analysis_result" not in st.session_state:

    st.session_state.analysis_result = None


if "lesion_result" not in st.session_state:

    st.session_state.lesion_result = None


# ======================================================
# Global Styling
# ======================================================

st.markdown(
    dedent(
        """
        <style>

    :root {
        --ex-navy: #0B1F3D;
        --ex-navy-light: #14294D;
        --ex-blue: #1B4C8C;
        --ex-cyan: #2DD4BF;
        --ex-bg: #F4F7FB;
        --ex-border: #E4E9F1;
        --ex-text-dark: #0F172A;
        --ex-text-muted: #64748B;
    }


    /* App background */

    .stApp {
        background: var(--ex-bg);
    }


    /* Hide default Streamlit chrome */

    #MainMenu,
    footer {
        visibility: hidden;
    }


    /* Page spacing */

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }


    /* ==================================================
       Header
       ================================================== */

    .ex-header {

        background:
            linear-gradient(
                120deg,
                var(--ex-navy) 0%,
                var(--ex-blue) 65%,
                #12507A 100%
            );

        border-radius: 20px;

        padding: 34px 40px;

        color: #F5F8FC;

        margin-bottom: 28px;

        box-shadow:
            0 12px 30px
            rgba(11, 31, 61, 0.25);
    }


    .ex-header-top {

        display: flex;

        justify-content: space-between;

        align-items: flex-start;

        flex-wrap: wrap;

        gap: 16px;
    }


    .ex-brand {

        display: flex;

        align-items: center;

        gap: 14px;
    }


    .ex-brand-icon {

        width: 52px;

        height: 52px;

        border-radius: 14px;

        background:
            rgba(45, 212, 191, 0.15);

        border:
            1px solid
            rgba(45, 212, 191, 0.45);

        display: flex;

        align-items: center;

        justify-content: center;

        font-size: 26px;
    }


    .ex-brand-title {

        font-size: 30px;

        font-weight: 700;

        line-height: 1.1;

        letter-spacing: 0.2px;
    }


    .ex-brand-subtitle {

        font-size: 13px;

        color: #B9CBE6;

        letter-spacing: 0.6px;

        text-transform: uppercase;

        margin-top: 3px;
    }


    .ex-badges {

        display: flex;

        gap: 8px;

        flex-wrap: wrap;

        align-items: center;
    }


    .ex-badge {

        font-size: 12px;

        font-weight: 600;

        padding: 6px 12px;

        border-radius: 999px;

        background:
            rgba(255, 255, 255, 0.08);

        border:
            1px solid
            rgba(255, 255, 255, 0.18);

        color: #E5ECF7;

        white-space: nowrap;
    }


    .ex-header-tagline {

        font-size: 19px;

        font-weight: 600;

        margin-top: 22px;

        color: #FFFFFF;
    }


    .ex-header-desc {

        font-size: 14.5px;

        color: #C7D6EA;

        margin-top: 6px;

        max-width: 760px;

        line-height: 1.55;
    }


    /* ==================================================
       Sidebar
       ================================================== */

    section[data-testid="stSidebar"] {

        background:
            var(--ex-navy);
    }


    section[data-testid="stSidebar"] * {

        color: #E6ECF6 !important;
    }


    .ex-sidebar-brand {

        font-size: 20px;

        font-weight: 800;

        letter-spacing: 1px;

        margin-bottom: 2px;
    }


    .ex-sidebar-sub {

        font-size: 12px;

        color: #93A9C8 !important;

        margin-bottom: 22px;
    }


    .ex-nav-item {

        display: flex;

        align-items: center;

        gap: 10px;

        padding: 10px 12px;

        border-radius: 10px;

        margin-bottom: 6px;

        background:
            rgba(255, 255, 255, 0.04);

        border:
            1px solid
            rgba(255, 255, 255, 0.06);

        font-size: 14px;

        font-weight: 500;
    }


    .ex-sidebar-divider {

        border-top:
            1px solid
            rgba(255, 255, 255, 0.12);

        margin: 22px 0 16px 0;
    }


    .ex-sidebar-meta-label {

        font-size: 11px;

        text-transform: uppercase;

        letter-spacing: 0.6px;

        color: #7E93B5 !important;

        margin-bottom: 2px;
    }


    .ex-sidebar-meta-value {

        font-size: 14px;

        font-weight: 600;

        margin-bottom: 14px;
    }


    /* ==================================================
       Section headings
       ================================================== */

    .ex-section-title {

        font-size: 22px;

        font-weight: 700;

        color: var(--ex-text-dark);

        margin-bottom: 2px;
    }


    .ex-section-subtitle {

        font-size: 14px;

        color: var(--ex-text-muted);

        margin-bottom: 18px;
    }


    /* ==================================================
       Cards
       ================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {

        border-radius: 16px !important;

        border:
            1px solid
            var(--ex-border) !important;

        background:
            #FFFFFF !important;

        box-shadow:
            0 4px 18px
            rgba(15, 23, 42, 0.05);
    }


    /* ==================================================
       Metadata pills
       ================================================== */

    .ex-pill-row {

        display: flex;

        flex-wrap: wrap;

        gap: 8px;

        margin-top: 6px;
    }


    .ex-pill {

        background: #F1F5FB;

        border:
            1px solid
            var(--ex-border);

        border-radius: 10px;

        padding: 8px 12px;

        font-size: 13px;

        color: var(--ex-text-dark);
    }


    .ex-pill b {

        color: var(--ex-text-muted);

        font-weight: 600;

        margin-right: 6px;
    }


    /* ==================================================
       Result cards
       ================================================== */

    .ex-result-label {

        font-size: 13px;

        font-weight: 600;

        color: var(--ex-text-muted);

        text-transform: uppercase;

        letter-spacing: 0.5px;

        margin-bottom: 8px;
    }


    .ex-result-value {

        font-size: 34px;

        font-weight: 800;

        color: var(--ex-text-dark);

        line-height: 1.15;
    }


    .ex-status-chip {

        display: inline-block;

        font-size: 12px;

        font-weight: 700;

        padding: 5px 12px;

        border-radius: 999px;

        margin-top: 10px;
    }


    /* ==================================================
       Image Quality Metrics
       ================================================== */

    /* Make Streamlit metric labels clearly visible */
    [data-testid="stMetricLabel"] {
        color: #475569 !important;
    }

    /* Make metric values dark and prominent */
    [data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-weight: 700 !important;
    }

    /* Make metric delta/status text readable */
    [data-testid="stMetricDelta"] {
        color: #475569 !important;
    }

    /* Make captions below quality metrics readable */
    [data-testid="stMetric"] + div {
        color: #475569 !important;
    }


    /* ==================================================
       Quality section
       ================================================== */

    .ex-quality-header {

        font-size: 16px;

        font-weight: 700;

        color: var(--ex-text-dark);

        margin-bottom: 5px;
    }


    .ex-quality-description {

        font-size: 13px;

        color: var(--ex-text-muted);

        margin-bottom: 12px;
    }


    /* ==================================================
       IDRiD Lesion Cards
       ================================================== */

    .ex-lesion-card {
        background: #FFFFFF;
        border: 1px solid #DCE5F0;
        border-radius: 14px;
        padding: 16px 16px 14px 16px;
        min-height: 150px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
    }

    .ex-lesion-name {
        font-size: 12px;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.55px;
        margin-bottom: 8px;
    }

    .ex-lesion-status {
        font-size: 25px;
        line-height: 1.15;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 12px;
    }

    .ex-lesion-stat {
        font-size: 13px;
        line-height: 1.65;
        color: #475569;
    }

    .ex-lesion-stat b {
        color: #0F172A;
        font-weight: 700;
    }

    .ex-lesion-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        margin: 12px 0 4px 0;
        font-size: 12px;
        color: #475569;
    }

    .ex-lesion-legend-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    .ex-lesion-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }

    /* ==================================================
       Explainability
       ================================================== */

    .ex-explain-title {

        font-size: 16px;

        font-weight: 700;

        color: var(--ex-text-dark);

        margin-top: 10px;
    }


    .ex-explain-desc {

        font-size: 13px;

        color: var(--ex-text-muted);

        margin-bottom: 12px;
    }


    .st-key-gradcam_original
    [data-testid="stImage"] img,

    .st-key-gradcam_heatmap
    [data-testid="stImage"] img,

    .st-key-gradcam_overlay
    [data-testid="stImage"] img {

        width: 100%;

        height: auto;

        max-height: 480px;

        object-fit: contain;

        border-radius: 10px;

        display: block;

        margin: 0 auto;

        background: #F3F6FB;
    }


    @media (min-width: 1100px) {

        .st-key-gradcam_original
        [data-testid="stImage"] img,

        .st-key-gradcam_heatmap
        [data-testid="stImage"] img,

        .st-key-gradcam_overlay
        [data-testid="stImage"] img {

            max-height: 560px;
        }
    }


    /* ==================================================
       Insight card
       ================================================== */

    .ex-insight-item {

        font-size: 14px;

        color: var(--ex-text-dark);

        margin-bottom: 8px;

        padding-left: 4px;

        line-height: 1.5;
    }


    /* ==================================================
       Empty state
       ================================================== */

    .ex-empty-state {

        text-align: center;

        padding: 30px 10px;

        color: var(--ex-text-muted);

        font-size: 14px;
    }


    /* ==================================================
       Disclaimer
       ================================================== */

    .ex-disclaimer {

        background: #FFF7ED;

        border:
            1px solid
            #FCE3B8;

        border-radius: 16px;

        padding: 20px 24px;

        margin-top: 8px;
    }


    .ex-disclaimer-title {

        font-size: 14px;

        font-weight: 700;

        color: #92400E;

        text-transform: uppercase;

        letter-spacing: 0.5px;

        margin-bottom: 6px;
    }


    .ex-disclaimer-text {

        font-size: 13.5px;

        color: #7C5A20;

        line-height: 1.55;
    }


    /* ==================================================
       Footer
       ================================================== */

    .ex-footer {

        text-align: center;

        margin-top: 36px;

        padding-top: 20px;

        border-top:
            1px solid
            var(--ex-border);
    }


    .ex-footer-brand {

        font-size: 15px;

        font-weight: 700;

        color: var(--ex-text-dark);
    }


    .ex-footer-sub {

        font-size: 12.5px;

        color: var(--ex-text-muted);

        margin-top: 2px;
    }


    /* ==================================================
       Buttons
       ================================================== */

    div.stButton > button {

        border-radius: 12px;

        font-weight: 700;

        padding: 0.65rem 1rem;

        border: none;

        background:
            linear-gradient(
                120deg,
                var(--ex-navy),
                var(--ex-blue)
            );

        color: #FFFFFF;
    }


    div.stButton > button:hover {

        background:
            linear-gradient(
                120deg,
                var(--ex-blue),
                #12507A
            );

        color: #FFFFFF;
    }


        </style>
        """.strip()
    ),
    unsafe_allow_html=True,
)


# ======================================================
# Sidebar
# ======================================================

with st.sidebar:

    st.markdown(
        '<div class="ex-sidebar-brand">EYEX</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="ex-sidebar-sub">'
        'AI-Assisted Retinal Analysis'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="ex-nav-item">'
        '🔬&nbsp;&nbsp;Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="ex-nav-item">'
        '🩻&nbsp;&nbsp;Image Quality'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="ex-nav-item">'
        '🧠&nbsp;&nbsp;Explainability'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="ex-nav-item">'
        'ℹ️&nbsp;&nbsp;About Model'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="ex-sidebar-divider"></div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="ex-sidebar-meta-label">'
        'Model'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="ex-sidebar-meta-value">'
        'EfficientNet-B3'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="ex-sidebar-meta-label">'
        'Classification'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="ex-sidebar-meta-value">'
        '4 Classes'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="ex-sidebar-meta-label">'
        'Explainability'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="ex-sidebar-meta-value">'
        'Grad-CAM'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="ex-sidebar-meta-label">'
        'Purpose'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="ex-sidebar-meta-value">'
        'Research &amp; Education'
        '</div>',
        unsafe_allow_html=True
    )


# ======================================================
# Header
# ======================================================

st.markdown(
    dedent(
        """
        <div class="ex-header">

        <div class="ex-header-top">

            <div class="ex-brand">

                <div class="ex-brand-icon">
                    👁️
                </div>

                <div>

                    <div class="ex-brand-title">
                        EyeX
                    </div>

                    <div class="ex-brand-subtitle">
                        Explainable Retinal AI
                    </div>

                </div>

            </div>


            <div class="ex-badges">

                <div class="ex-badge">
                    EfficientNet-B3
                </div>

                <div class="ex-badge">
                    Image Quality
                </div>

                <div class="ex-badge">
                    Grad-CAM
                </div>

                <div class="ex-badge">
                    Research Prototype
                </div>

            </div>

        </div>


        <div class="ex-header-tagline">
            AI-assisted retinal fundus image analysis
            with explainable deep learning.
        </div>


        <div class="ex-header-desc">

            EyeX first checks whether a fundus image is
            suitable for analysis and then performs
            AI-assisted retinal disease classification
            using a trained EfficientNet-B3 model with
            Grad-CAM explainability.

        </div>

        </div>
        """.strip()
    ),
    unsafe_allow_html=True,
)


# ======================================================
# Upload Section
# ======================================================

with st.container(border=True):

    st.markdown(
        '<div class="ex-section-title">'
        'Analyze a Retinal Fundus Image'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="ex-section-subtitle">'
        'Upload a fundus photograph and follow the '
        'two-step EyeX analysis workflow.'
        '</div>',
        unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(
        "Upload Fundus Eye Image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "tif",
            "tiff"
        ],
        label_visibility="collapsed",
    )


    st.caption(
        "Supported formats: "
        "JPG · JPEG · PNG · BMP · TIFF"
    )

    st.caption(
        "Recommended: clear, centered retinal "
        "fundus photograph."
    )


# ======================================================
# Image Processing
# ======================================================

if uploaded_file is None:

    st.markdown(
        '<div class="ex-empty-state">'
        'No image uploaded yet. '
        'Upload a fundus photograph above '
        'to begin EyeX analysis.'
        '</div>',
        unsafe_allow_html=True
    )


else:

    # --------------------------------------------------
    # Open Uploaded Image
    # --------------------------------------------------

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_format = (
            uploaded_file.type or "unknown"
        ).split("/")[-1].upper()

        image_valid = True

    except (
        UnidentifiedImageError,
        OSError
    ):

        image = None

        image_valid = False


    if not image_valid:

        st.error(
            "Unable to process this image. "
            "Please upload a valid fundus image."
        )

    else:

        # --------------------------------------------------
        # Reset session state for a newly uploaded image
        # --------------------------------------------------

        if (
            st.session_state.quality_image_name
            != uploaded_file.name
        ):

            st.session_state.quality_result = None

            st.session_state.analysis_result = None

            st.session_state.lesion_result = None

            st.session_state.quality_image_name = (
                uploaded_file.name
            )


        st.write("")


        # ==================================================
        # Uploaded Image
        # ==================================================

        with st.container(border=True):

            st.markdown(
                '<div class="ex-section-title">'
                'Uploaded Image'
                '</div>',
                unsafe_allow_html=True
            )


            image_column, information_column = (
                st.columns([2, 1])
            )


            with image_column:

                st.image(
                    image,
                    use_container_width=True
                )


            with information_column:

                st.markdown(
                    dedent(
                        f"""
                        <div class="ex-pill-row"
                         style="flex-direction:column;">

                        <div class="ex-pill">
                            <b>File</b>
                            {uploaded_file.name}
                        </div>

                        <div class="ex-pill">
                            <b>Size</b>
                            {image.width} ×
                            {image.height}px
                        </div>

                        <div class="ex-pill">
                            <b>Format</b>
                            {image_format}
                        </div>

                        </div>
                        """
                    ),
                    unsafe_allow_html=True
                )


                st.success(
                    "Image uploaded successfully"
                )


        st.write("")


        # ==================================================
        # STEP 1 — IMAGE QUALITY
        # ==================================================

        st.markdown(
            '<div class="ex-section-title">'
            'Step 1 — Image Quality Assessment'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="ex-section-subtitle">'
            'Check whether the uploaded fundus image '
            'is suitable for AI analysis.'
            '</div>',
            unsafe_allow_html=True
        )


        quality_check_clicked = st.button(
            "🩻  Check Image Quality",
            use_container_width=True
        )


        # --------------------------------------------------
        # Run Quality Assessment
        # --------------------------------------------------

        if quality_check_clicked:

            os.makedirs(
                "results",
                exist_ok=True
            )


            quality_temp_path = (
                "results/quality_check_image.jpg"
            )


            try:

                image.save(
                    quality_temp_path
                )


                with st.spinner(
                    "Checking retinal image quality..."
                ):

                    st.session_state.quality_result = (
                        assess_image_quality(
                            quality_temp_path
                        )
                    )


                # Clear old analysis if quality is
                # checked again.

                st.session_state.analysis_result = None


            except Exception:

                st.error(
                    "Unable to assess image quality."
                )

                st.session_state.quality_result = None


        # --------------------------------------------------
        # Display Quality Result
        # --------------------------------------------------

        quality_result = (
            st.session_state.quality_result
        )


        if quality_result is not None:

            st.write("")


            with st.container(border=True):

                st.markdown(
                    '<div class="ex-quality-header">'
                    'Image Quality Result'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="ex-quality-description">'
                    'The following measurements were '
                    'used for the preliminary quality check.'
                    '</div>',
                    unsafe_allow_html=True
                )


                quality = (
                    quality_result["quality"]
                )


                # ------------------------------------------
                # Overall Quality
                # ------------------------------------------

                if quality == "Good":

                    st.success(
                        "🟢 GOOD QUALITY — "
                        "The image is suitable for "
                        "AI analysis."
                    )

                elif quality == "Fair":

                    st.warning(
                        "🟡 FAIR QUALITY — "
                        "The image can be analyzed, "
                        "but a clearer image is recommended."
                    )

                else:

                    st.error(
                        "🔴 POOR QUALITY — "
                        "Please upload a clearer "
                        "fundus image."
                    )


                st.write("")


                # ------------------------------------------
                # Quality Metrics
                # ------------------------------------------

                q1, q2, q3, q4 = st.columns(4)


                with q1:

                    st.metric(
                        "Sharpness",
                        f"{quality_result['sharpness']:.2f}"
                    )

                    if quality_result["sharpness_ok"]:

                        st.caption(
                            "✓ Acceptable"
                        )

                    else:

                        st.caption(
                            "⚠ Low sharpness"
                        )


                with q2:

                    st.metric(
                        "Brightness",
                        f"{quality_result['brightness']:.2f}"
                    )

                    if quality_result["brightness_ok"]:

                        st.caption(
                            "✓ Acceptable"
                        )

                    else:

                        st.caption(
                            "⚠ Poor brightness"
                        )


                with q3:

                    st.metric(
                        "Contrast",
                        f"{quality_result['contrast']:.2f}"
                    )

                    if quality_result["contrast_ok"]:

                        st.caption(
                            "✓ Acceptable"
                        )

                    else:

                        st.caption(
                            "⚠ Low contrast"
                        )


                with q4:

                    st.metric(
                        "Resolution",
                        f"{quality_result['width']} × "
                        f"{quality_result['height']}"
                    )

                    if quality_result["resolution_ok"]:

                        st.caption(
                            "✓ Acceptable"
                        )

                    else:

                        st.caption(
                            "⚠ Low resolution"
                        )


                st.write("")


                # ------------------------------------------
                # Quality Score
                # ------------------------------------------

                passed_checks = (
                    quality_result["passed_checks"]
                )


                quality_score = (
                    passed_checks / 4
                )


                st.markdown(
                    '<div class="ex-result-label">'
                    'Quality Check Score'
                    '</div>',
                    unsafe_allow_html=True
                )


                st.progress(
                    quality_score
                )


                st.caption(
                    f"{passed_checks} of 4 quality checks passed."
                )


        # ==================================================
        # STEP 2 — AI RETINAL ANALYSIS
        # ==================================================

        if quality_result is not None:

            st.divider()


            st.markdown(
                '<div class="ex-section-title">'
                'Step 2 — AI Retinal Analysis'
                '</div>',
                unsafe_allow_html=True
            )


            st.markdown(
                '<div class="ex-section-subtitle">'
                'Run the existing EyeX EfficientNet-B3 '
                'classification model and generate '
                'a Grad-CAM explanation.'
                '</div>',
                unsafe_allow_html=True
            )


            quality = (
                quality_result["quality"]
            )


            if quality == "Poor":

                st.error(
                    "⚠️ AI analysis is currently "
                    "disabled because the image quality "
                    "is poor. Please upload a clearer "
                    "fundus image."
                )


                analyze_clicked = False


            else:

                if quality == "Fair":

                    st.warning(
                        "The image has fair quality. "
                        "You may continue, but a clearer "
                        "image is recommended."
                    )


                analyze_clicked = st.button(
                    "🔍  Analyze Retina",
                    type="primary",
                    use_container_width=True
                )


        else:

            analyze_clicked = False


            st.info(
                "Complete Step 1 first: "
                "click **Check Image Quality** "
                "to continue to AI analysis."
            )


        # ==================================================
        # RUN AI ANALYSIS
        # ==================================================

        if analyze_clicked:

            # ----------------------------------------------
            # Save Temporary Image
            # ----------------------------------------------

            os.makedirs(
                "results",
                exist_ok=True
            )


            temp_path = (
                "results/uploaded_fundus_image.jpg"
            )


            try:

                image.save(
                    temp_path
                )

            except OSError:

                st.error(
                    "Unable to process this image. "
                    "Please upload a valid fundus image."
                )

                st.stop()


            # ----------------------------------------------
            # Existing RetinaSense Prediction
            # ----------------------------------------------

            try:

                with st.spinner(
                    "Analyzing retinal image..."
                ):

                    with st.spinner(
                        "Running EfficientNet-B3 inference..."
                    ):

                        with st.spinner(
                            "Generating Grad-CAM explanation..."
                        ):

                            result = predict_image(
                                temp_path
                            )


            except Exception:

                st.error(
                    "Unable to process this image. "
                    "Please upload a valid fundus image."
                )

                st.stop()


            # --------------------------------------------------
            # Store classification result
            # --------------------------------------------------

            st.session_state.analysis_result = result

            # --------------------------------------------------
            # Automatic IDRiD Lesion Analysis for DR
            # --------------------------------------------------

            if result["disease"] == "DR":

                try:

                    with st.spinner(
                        "Diabetic Retinopathy detected. "
                        "Running IDRiD lesion analysis..."
                    ):

                        lesion_detector = load_lesion_detector()

                        lesion_result = lesion_detector.predict(
                            temp_path,
                            threshold=0.5
                        )

                        st.session_state.lesion_result = (
                            lesion_result
                        )

                except Exception as lesion_error:

                    st.session_state.lesion_result = None

                    st.error(
                        "Unable to complete IDRiD lesion analysis."
                    )

                    st.caption(
                        f"Technical detail: {lesion_error}"
                    )

            else:

                st.session_state.lesion_result = None


        # ==================================================
        # DISPLAY ANALYSIS RESULT
        # ==================================================

        result = (
            st.session_state.analysis_result
        )


        if result is not None:

            accent = get_accent(
                result["disease"]
            )


            # ==================================================
            # Results Section
            # ==================================================

            st.divider()


            st.markdown(
                '<div class="ex-section-title">'
                'Analysis Result'
                '</div>',
                unsafe_allow_html=True
            )


            st.markdown(
                '<div class="ex-section-subtitle">'
                'AI output generated from the uploaded '
                'fundus image.'
                '</div>',
                unsafe_allow_html=True
            )


            result_column, confidence_column = (
                st.columns(2)
            )


            # ----------------------------------------------
            # Disease
            # ----------------------------------------------

            with result_column:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        '<div class="ex-result-label">'
                        'Predicted Condition'
                        '</div>',
                        unsafe_allow_html=True
                    )


                    st.markdown(
                        f'<div class="ex-result-value">'
                        f'{result["disease"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )


                    st.markdown(
                        f'<div class="ex-status-chip" '
                        f'style="background:{accent["bg"]}; '
                        f'color:{accent["color"]};">'
                        f'{accent["label"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )


            # ----------------------------------------------
            # Confidence
            # ----------------------------------------------

            with confidence_column:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        '<div class="ex-result-label">'
                        'Model Confidence'
                        '</div>',
                        unsafe_allow_html=True
                    )


                    st.markdown(
                        f'<div class="ex-result-value">'
                        f'{result["confidence"] * 100:.2f}%'
                        f'</div>',
                        unsafe_allow_html=True
                    )


                    st.markdown(
                        '<div class="ex-status-chip" '
                        'style="background:#EEF2F8; '
                        'color:#1B4C8C;">'
                        'Model prediction generated'
                        '</div>',
                        unsafe_allow_html=True
                    )


            st.write("")


            # ----------------------------------------------
            # Confidence Bar
            # ----------------------------------------------

            st.markdown(
                '<div class="ex-result-label">'
                'Confidence Level'
                '</div>',
                unsafe_allow_html=True
            )


            st.progress(
                min(
                    max(
                        result["confidence"],
                        0.0
                    ),
                    1.0
                )
            )


            # ==================================================
            # Grad-CAM Explainability
            # ==================================================

            st.divider()


            st.markdown(
                '<div class="ex-section-title">'
                'Explainable AI'
                '</div>',
                unsafe_allow_html=True
            )


            st.markdown(
                '<div class="ex-section-subtitle">'
                'Where did the model focus?'
                '</div>',
                unsafe_allow_html=True
            )


            st.write(
                "Grad-CAM highlights the regions of the "
                "fundus image that contributed most to "
                "the model's prediction."
            )


            gc1, gc2, gc3 = st.columns(
                3,
                gap="medium"
            )


            # ----------------------------------------------
            # Original
            # ----------------------------------------------

            with gc1:

                with st.container(
                    border=True,
                    key="gradcam_original"
                ):

                    st.markdown(
                        '<div class="ex-explain-title">'
                        'Original Fundus'
                        '</div>',
                        unsafe_allow_html=True
                    )


                    st.markdown(
                        '<div class="ex-explain-desc">'
                        'Input image provided to the model.'
                        '</div>',
                        unsafe_allow_html=True
                    )


                    st.image(
                        result["original_image"],
                        use_container_width=True
                    )


            # ----------------------------------------------
            # Heatmap
            # ----------------------------------------------

            with gc2:

                with st.container(
                    border=True,
                    key="gradcam_heatmap"
                ):

                    st.markdown(
                        '<div class="ex-explain-title">'
                        'Grad-CAM Heatmap'
                        '</div>',
                        unsafe_allow_html=True
                    )


                    st.markdown(
                        '<div class="ex-explain-desc">'
                        'Model activation regions.'
                        '</div>',
                        unsafe_allow_html=True
                    )


                    st.image(
                        result["heatmap"],
                        use_container_width=True
                    )


            # ----------------------------------------------
            # Overlay
            # ----------------------------------------------

            with gc3:

                with st.container(
                    border=True,
                    key="gradcam_overlay"
                ):

                    st.markdown(
                        '<div class="ex-explain-title">'
                        'Prediction Overlay'
                        '</div>',
                        unsafe_allow_html=True
                    )


                    st.markdown(
                        '<div class="ex-explain-desc">'
                        'Activation regions overlaid on '
                        'the original image.'
                        '</div>',
                        unsafe_allow_html=True
                    )


                    st.image(
                        result["overlay"],
                        use_container_width=True
                    )


            # ==================================================
            # Explanation Insight
            # ==================================================

            st.write("")


            with st.container(
                border=True
            ):

                st.markdown(
                    '<div class="ex-section-title" '
                    'style="font-size:17px;">'
                    'How to read the explanation'
                    '</div>',
                    unsafe_allow_html=True
                )


                st.markdown(
                    '<div class="ex-insight-item">'
                    '• Brighter heatmap regions indicate '
                    'stronger model activation.'
                    '</div>'
                    '<div class="ex-insight-item">'
                    '• The overlay shows where the model '
                    'focused during prediction.'
                    '</div>'
                    '<div class="ex-insight-item">'
                    '• Grad-CAM explains model attention '
                    'but does not establish a clinical diagnosis.'
                    '</div>',
                    unsafe_allow_html=True
                )


            # ==================================================
            # IDRiD LESION ANALYSIS
            # ==================================================

            lesion_result = (
                st.session_state.lesion_result
            )

            if (
                result["disease"] == "DR"
                and lesion_result is not None
            ):

                st.divider()

                st.markdown(
                    '<div class="ex-section-title">'
                    'Diabetic Retinopathy Lesion Analysis'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="ex-section-subtitle">'
                    'Automatic lesion-level analysis using the '
                    'trained IDRiD U-Net segmentation model.'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.info(
                    "The lesion findings below are AI-detected "
                    "segmentation results and are not a clinical "
                    "diagnosis."
                )

                statistics = lesion_result["statistics"]

                ma = statistics["Microaneurysms"]
                he = statistics["Haemorrhages"]
                ex = statistics["Hard Exudates"]
                se = statistics["Soft Exudates"]

                def render_lesion_card(title, item):
                    status = "Detected" if item["detected"] else "Not detected"
                    st.markdown(
                        f"""
                        <div class="ex-lesion-card">
                            <div class="ex-lesion-name">{title}</div>
                            <div class="ex-lesion-status">{status}</div>
                            <div class="ex-lesion-stat">
                                <b>Area:</b> {item["percentage"]:.3f}%
                            </div>
                            <div class="ex-lesion-stat">
                                <b>Regions:</b> {item["regions"]:,}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                l1, l2, l3, l4 = st.columns(4)

                with l1:
                    render_lesion_card("Microaneurysms", ma)

                with l2:
                    render_lesion_card("Haemorrhages", he)

                with l3:
                    render_lesion_card("Hard Exudates", ex)

                with l4:
                    render_lesion_card("Soft Exudates", se)

                st.write("")

                st.markdown(
                    '<div class="ex-explain-title">'
                    'Lesion Visualization'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="ex-explain-desc">'
                    'The visualization shows regions segmented by '
                    'the IDRiD U-Net model.'
                    '</div>',
                    unsafe_allow_html=True
                )

                overlay = lesion_detector.create_overlay(
                    lesion_result,
                    alpha=0.28
                )

                st.image(
                    overlay,
                    caption="AI-detected retinal lesion regions",
                    use_container_width=True
                )

                st.markdown(
                    """
                    <div class="ex-lesion-legend">
                        <span class="ex-lesion-legend-item">
                            <span class="ex-lesion-dot"
                                  style="background:#FF3B30;"></span>
                            Microaneurysms
                        </span>
                        <span class="ex-lesion-legend-item">
                            <span class="ex-lesion-dot"
                                  style="background:#7C3AED;"></span>
                            Haemorrhages
                        </span>
                        <span class="ex-lesion-legend-item">
                            <span class="ex-lesion-dot"
                                  style="background:#FFD60A;"></span>
                            Hard Exudates
                        </span>
                        <span class="ex-lesion-legend-item">
                            <span class="ex-lesion-dot"
                                  style="background:#34C759;"></span>
                            Soft Exudates
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ==================================================
            # Disclaimer
            # ==================================================

            st.write("")


            st.markdown(
                dedent(
                    """
                    <div class="ex-disclaimer">

                    <div class="ex-disclaimer-title">
                        Research &amp; Educational Prototype
                    </div>

                    <div class="ex-disclaimer-text">

                        EyeX is intended for research and
                        educational purposes. Model predictions
                        should not be considered a medical
                        diagnosis or a substitute for evaluation
                        by a qualified ophthalmologist.

                    </div>

                    </div>
                    """
                ),
                unsafe_allow_html=True
            )


# ======================================================
# About EyeX
# ======================================================

st.write("")


with st.expander(
    "About EyeX"
):

    about_col1, about_col2 = (
        st.columns(2)
    )


    with about_col1:

        st.markdown(
            "**Deep Learning Model**"
        )

        st.write(
            "EfficientNet-B3"
        )


        st.markdown(
            "**Classification Task**"
        )

        st.write(
            "4-class retinal fundus image classification"
        )


        st.markdown(
            "**Image Quality Assessment**"
        )

        st.write(
            "Sharpness, brightness, contrast and resolution"
        )


    with about_col2:

        st.markdown(
            "**Disease Classes**"
        )

        st.write(
            "Healthy, DR, Glaucoma, AMD"
        )


        st.markdown(
            "**Explainability**"
        )

        st.write(
            "Grad-CAM"
        )


        st.markdown(
            "**Purpose**"
        )

        st.write(
            "AI-assisted research and educational prototype"
        )


# ======================================================
# Footer
# ======================================================

st.markdown(
    dedent(
        """
        <div class="ex-footer">

        <div class="ex-footer-brand">
            EyeX — Explainable Retinal AI
        </div>

        <div class="ex-footer-sub">
            EfficientNet-B3 • Image Quality Assessment
            • Grad-CAM • Research Prototype
        </div>

        </div>
        """.strip()
    ),
    unsafe_allow_html=True,
)