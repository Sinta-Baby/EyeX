"""
==========================================================
EyeX
AI Retinal Screening PDF Report
==========================================================

Includes only currently implemented EyeX modules:

1. Uploaded Fundus Image
2. Image Quality Assessment
3. AI Screening Result
4. Diabetic Retinopathy Lesion Assessment
5. AI Model Comparison
6. Grad-CAM Explainability
7. Lesion Visualization
8. Medical Disclaimer

No unimplemented clinical modules are included.
==========================================================
"""

import os
from io import BytesIO
from datetime import datetime

import numpy as np
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle,
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)


# ==========================================================
# REPORT ID
# ==========================================================

def generate_report_id():
    """Generate a unique EyeX report ID."""

    return datetime.now().strftime(
        "EYEX-%Y%m%d-%H%M%S"
    )


# ==========================================================
# IMAGE CONVERSION
# ==========================================================

def _convert_to_pil(image):

    if image is None:
        return None

    # PIL Image
    if isinstance(
        image,
        PILImage.Image
    ):
        return image.convert("RGB")

    # File path
    if isinstance(
        image,
        str
    ):

        if os.path.exists(image):

            return PILImage.open(
                image
            ).convert("RGB")

        return None

    # NumPy array
    if isinstance(
        image,
        np.ndarray
    ):

        array = image

        if (
            array.ndim == 4
            and array.shape[0] == 1
        ):
            array = array[0]

        if np.issubdtype(
            array.dtype,
            np.floating
        ):

            if array.max() <= 1.0:
                array = array * 255.0

            array = np.clip(
                array,
                0,
                255
            ).astype(np.uint8)

        else:

            array = np.clip(
                array,
                0,
                255
            ).astype(np.uint8)

        if array.ndim == 2:

            return PILImage.fromarray(
                array,
                mode="L"
            ).convert("RGB")

        if array.ndim == 3:

            if array.shape[2] == 4:

                return PILImage.fromarray(
                    array,
                    mode="RGBA"
                ).convert("RGB")

            return PILImage.fromarray(
                array,
                mode="RGB"
            )

    return None


def _image_to_buffer(image):

    pil_image = _convert_to_pil(
        image
    )

    if pil_image is None:
        return None

    buffer = BytesIO()

    pil_image.save(
        buffer,
        format="PNG"
    )

    buffer.seek(0)

    return buffer


def _create_reportlab_image(
    image,
    max_width=80 * mm,
    max_height=70 * mm
):

    buffer = _image_to_buffer(
        image
    )

    if buffer is None:
        return None

    pil_image = PILImage.open(
        buffer
    )

    width, height = pil_image.size

    if (
        width <= 0
        or height <= 0
    ):
        return None

    scale = min(
        max_width / width,
        max_height / height
    )

    display_width = width * scale
    display_height = height * scale

    buffer.seek(0)

    return Image(
        buffer,
        width=display_width,
        height=display_height
    )


# ==========================================================
# VALUE HELPERS
# ==========================================================

def _safe_value(
    value,
    default="Not available"
):

    if value is None:
        return default

    return str(value)


def _confidence(value):

    try:

        return (
            f"{float(value) * 100:.2f}%"
        )

    except (
        TypeError,
        ValueError
    ):

        return "Not available"


def _percentage(value):

    try:

        return (
            f"{float(value):.3f}%"
        )

    except (
        TypeError,
        ValueError
    ):

        return "Not available"


# ==========================================================
# STYLES
# ==========================================================

def _build_styles():

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="EyeXTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            alignment=TA_CENTER,
            spaceAfter=6,
            textColor=colors.HexColor(
                "#173B65"
            ),
        )
    )

    styles.add(
        ParagraphStyle(
            name="EyeXSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor(
                "#5B6573"
            ),
            spaceAfter=12,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=colors.HexColor(
                "#173B65"
            ),
            spaceBefore=8,
            spaceAfter=8,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SubHeading",
            parent=styles["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor(
                "#26384D"
            ),
            spaceBefore=6,
            spaceAfter=5,
        )
    )

    styles.add(
        ParagraphStyle(
            name="BodyEyeX",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor(
                "#333333"
            ),
            spaceAfter=5,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SmallEyeX",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor(
                "#5B6573"
            ),
        )
    )

    styles.add(
        ParagraphStyle(
            name="ClinicalResult",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=17,
            textColor=colors.HexColor(
                "#173B65"
            ),
        )
    )

    styles.add(
        ParagraphStyle(
            name="DisclaimerEyeX",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=12,
            textColor=colors.HexColor(
                "#4B5563"
            ),
        )
    )

    return styles


# ==========================================================
# TABLE STYLE
# ==========================================================

def _style_table(
    table,
    header=True
):

    commands = [

        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.5,
            colors.HexColor(
                "#D6DCE5"
            )
        ),

        (
            "VALIGN",
            (0, 0),
            (-1, -1),
            "MIDDLE"
        ),

        (
            "FONTNAME",
            (0, 0),
            (-1, -1),
            "Helvetica"
        ),

        (
            "FONTSIZE",
            (0, 0),
            (-1, -1),
            8.5
        ),

        (
            "TOPPADDING",
            (0, 0),
            (-1, -1),
            6
        ),

        (
            "BOTTOMPADDING",
            (0, 0),
            (-1, -1),
            6
        ),
    ]

    if header:

        commands.extend([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor(
                    "#173B65"
                )
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
        ])

    table.setStyle(
        TableStyle(commands)
    )

    return table


# ==========================================================
# PAGE HEADER / FOOTER
# ==========================================================

def _add_page_number(
    canvas,
    doc
):

    canvas.saveState()

    page_width, page_height = A4

    canvas.setStrokeColor(
        colors.HexColor(
            "#D6DCE5"
        )
    )

    canvas.line(
        18 * mm,
        page_height - 15 * mm,
        page_width - 18 * mm,
        page_height - 15 * mm
    )

    canvas.setFont(
        "Helvetica",
        7.5
    )

    canvas.setFillColor(
        colors.HexColor(
            "#6B7280"
        )
    )

    canvas.drawString(
        18 * mm,
        page_height - 11 * mm,
        "EyeX • AI Retinal Screening Prototype"
    )

    canvas.line(
        18 * mm,
        14 * mm,
        page_width - 18 * mm,
        14 * mm
    )

    canvas.drawString(
        18 * mm,
        9 * mm,
        "EyeX"
    )

    canvas.drawRightString(
        page_width - 18 * mm,
        9 * mm,
        f"Page {doc.page}"
    )

    canvas.restoreState()


# ==========================================================
# MAIN PDF GENERATOR
# ==========================================================

def generate_pdf_report(
    output_path,
    image_path=None,
    quality_result=None,
    analysis_result=None,
    classification_comparison=None,
    lesion_result=None,
    lesion_overlay=None,
    report_id=None,
):

    # ======================================================
    # OUTPUT DIRECTORY
    # ======================================================

    output_directory = os.path.dirname(
        os.path.abspath(
            output_path
        )
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # ======================================================
    # REPORT ID
    # ======================================================

    if report_id is None:

        report_id = generate_report_id()

    # ======================================================
    # STYLES
    # ======================================================

    styles = _build_styles()

    # ======================================================
    # DOCUMENT
    # ======================================================

    document = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=22 * mm,
        bottomMargin=20 * mm,
        title="EyeX AI Retinal Screening Report",
        author="EyeX",
    )

    story = []

    # ======================================================
    # TITLE
    # ======================================================

    story.append(
        Spacer(
            1,
            8 * mm
        )
    )

    story.append(
        Paragraph(
            "EyeX",
            styles["EyeXTitle"]
        )
    )

    story.append(
        Paragraph(
            "AI Retinal Screening Report",
            styles["EyeXSubtitle"]
        )
    )

    story.append(
        Paragraph(
            "Research Prototype",
            styles["EyeXSubtitle"]
        )
    )

    # ======================================================
    # REPORT DETAILS
    # ======================================================

    report_info = [

        [
            Paragraph(
                "<b>Report ID</b>",
                styles["BodyEyeX"]
            ),

            Paragraph(
                _safe_value(
                    report_id
                ),
                styles["BodyEyeX"]
            ),
        ],

        [
            Paragraph(
                "<b>Generated</b>",
                styles["BodyEyeX"]
            ),

            Paragraph(
                datetime.now().strftime(
                    "%d %B %Y, %I:%M:%S %p"
                ),
                styles["BodyEyeX"]
            ),
        ],
    ]

    report_table = Table(
        report_info,
        colWidths=[
            40 * mm,
            120 * mm
        ]
    )

    report_table.setStyle(
        TableStyle([

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.6,
                colors.HexColor(
                    "#D6DCE5"
                )
            ),

            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.4,
                colors.HexColor(
                    "#E5E7EB"
                )
            ),

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor(
                    "#F4F7FB"
                )
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                7
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                7
            ),
        ])
    )

    story.append(
        report_table
    )

    story.append(
        Spacer(
            1,
            6 * mm
        )
    )

    # ======================================================
    # 1. UPLOADED FUNDUS IMAGE
    # ======================================================

    story.append(
        Paragraph(
            "1. Uploaded Fundus Image",
            styles["SectionHeading"]
        )
    )

    uploaded_image = _create_reportlab_image(
        image_path,
        max_width=145 * mm,
        max_height=90 * mm
    )

    if uploaded_image is not None:

        image_table = Table(
            [[uploaded_image]],
            colWidths=[
                160 * mm
            ]
        )

        image_table.setStyle(
            TableStyle([

                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.6,
                    colors.HexColor(
                        "#D6DCE5"
                    )
                ),

                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER"
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),
            ])
        )

        story.append(
            image_table
        )

    # ======================================================
    # 2. IMAGE QUALITY
    # ======================================================

    if quality_result is not None:

        story.append(
            Paragraph(
                "2. Image Quality Assessment",
                styles["SectionHeading"]
            )
        )

        quality_data = [

            [
                "Metric",
                "Value",
                "Result"
            ],

            [
                "Sharpness",

                (
                    f"{float(quality_result['sharpness']):.2f}"
                    if quality_result.get(
                        "sharpness"
                    ) is not None
                    else "Not available"
                ),

                (
                    "Acceptable"
                    if quality_result.get(
                        "sharpness_ok",
                        False
                    )
                    else "Check"
                ),
            ],

            [
                "Brightness",

                (
                    f"{float(quality_result['brightness']):.2f}"
                    if quality_result.get(
                        "brightness"
                    ) is not None
                    else "Not available"
                ),

                (
                    "Acceptable"
                    if quality_result.get(
                        "brightness_ok",
                        False
                    )
                    else "Check"
                ),
            ],

            [
                "Contrast",

                (
                    f"{float(quality_result['contrast']):.2f}"
                    if quality_result.get(
                        "contrast"
                    ) is not None
                    else "Not available"
                ),

                (
                    "Acceptable"
                    if quality_result.get(
                        "contrast_ok",
                        False
                    )
                    else "Check"
                ),
            ],

            [
                "Resolution",

                (
                    f"{quality_result.get('width', 'N/A')} × "
                    f"{quality_result.get('height', 'N/A')}"
                ),

                (
                    "Acceptable"
                    if quality_result.get(
                        "resolution_ok",
                        False
                    )
                    else "Check"
                ),
            ],

            [
                "Overall Quality",

                _safe_value(
                    quality_result.get(
                        "quality"
                    )
                ),

                (
                    "Passed"
                    if quality_result.get(
                        "passed_checks",
                        0
                    ) == 4
                    else "Review"
                ),
            ],
        ]

        quality_table = Table(
            quality_data,
            colWidths=[
                60 * mm,
                55 * mm,
                45 * mm
            ]
        )

        story.append(
            _style_table(
                quality_table
            )
        )

        story.append(
            Spacer(
                1,
                3 * mm
            )
        )

        story.append(
            Paragraph(
                f"{quality_result.get('passed_checks', 0)} "
                "of 4 quality checks passed.",
                styles["SmallEyeX"]
            )
        )

    # ======================================================
    # 3. AI SCREENING RESULT
    # ======================================================

    if analysis_result is not None:

        story.append(
            Paragraph(
                "3. AI Screening Result",
                styles["SectionHeading"]
            )
        )

        disease = analysis_result.get(
            "disease",
            "Not available"
        )

        confidence = analysis_result.get(
            "confidence"
        )

        # --------------------------------------------------
        # Main result box
        # --------------------------------------------------

        result_data = [

            [
                Paragraph(
                    "<b>Detected condition</b>",
                    styles["BodyEyeX"]
                ),

                Paragraph(
                    f"<b>{_safe_value(disease)}</b>",
                    styles["ClinicalResult"]
                ),
            ],

            [
                Paragraph(
                    "<b>Classification model</b>",
                    styles["BodyEyeX"]
                ),

                Paragraph(
                    "EfficientNet-B3",
                    styles["BodyEyeX"]
                ),
            ],

            [
                Paragraph(
                    "<b>Model confidence</b>",
                    styles["BodyEyeX"]
                ),

                Paragraph(
                    _confidence(
                        confidence
                    ),
                    styles["BodyEyeX"]
                ),
            ],
        ]

        result_table = Table(
            result_data,
            colWidths=[
                55 * mm,
                105 * mm
            ]
        )

        result_table.setStyle(
            TableStyle([

                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.8,
                    colors.HexColor(
                        "#B9C7D8"
                    )
                ),

                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor(
                        "#DDE3EA"
                    )
                ),

                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor(
                        "#F4F7FB"
                    )
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),
            ])
        )

        story.append(
            result_table
        )

        story.append(
            Spacer(
                1,
                5 * mm
            )
        )

        # --------------------------------------------------
        # DR-SPECIFIC CLINICAL SECTION
        # --------------------------------------------------

        if disease == "DR":

            story.append(
                Paragraph(
                    "Diabetic Retinopathy — Lesion Assessment",
                    styles["SubHeading"]
                )
            )

            story.append(
                Paragraph(
                    "The EyeX diabetic retinopathy analysis "
                    "assesses four retinal lesion categories "
                    "using the trained IDRiD U-Net segmentation "
                    "model: Microaneurysms, Haemorrhages, "
                    "Hard Exudates, and Soft Exudates.",
                    styles["BodyEyeX"]
                )
            )

            story.append(
                Paragraph(
                    "The findings below represent regions "
                    "identified by the EyeX lesion segmentation "
                    "model for the uploaded fundus image.",
                    styles["SmallEyeX"]
                )
            )

    # ======================================================
    # 4. DR LESION ASSESSMENT
    # ======================================================

    if (
        analysis_result is not None
        and analysis_result.get(
            "disease"
        ) == "DR"
        and lesion_result is not None
    ):

        story.append(
            Paragraph(
                "4. Diabetic Retinopathy Lesion Assessment",
                styles["SectionHeading"]
            )
        )

        statistics = lesion_result.get(
            "statistics",
            {}
        )

        lesion_data = [

            [
                "Lesion Category",
                "AI Finding",
                "Area",
                "Regions"
            ]
        ]

        lesion_names = [

            "Microaneurysms",
            "Haemorrhages",
            "Hard Exudates",
            "Soft Exudates",
        ]

        for lesion_name in lesion_names:

            item = statistics.get(
                lesion_name,
                {}
            )

            detected = item.get(
                "detected",
                False
            )

            lesion_data.append([

                lesion_name,

                (
                    "Detected"
                    if detected
                    else "Not detected"
                ),

                _percentage(
                    item.get(
                        "percentage"
                    )
                ),

                _safe_value(
                    item.get(
                        "regions"
                    )
                ),
            ])

        lesion_table = Table(
            lesion_data,
            colWidths=[
                55 * mm,
                40 * mm,
                30 * mm,
                35 * mm
            ]
        )

        story.append(
            _style_table(
                lesion_table
            )
        )

        story.append(
            Spacer(
                1,
                4 * mm
            )
        )

        story.append(
            Paragraph(
                "Lesion categories assessed",
                styles["SubHeading"]
            )
        )

        lesion_category_data = [

            [
                Paragraph(
                    "<b>Microaneurysms</b>",
                    styles["BodyEyeX"]
                ),

                Paragraph(
                    "AI-assessed retinal lesion category.",
                    styles["SmallEyeX"]
                ),
            ],

            [
                Paragraph(
                    "<b>Haemorrhages</b>",
                    styles["BodyEyeX"]
                ),

                Paragraph(
                    "AI-assessed retinal lesion category.",
                    styles["SmallEyeX"]
                ),
            ],

            [
                Paragraph(
                    "<b>Hard Exudates</b>",
                    styles["BodyEyeX"]
                ),

                Paragraph(
                    "AI-assessed retinal lesion category.",
                    styles["SmallEyeX"]
                ),
            ],

            [
                Paragraph(
                    "<b>Soft Exudates</b>",
                    styles["BodyEyeX"]
                ),

                Paragraph(
                    "AI-assessed retinal lesion category.",
                    styles["SmallEyeX"]
                ),
            ],
        ]

        lesion_category_table = Table(
            lesion_category_data,
            colWidths=[
                50 * mm,
                110 * mm
            ]
        )

        lesion_category_table.setStyle(
            TableStyle([

                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(
                        "#D6DCE5"
                    )
                ),

                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor(
                        "#E5E7EB"
                    )
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),

                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor(
                        "#F4F7FB"
                    )
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
            ])
        )

        story.append(
            lesion_category_table
        )

    # ======================================================
    # 5. AI MODEL COMPARISON
    # ======================================================

    if classification_comparison is not None:

        story.append(
            PageBreak()
        )

        story.append(
            Paragraph(
                "5. AI Model Comparison",
                styles["SectionHeading"]
            )
        )

        story.append(
            Paragraph(
                "The uploaded fundus image was evaluated by "
                "the three trained EyeX classification models.",
                styles["BodyEyeX"]
            )
        )

        models = (
            classification_comparison.get(
                "models",
                {}
            )
        )

        comparison_data = [

            [
                "Model",
                "Prediction",
                "Confidence"
            ]
        ]

        model_order = [

            "EfficientNet-B3",
            "ConvNeXt-Tiny",
            "Swin-Tiny",
        ]

        for model_name in model_order:

            model_result = models.get(
                model_name
            )

            if model_result is None:
                continue

            prediction = model_result.get(
                "prediction",
                model_result.get(
                    "disease"
                )
            )

            confidence = model_result.get(
                "confidence"
            )

            comparison_data.append([

                model_name,

                _safe_value(
                    prediction
                ),

                _confidence(
                    confidence
                ),
            ])

        if len(comparison_data) > 1:

            comparison_table = Table(
                comparison_data,
                colWidths=[
                    65 * mm,
                    55 * mm,
                    40 * mm
                ]
            )

            story.append(
                _style_table(
                    comparison_table
                )
            )

        # --------------------------------------------------
        # Agreement
        # --------------------------------------------------

        agreement = (
            classification_comparison.get(
                "agreement",
                {}
            )
        )

        agreement_data = [

            [
                "Comparison Result",
                "Value"
            ],

            [
                "Model Agreement",

                (
                    f"{agreement.get('agreement_count', 'N/A')}/"
                    f"{agreement.get('total_models', 'N/A')}"
                ),
            ],

            [
                "Comparison Prediction",

                _safe_value(
                    agreement.get(
                        "final_prediction"
                    )
                ),
            ],
        ]

        agreement_table = Table(
            agreement_data,
            colWidths=[
                75 * mm,
                85 * mm
            ]
        )

        story.append(
            Spacer(
                1,
                5 * mm
            )
        )

        story.append(
            _style_table(
                agreement_table
            )
        )

        story.append(
            Spacer(
                1,
                3 * mm
            )
        )

        story.append(
            Paragraph(
                "Confidence values represent predictions "
                "for this uploaded image and are distinct "
                "from dataset-level accuracy.",
                styles["SmallEyeX"]
            )
        )

    # ======================================================
    # 6. GRAD-CAM
    # ======================================================

    if analysis_result is not None:

        story.append(
            Paragraph(
                "6. Explainable AI — Grad-CAM",
                styles["SectionHeading"]
            )
        )

        story.append(
            Paragraph(
                "Grad-CAM highlights regions of the fundus "
                "image that contributed to the model's "
                "prediction.",
                styles["BodyEyeX"]
            )
        )

        gradcam_images = [

            (
                "Original Fundus",

                analysis_result.get(
                    "original_image"
                )
            ),

            (
                "Grad-CAM Heatmap",

                analysis_result.get(
                    "heatmap"
                )
            ),

            (
                "Prediction Overlay",

                analysis_result.get(
                    "overlay"
                )
            ),
        ]

        gradcam_cells = []

        for (
            title,
            image
        ) in gradcam_images:

            report_image = (
                _create_reportlab_image(
                    image,
                    max_width=50 * mm,
                    max_height=50 * mm
                )
            )

            if report_image is not None:

                cell = [

                    Paragraph(
                        f"<b>{title}</b>",
                        styles["SmallEyeX"]
                    ),

                    Spacer(
                        1,
                        2 * mm
                    ),

                    report_image,
                ]

            else:

                cell = [

                    Paragraph(
                        f"<b>{title}</b>",
                        styles["SmallEyeX"]
                    ),

                    Spacer(
                        1,
                        2 * mm
                    ),

                    Paragraph(
                        "Image unavailable",
                        styles["SmallEyeX"]
                    ),
                ]

            gradcam_cells.append(
                cell
            )

        gradcam_table = Table(
            [gradcam_cells],
            colWidths=[
                53 * mm,
                53 * mm,
                53 * mm
            ]
        )

        gradcam_table.setStyle(
            TableStyle([

                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(
                        "#D6DCE5"
                    )
                ),

                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(
                        "#E5E7EB"
                    )
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),

                (
                    "ALIGN",
                    (0, 0),
                    (-1, -1),
                    "CENTER"
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7
                ),
            ])
        )

        story.append(
            gradcam_table
        )

        story.append(
            Spacer(
                1,
                5 * mm
            )
        )

        explanation_data = [

            [
                Paragraph(
                    "<b>How to read the explanation</b>",
                    styles["BodyEyeX"]
                )
            ],

            [
                Paragraph(
                    "• Brighter heatmap regions indicate "
                    "stronger model activation.",
                    styles["BodyEyeX"]
                )
            ],

            [
                Paragraph(
                    "• The overlay shows where the model "
                    "focused during prediction.",
                    styles["BodyEyeX"]
                )
            ],

            [
                Paragraph(
                    "• Grad-CAM represents model attention "
                    "and does not establish a clinical "
                    "diagnosis.",
                    styles["BodyEyeX"]
                )
            ],
        ]

        explanation_table = Table(
            explanation_data,
            colWidths=[
                160 * mm
            ]
        )

        explanation_table.setStyle(
            TableStyle([

                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(
                        "#D6DCE5"
                    )
                ),

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#F4F7FB"
                    )
                ),

                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),

                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),

                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                ),

                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                ),
            ])
        )

        story.append(
            explanation_table
        )

    # ======================================================
    # 7. LESION VISUALIZATION
    # ======================================================

    if (
        lesion_result is not None
        and lesion_overlay is not None
    ):

        story.append(
            PageBreak()
        )

        story.append(
            Paragraph(
                "7. Lesion Visualization",
                styles["SectionHeading"]
            )
        )

        story.append(
            Paragraph(
                "The visualization shows regions segmented "
                "by the EyeX IDRiD U-Net model.",
                styles["BodyEyeX"]
            )
        )

        lesion_image = (
            _create_reportlab_image(
                lesion_overlay,
                max_width=145 * mm,
                max_height=90 * mm
            )
        )

        if lesion_image is not None:

            lesion_image_table = Table(
                [[lesion_image]],
                colWidths=[
                    160 * mm
                ]
            )

            lesion_image_table.setStyle(
                TableStyle([

                    (
                        "BOX",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        colors.HexColor(
                            "#D6DCE5"
                        )
                    ),

                    (
                        "ALIGN",
                        (0, 0),
                        (-1, -1),
                        "CENTER"
                    ),

                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        8
                    ),

                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        8
                    ),
                ])
            )

            story.append(
                lesion_image_table
            )

        story.append(
            Spacer(
                1,
                5 * mm
            )
        )

        story.append(
            Paragraph(
                "AI-detected retinal lesion regions",
                styles["SmallEyeX"]
            )
        )

    # ======================================================
    # 8. MEDICAL DISCLAIMER
    # ======================================================

    story.append(
        Spacer(
            1,
            8 * mm
        )
    )

    story.append(
        Paragraph(
            "8. Medical Disclaimer",
            styles["SectionHeading"]
        )
    )

    disclaimer_text = (
        "EyeX is a research prototype designed for "
        "AI-assisted retinal image analysis. The results "
        "presented in this report are generated by "
        "machine-learning models and should not be considered "
        "a medical diagnosis. Model predictions, confidence "
        "values, Grad-CAM visualizations, and lesion "
        "segmentation results should be interpreted by a "
        "qualified healthcare professional. This report does "
        "not replace professional medical examination, "
        "diagnosis, or treatment."
    )

    disclaimer_table = Table(
        [[
            Paragraph(
                disclaimer_text,
                styles["DisclaimerEyeX"]
            )
        ]],
        colWidths=[
            160 * mm
        ]
    )

    disclaimer_table.setStyle(
        TableStyle([

            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.7,
                colors.HexColor(
                    "#C9D2DF"
                )
            ),

            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                colors.HexColor(
                    "#F6F8FA"
                )
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                10
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                10
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                9
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                9
            ),
        ])
    )

    story.append(
        disclaimer_table
    )

    # ======================================================
    # BUILD PDF
    # ======================================================

    document.build(
        story,
        onFirstPage=_add_page_number,
        onLaterPages=_add_page_number,
    )

    return output_path