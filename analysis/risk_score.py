"""
EyeX Risk Score Engine

Purpose:
    Combines EyeX findings into a transparent project-level
    screening risk score.

Important:
    This is NOT a clinically validated medical risk calculator.
    It is intended for project-level screening and prioritization.

Inputs may include:
    - Disease classification
    - Classification confidence
    - Disease severity
    - Lesion findings
    - Optic-disc findings
    - Image quality

Output:
    - Risk score (0-100)
    - Risk category
    - Priority
    - Contributing factors
    - Limitations
"""


class RiskScoreEngine:
    """
    Transparent rule-based risk scoring engine for EyeX.
    """

    def __init__(self):
        pass

    # =========================================================
    # MAIN FUNCTION
    # =========================================================

    def calculate(
        self,
        disease,
        classification_confidence=None,
        severity_level=None,
        lesion_findings=None,
        optic_disc_findings=None,
        image_quality=None,
    ):
        """
        Calculate a project-level risk score.

        Parameters
        ----------
        disease : str
            Disease predicted by the classification model.

        classification_confidence : float, optional
            Model confidence. Can be supplied as decimal
            (0.0-1.0) or percentage (0-100).

        severity_level : int, optional
            Severity level produced by DiseaseSeverityEngine.

        lesion_findings : dict, optional
            Lesion counts from lesion detection.

        optic_disc_findings : dict, optional
            Optic disc/cup findings.

        image_quality : str, optional
            Image quality category such as:
            Good, Acceptable, Poor.

        Returns
        -------
        dict
            Structured risk assessment.
        """

        disease = self._normalize_disease(disease)
        confidence = self._normalize_confidence(
            classification_confidence
        )

        score = 0.0
        factors = []

        # ---------------------------------------------------------
        # 1. DISEASE CONTRIBUTION
        # ---------------------------------------------------------

        disease_points = {
            "Healthy": 0,
            "DR": 25,
            "Glaucoma": 30,
            "AMD": 25,
        }

        disease_score = disease_points.get(
            disease,
            20
        )

        score += disease_score

        if disease == "Healthy":
            factors.append(
                "No disease detected by the classification model."
            )
        else:
            factors.append(
                f"{disease} classification contributes "
                f"{disease_score} risk points."
            )

        # ---------------------------------------------------------
        # 2. CLASSIFICATION CONFIDENCE
        # ---------------------------------------------------------

        if confidence is not None:

            if confidence >= 90:
                confidence_points = 10

            elif confidence >= 75:
                confidence_points = 7

            elif confidence >= 60:
                confidence_points = 4

            else:
                confidence_points = 1

            # Confidence contributes only a limited amount.
            score += confidence_points

            factors.append(
                f"Classification confidence: {confidence:.2f}%."
            )

        # ---------------------------------------------------------
        # 3. SEVERITY CONTRIBUTION
        # ---------------------------------------------------------

        if severity_level is not None:

            try:
                severity_level = int(severity_level)
            except (TypeError, ValueError):
                severity_level = None

        if severity_level is not None:

            severity_points = {
                0: 0,
                1: 10,
                2: 20,
                3: 30,
            }

            severity_score = severity_points.get(
                severity_level,
                0
            )

            score += severity_score

            factors.append(
                f"Severity level {severity_level} contributes "
                f"{severity_score} risk points."
            )

        # ---------------------------------------------------------
        # 4. LESION CONTRIBUTION
        # ---------------------------------------------------------

        if lesion_findings:

            lesion_score, lesion_factors = (
                self._calculate_lesion_score(
                    lesion_findings
                )
            )

            score += lesion_score
            factors.extend(lesion_factors)

        # ---------------------------------------------------------
        # 5. OPTIC DISC CONTRIBUTION
        # ---------------------------------------------------------

        if optic_disc_findings:

            optic_score, optic_factors = (
                self._calculate_optic_disc_score(
                    optic_disc_findings
                )
            )

            score += optic_score
            factors.extend(optic_factors)

        # ---------------------------------------------------------
        # 6. IMAGE QUALITY
        # ---------------------------------------------------------

        quality_score, quality_factor = (
            self._calculate_quality_score(
                image_quality
            )
        )

        score += quality_score

        if quality_factor:
            factors.append(quality_factor)

        # ---------------------------------------------------------
        # LIMIT SCORE TO 100
        # ---------------------------------------------------------

        score = min(
            max(score, 0),
            100
        )

        score = round(score, 2)

        # ---------------------------------------------------------
        # RISK CATEGORY
        # ---------------------------------------------------------

        category, priority = (
            self._get_risk_category(score)
        )

        return {
            "disease": disease,
            "risk_score": score,
            "risk_category": category,
            "priority": priority,
            "classification_confidence": confidence,
            "severity_level": severity_level,
            "contributing_factors": factors,
            "limitations": (
                "This is a project-level screening and "
                "prioritization score. It is not a clinically "
                "validated medical risk calculator and should "
                "not be used as a standalone diagnostic decision."
            ),
        }

    # =========================================================
    # LESION SCORE
    # =========================================================

    def _calculate_lesion_score(
        self,
        findings
    ):
        """
        Calculate contribution from retinal lesion findings.

        Expected keys:
            microaneurysms
            haemorrhages
            hard_exudates
            soft_exudates
        """

        ma = self._safe_count(
            findings.get("microaneurysms", 0)
        )

        he = self._safe_count(
            findings.get("haemorrhages", 0)
        )

        ex = self._safe_count(
            findings.get("hard_exudates", 0)
        )

        se = self._safe_count(
            findings.get("soft_exudates", 0)
        )

        score = 0
        factors = []

        # Microaneurysms
        if ma > 0:
            points = min(ma, 10)
            score += points

            factors.append(
                f"{ma} microaneurysm(s) detected."
            )

        # Haemorrhages
        if he > 0:
            points = min(he * 2, 15)
            score += points

            factors.append(
                f"{he} haemorrhage(s) detected."
            )

        # Hard exudates
        if ex > 0:
            points = min(ex, 10)
            score += points

            factors.append(
                f"{ex} hard exudate(s) detected."
            )

        # Soft exudates
        if se > 0:
            points = min(se * 2, 15)
            score += points

            factors.append(
                f"{se} soft exudate(s) detected."
            )

        return score, factors

    # =========================================================
    # OPTIC DISC SCORE
    # =========================================================

    def _calculate_optic_disc_score(
        self,
        findings
    ):
        """
        Calculate contribution from optic-disc findings.

        Expected key:
            cup_to_disc_ratio
        """

        cdr = findings.get(
            "cup_to_disc_ratio"
        )

        if cdr is None:
            return 0, []

        try:
            cdr = float(cdr)
        except (TypeError, ValueError):
            return 0, []

        factors = []

        if cdr >= 0.70:
            score = 20

        elif cdr >= 0.50:
            score = 10

        else:
            score = 0

        if score > 0:
            factors.append(
                f"Cup-to-disc ratio: {cdr:.2f}."
            )

        return score, factors

    # =========================================================
    # IMAGE QUALITY SCORE
    # =========================================================

    def _calculate_quality_score(
        self,
        image_quality
    ):
        """
        Poor image quality increases uncertainty.

        Therefore the score receives a small uncertainty
        contribution rather than treating poor quality as
        disease severity.
        """

        if image_quality is None:
            return 0, None

        quality = str(
            image_quality
        ).strip().lower()

        if quality == "poor":
            return (
                5,
                "Poor image quality increases assessment uncertainty."
            )

        if quality in {
            "acceptable",
            "fair",
            "moderate"
        }:
            return (
                2,
                "Moderate image quality introduces some uncertainty."
            )

        if quality == "good":
            return (
                0,
                "Good image quality."
            )

        return 0, None

    # =========================================================
    # RISK CATEGORY
    # =========================================================

    @staticmethod
    def _get_risk_category(score):

        if score < 25:
            return (
                "Low",
                "Routine"
            )

        if score < 50:
            return (
                "Moderate",
                "Review"
            )

        if score < 75:
            return (
                "High",
                "Priority Review"
            )

        return (
            "Very High",
            "Urgent Review"
        )

    # =========================================================
    # NORMALIZATION HELPERS
    # =========================================================

    @staticmethod
    def _normalize_disease(
        disease
    ):

        if disease is None:
            return "Unknown"

        disease = str(
            disease
        ).strip()

        aliases = {
            "healthy": "Healthy",
            "normal": "Healthy",
            "dr": "DR",
            "diabetic retinopathy": "DR",
            "glaucoma": "Glaucoma",
            "amd": "AMD",
            "age-related macular degeneration": "AMD",
        }

        return aliases.get(
            disease.lower(),
            disease
        )

    @staticmethod
    def _normalize_confidence(
        confidence
    ):

        if confidence is None:
            return None

        try:
            confidence = float(
                confidence
            )
        except (TypeError, ValueError):
            return None

        if 0.0 <= confidence <= 1.0:
            confidence *= 100.0

        confidence = min(
            max(confidence, 0),
            100
        )

        return round(
            confidence,
            2
        )

    @staticmethod
    def _safe_count(
        value
    ):

        try:
            value = int(value)
            return max(value, 0)

        except (TypeError, ValueError):
            return 0


# =============================================================
# TESTS
# =============================================================

if __name__ == "__main__":

    engine = RiskScoreEngine()

    print("\n" + "=" * 60)
    print("EyeX Risk Score Engine")
    print("=" * 60)

    # ---------------------------------------------------------
    # TEST 1 — HEALTHY
    # ---------------------------------------------------------

    result = engine.calculate(
        disease="Healthy",
        classification_confidence=99.2,
        severity_level=0,
        image_quality="Good",
    )

    print("\nTEST 1 — HEALTHY")
    print(result)

    # ---------------------------------------------------------
    # TEST 2 — DR WITHOUT LESIONS
    # ---------------------------------------------------------

    result = engine.calculate(
        disease="DR",
        classification_confidence=91.5,
        image_quality="Good",
    )

    print("\nTEST 2 — DR WITHOUT LESIONS")
    print(result)

    # ---------------------------------------------------------
    # TEST 3 — DR WITH LESIONS
    # ---------------------------------------------------------

    result = engine.calculate(
        disease="DR",
        classification_confidence=91.5,
        severity_level=2,
        lesion_findings={
            "microaneurysms": 12,
            "haemorrhages": 4,
            "hard_exudates": 8,
            "soft_exudates": 0,
        },
        image_quality="Good",
    )

    print("\nTEST 3 — DR WITH LESIONS")
    print(result)

    # ---------------------------------------------------------
    # TEST 4 — GLAUCOMA
    # ---------------------------------------------------------

    result = engine.calculate(
        disease="Glaucoma",
        classification_confidence=88.7,
        image_quality="Good",
    )

    print("\nTEST 4 — GLAUCOMA")
    print(result)

    # ---------------------------------------------------------
    # TEST 5 — GLAUCOMA WITH C/D RATIO
    # ---------------------------------------------------------

    result = engine.calculate(
        disease="Glaucoma",
        classification_confidence=88.7,
        severity_level=3,
        optic_disc_findings={
            "cup_to_disc_ratio": 0.72
        },
        image_quality="Good",
    )

    print("\nTEST 5 — GLAUCOMA WITH C/D RATIO")
    print(result)

    # ---------------------------------------------------------
    # TEST 6 — AMD
    # ---------------------------------------------------------

    result = engine.calculate(
        disease="AMD",
        classification_confidence=95.4,
        image_quality="Good",
    )

    print("\nTEST 6 — AMD")
    print(result)

    # ---------------------------------------------------------
    # TEST 7 — POOR IMAGE QUALITY
    # ---------------------------------------------------------

    result = engine.calculate(
        disease="DR",
        classification_confidence=91.5,
        severity_level=2,
        lesion_findings={
            "microaneurysms": 10,
            "haemorrhages": 2,
            "hard_exudates": 4,
            "soft_exudates": 0,
        },
        image_quality="Poor",
    )

    print("\nTEST 7 — POOR IMAGE QUALITY")
    print(result)

    print("\n" + "=" * 60)
    print("All risk-score tests completed.")
    print("=" * 60)