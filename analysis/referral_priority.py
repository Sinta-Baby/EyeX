"""
EyeX Referral / Priority System

Purpose:
    Converts EyeX screening findings into a transparent
    referral/review priority recommendation.

Important:
    This is a project-level prioritization system.
    It is NOT a medical diagnosis and does NOT determine
    whether a patient is experiencing a medical emergency.

The module considers:
    - Disease classification
    - Risk score
    - Severity level
    - Image quality
    - Availability of disease-specific findings

Output:
    - Referral priority
    - Recommended action
    - Reason
    - Supporting factors
    - Limitations
"""


class ReferralPriorityEngine:
    """
    Rule-based referral and review prioritization engine.
    """

    def __init__(self):
        pass

    # =========================================================
    # MAIN FUNCTION
    # =========================================================

    def assess(
        self,
        disease,
        risk_score=None,
        severity_level=None,
        image_quality=None,
        severity_status=None,
    ):
        """
        Determine EyeX review/referral priority.

        Parameters
        ----------
        disease : str
            Disease predicted by the classification model.

        risk_score : float
            Project-level risk score from RiskScoreEngine.

        severity_level : int, optional
            Severity level from DiseaseSeverityEngine.

        image_quality : str, optional
            Image quality category.

        severity_status : str, optional
            Text status returned by the severity engine, such as:
                "Requires lesion assessment"
                "Requires structural assessment"
                "Requires macular assessment"

        Returns
        -------
        dict
            Structured referral-priority recommendation.
        """

        disease = self._normalize_disease(disease)
        risk_score = self._normalize_score(risk_score)
        severity_level = self._normalize_severity(
            severity_level
        )

        factors = []

        # ---------------------------------------------------------
        # INVALID / MISSING RISK SCORE
        # ---------------------------------------------------------

        if risk_score is None:
            return self._insufficient_information(
                disease=disease,
                severity_level=severity_level,
                image_quality=image_quality,
                severity_status=severity_status,
            )

        # ---------------------------------------------------------
        # IMAGE QUALITY CHECK
        # ---------------------------------------------------------

        quality = self._normalize_quality(
            image_quality
        )

        # Poor-quality images should trigger re-acquisition
        # rather than automatically implying severe disease.
        if quality == "poor":
            factors.append(
                "Image quality is poor and may limit reliable assessment."
            )

            if risk_score < 50:
                return {
                    "disease": disease,
                    "risk_score": risk_score,
                    "priority": "Image Re-acquisition",
                    "priority_level": 2,
                    "recommended_action": (
                        "Repeat fundus image acquisition "
                        "before relying on the screening result."
                    ),
                    "reason": (
                        "The image quality may be insufficient "
                        "for reliable automated assessment."
                    ),
                    "supporting_factors": factors,
                    "limitations": self._limitations(),
                }

        # ---------------------------------------------------------
        # HEALTHY
        # ---------------------------------------------------------

        if disease == "Healthy":

            if risk_score < 25:
                return {
                    "disease": disease,
                    "risk_score": risk_score,
                    "priority": "Routine",
                    "priority_level": 1,
                    "recommended_action": (
                        "No priority referral indicated by the "
                        "current EyeX screening result."
                    ),
                    "reason": (
                        "No disease was detected and the project-level "
                        "risk score is low."
                    ),
                    "supporting_factors": factors,
                    "limitations": self._limitations(),
                }

        # ---------------------------------------------------------
        # VERY HIGH RISK
        # ---------------------------------------------------------

        if risk_score >= 75:

            factors.append(
                "Risk score is in the very-high range."
            )

            if severity_level is not None:
                factors.append(
                    f"Severity level: {severity_level}."
                )

            return {
                "disease": disease,
                "risk_score": risk_score,
                "priority": "High Priority Review",
                "priority_level": 4,
                "recommended_action": (
                    "Prioritize ophthalmologist review "
                    "of the EyeX screening findings."
                ),
                "reason": (
                    "The combined project-level screening findings "
                    "indicate high review priority."
                ),
                "supporting_factors": factors,
                "limitations": self._limitations(),
            }

        # ---------------------------------------------------------
        # HIGH RISK
        # ---------------------------------------------------------

        if risk_score >= 50:

            factors.append(
                "Risk score is in the high-risk range."
            )

            if severity_level is not None:
                factors.append(
                    f"Severity level: {severity_level}."
                )

            return {
                "disease": disease,
                "risk_score": risk_score,
                "priority": "Priority Review",
                "priority_level": 3,
                "recommended_action": (
                    "Recommend ophthalmologist review "
                    "of the screening findings."
                ),
                "reason": (
                    "The project-level screening risk is elevated."
                ),
                "supporting_factors": factors,
                "limitations": self._limitations(),
            }

        # ---------------------------------------------------------
        # MODERATE RISK
        # ---------------------------------------------------------

        if risk_score >= 25:

            factors.append(
                "Risk score is in the moderate range."
            )

            return {
                "disease": disease,
                "risk_score": risk_score,
                "priority": "Review",
                "priority_level": 2,
                "recommended_action": (
                    "Recommend routine ophthalmic review "
                    "based on the screening result."
                ),
                "reason": (
                    "The project-level screening result "
                    "warrants review."
                ),
                "supporting_factors": factors,
                "limitations": self._limitations(),
            }

        # ---------------------------------------------------------
        # LOW RISK
        # ---------------------------------------------------------

        return {
            "disease": disease,
            "risk_score": risk_score,
            "priority": "Routine",
            "priority_level": 1,
            "recommended_action": (
                "No elevated review priority indicated "
                "by the current screening score."
            ),
            "reason": (
                "The project-level screening risk is low."
            ),
            "supporting_factors": factors,
            "limitations": self._limitations(),
        }

    # =========================================================
    # INSUFFICIENT INFORMATION
    # =========================================================

    def _insufficient_information(
        self,
        disease,
        severity_level,
        image_quality,
        severity_status,
    ):

        factors = []

        if severity_status:
            factors.append(
                f"Severity status: {severity_status}."
            )

        if image_quality:
            factors.append(
                f"Image quality: {image_quality}."
            )

        return {
            "disease": disease,
            "risk_score": None,
            "priority": "Insufficient Information",
            "priority_level": None,
            "recommended_action": (
                "Complete the required EyeX analysis "
                "before assigning a review priority."
            ),
            "reason": (
                "A valid project-level risk score was not supplied."
            ),
            "supporting_factors": factors,
            "limitations": self._limitations(),
        }

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _normalize_disease(disease):

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
    def _normalize_score(score):

        if score is None:
            return None

        try:
            score = float(score)
        except (TypeError, ValueError):
            return None

        score = min(
            max(score, 0),
            100
        )

        return round(
            score,
            2
        )

    @staticmethod
    def _normalize_severity(
        severity
    ):

        if severity is None:
            return None

        try:
            return int(severity)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_quality(
        quality
    ):

        if quality is None:
            return None

        quality = str(
            quality
        ).strip().lower()

        if quality in {
            "poor",
            "bad",
            "unusable"
        }:
            return "poor"

        if quality in {
            "good",
            "acceptable",
            "fair",
            "moderate"
        }:
            return quality

        return None

    @staticmethod
    def _limitations():

        return (
            "This referral priority is generated from a "
            "project-level screening system. It is not a "
            "clinical diagnosis, emergency determination, "
            "or substitute for professional ophthalmic evaluation."
        )


# =============================================================
# TESTS
# =============================================================

if __name__ == "__main__":

    engine = ReferralPriorityEngine()

    print("\n" + "=" * 60)
    print("EyeX Referral / Priority System")
    print("=" * 60)

    # ---------------------------------------------------------
    # TEST 1 — HEALTHY / LOW RISK
    # ---------------------------------------------------------

    result = engine.assess(
        disease="Healthy",
        risk_score=10,
        severity_level=0,
        image_quality="Good",
    )

    print("\nTEST 1 — HEALTHY")
    print(result)

    # ---------------------------------------------------------
    # TEST 2 — MODERATE RISK
    # ---------------------------------------------------------

    result = engine.assess(
        disease="DR",
        risk_score=35,
        image_quality="Good",
        severity_status="Requires lesion assessment",
    )

    print("\nTEST 2 — MODERATE RISK")
    print(result)

    # ---------------------------------------------------------
    # TEST 3 — HIGH RISK
    # ---------------------------------------------------------

    result = engine.assess(
        disease="DR",
        risk_score=65,
        severity_level=2,
        image_quality="Good",
    )

    print("\nTEST 3 — HIGH RISK")
    print(result)

    # ---------------------------------------------------------
    # TEST 4 — VERY HIGH RISK
    # ---------------------------------------------------------

    result = engine.assess(
        disease="DR",
        risk_score=81,
        severity_level=2,
        image_quality="Good",
    )

    print("\nTEST 4 — VERY HIGH RISK")
    print(result)

    # ---------------------------------------------------------
    # TEST 5 — GLAUCOMA STRUCTURAL FINDING
    # ---------------------------------------------------------

    result = engine.assess(
        disease="Glaucoma",
        risk_score=87,
        severity_level=3,
        image_quality="Good",
    )

    print("\nTEST 5 — GLAUCOMA")
    print(result)

    # ---------------------------------------------------------
    # TEST 6 — POOR IMAGE QUALITY
    # ---------------------------------------------------------

    result = engine.assess(
        disease="DR",
        risk_score=35,
        image_quality="Poor",
    )

    print("\nTEST 6 — POOR IMAGE QUALITY")
    print(result)

    # ---------------------------------------------------------
    # TEST 7 — MISSING RISK SCORE
    # ---------------------------------------------------------

    result = engine.assess(
        disease="AMD",
        risk_score=None,
        image_quality="Good",
        severity_status="Requires macular assessment",
    )

    print("\nTEST 7 — MISSING RISK SCORE")
    print(result)

    print("\n" + "=" * 60)
    print("All referral-priority tests completed.")
    print("=" * 60)