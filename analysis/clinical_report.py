"""
EyeX Automated Clinical Report Engine

Purpose:
    Combine EyeX analysis outputs into a structured,
    human-readable screening report.

This module does NOT perform diagnosis or medical staging.
It only organizes results produced by the EyeX analysis modules.

Current inputs:
    - Disease classification
    - Image quality
    - Disease severity
    - Lesion findings
    - Risk score
    - Referral priority
    - Grad-CAM availability

Output:
    - Structured report dictionary
    - Human-readable text report
"""


class ClinicalReportGenerator:
    """
    Generates a structured EyeX screening report.
    """

    DISCLAIMER = (
        "This is an AI-assisted screening result and not a "
        "clinical diagnosis. The result should be reviewed by "
        "a qualified ophthalmic professional."
    )

    def __init__(self):
        pass

    # =========================================================
    # MAIN REPORT GENERATOR
    # =========================================================

    def generate(
        self,
        disease,
        classification_confidence=None,
        image_quality=None,
        severity_result=None,
        risk_result=None,
        referral_result=None,
        lesion_findings=None,
        gradcam_available=False,
    ):
        """
        Generate a complete structured EyeX report.

        Parameters
        ----------
        disease : str
            Disease classification result.

        classification_confidence : float, optional
            Classification confidence.

        image_quality : str, optional
            Image quality result.

        severity_result : dict, optional
            Output from DiseaseSeverityEngine.

        risk_result : dict, optional
            Output from RiskScoreEngine.

        referral_result : dict, optional
            Output from ReferralPriorityEngine.

        lesion_findings : dict, optional
            Lesion findings from the lesion detection module.

        gradcam_available : bool
            Whether Grad-CAM visualization is available.

        Returns
        -------
        dict
            Structured report.
        """

        disease = self._normalize_disease(disease)
        confidence = self._normalize_confidence(
            classification_confidence
        )

        # ---------------------------------------------------------
        # SEVERITY
        # ---------------------------------------------------------

        severity = self._extract_severity(
            severity_result
        )

        # ---------------------------------------------------------
        # RISK
        # ---------------------------------------------------------

        risk = self._extract_risk(
            risk_result
        )

        # ---------------------------------------------------------
        # REFERRAL
        # ---------------------------------------------------------

        referral = self._extract_referral(
            referral_result
        )

        # ---------------------------------------------------------
        # LESIONS
        # ---------------------------------------------------------

        lesions = self._extract_lesions(
            lesion_findings
        )

        # ---------------------------------------------------------
        # REPORT
        # ---------------------------------------------------------

        report = {
            "report_title": "EyeX Automated Screening Report",

            "disease_classification": {
                "prediction": disease,
                "confidence": confidence,
            },

            "image_quality": {
                "quality": image_quality,
            },

            "disease_severity": severity,

            "lesion_findings": lesions,

            "risk_assessment": risk,

            "referral_priority": referral,

            "explainability": {
                "gradcam_available": bool(
                    gradcam_available
                ),
                "description": (
                    "Grad-CAM visualization is available to "
                    "show image regions receiving attention "
                    "from the classification model."
                    if gradcam_available
                    else
                    "Grad-CAM visualization is not available "
                    "for this analysis."
                ),
            },

            "disclaimer": self.DISCLAIMER,
        }

        return report

    # =========================================================
    # TEXT REPORT
    # =========================================================

    def generate_text(
        self,
        report
    ):
        """
        Convert structured report into a readable text report.
        """

        classification = report.get(
            "disease_classification",
            {}
        )

        quality = report.get(
            "image_quality",
            {}
        )

        severity = report.get(
            "disease_severity",
            {}
        )

        lesions = report.get(
            "lesion_findings",
            {}
        )

        risk = report.get(
            "risk_assessment",
            {}
        )

        referral = report.get(
            "referral_priority",
            {}
        )

        explainability = report.get(
            "explainability",
            {}
        )

        lines = []

        lines.append("=" * 60)
        lines.append(
            "                 EYEX REPORT"
        )
        lines.append("=" * 60)

        # ---------------------------------------------------------
        # CLASSIFICATION
        # ---------------------------------------------------------

        lines.append("")
        lines.append(
            "DISEASE CLASSIFICATION"
        )
        lines.append("-" * 60)

        lines.append(
            f"Prediction: "
            f"{classification.get('prediction', 'Unavailable')}"
        )

        confidence = classification.get(
            "confidence"
        )

        if confidence is not None:
            lines.append(
                f"Confidence: {confidence:.2f}%"
            )
        else:
            lines.append(
                "Confidence: Unavailable"
            )

        # ---------------------------------------------------------
        # IMAGE QUALITY
        # ---------------------------------------------------------

        lines.append("")
        lines.append(
            "IMAGE QUALITY"
        )
        lines.append("-" * 60)

        lines.append(
            f"Quality: "
            f"{quality.get('quality', 'Unavailable')}"
        )

        # ---------------------------------------------------------
        # SEVERITY
        # ---------------------------------------------------------

        lines.append("")
        lines.append(
            "DISEASE SEVERITY"
        )
        lines.append("-" * 60)

        lines.append(
            f"Assessment: "
            f"{severity.get('severity', 'Unavailable')}"
        )

        if severity.get("basis"):
            lines.append(
                f"Basis: {severity['basis']}"
            )

        if severity.get("rationale"):
            lines.append(
                f"Rationale: {severity['rationale']}"
            )

        # ---------------------------------------------------------
        # LESIONS
        # ---------------------------------------------------------

        if lesions:
            lines.append("")
            lines.append(
                "LESION FINDINGS"
            )
            lines.append("-" * 60)

            lesion_labels = {
                "microaneurysms": "Microaneurysms",
                "haemorrhages": "Haemorrhages",
                "hard_exudates": "Hard Exudates",
                "soft_exudates": "Soft Exudates",
                "total": "Total Lesions",
            }

            for key, label in lesion_labels.items():

                if key in lesions:
                    lines.append(
                        f"{label}: {lesions[key]}"
                    )

        # ---------------------------------------------------------
        # RISK
        # ---------------------------------------------------------

        lines.append("")
        lines.append(
            "RISK ASSESSMENT"
        )
        lines.append("-" * 60)

        if risk.get("risk_score") is not None:

            lines.append(
                f"Risk Score: "
                f"{risk['risk_score']} / 100"
            )

            lines.append(
                f"Category: "
                f"{risk.get('risk_category', 'Unavailable')}"
            )

        else:
            lines.append(
                "Risk Score: Unavailable"
            )

        # ---------------------------------------------------------
        # REFERRAL
        # ---------------------------------------------------------

        lines.append("")
        lines.append(
            "REFERRAL / PRIORITY"
        )
        lines.append("-" * 60)

        lines.append(
            f"Priority: "
            f"{referral.get('priority', 'Unavailable')}"
        )

        lines.append(
            f"Recommended Action: "
            f"{referral.get('recommended_action', 'Unavailable')}"
        )

        if referral.get("reason"):
            lines.append(
                f"Reason: {referral['reason']}"
            )

        # ---------------------------------------------------------
        # EXPLAINABILITY
        # ---------------------------------------------------------

        lines.append("")
        lines.append(
            "EXPLAINABILITY"
        )
        lines.append("-" * 60)

        if explainability.get(
            "gradcam_available",
            False
        ):
            lines.append(
                "Grad-CAM: Available"
            )
        else:
            lines.append(
                "Grad-CAM: Not available"
            )

        # ---------------------------------------------------------
        # DISCLAIMER
        # ---------------------------------------------------------

        lines.append("")
        lines.append(
            "DISCLAIMER"
        )
        lines.append("-" * 60)

        lines.append(
            report.get(
                "disclaimer",
                self.DISCLAIMER
            )
        )

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    # =========================================================
    # EXTRACTION HELPERS
    # =========================================================

    @staticmethod
    def _extract_severity(
        severity_result
    ):

        if not severity_result:
            return {
                "severity": "Unavailable",
                "severity_level": None,
                "basis": None,
                "rationale": None,
            }

        return {
            "severity": severity_result.get(
                "severity",
                "Unavailable"
            ),
            "severity_level": severity_result.get(
                "severity_level"
            ),
            "basis": severity_result.get(
                "basis"
            ),
            "rationale": severity_result.get(
                "rationale"
            ),
            "limitations": severity_result.get(
                "limitations"
            ),
        }

    @staticmethod
    def _extract_risk(
        risk_result
    ):

        if not risk_result:
            return {
                "risk_score": None,
                "risk_category": "Unavailable",
                "priority": "Unavailable",
            }

        return {
            "risk_score": risk_result.get(
                "risk_score"
            ),
            "risk_category": risk_result.get(
                "risk_category",
                "Unavailable"
            ),
            "priority": risk_result.get(
                "priority",
                "Unavailable"
            ),
            "contributing_factors": risk_result.get(
                "contributing_factors",
                []
            ),
        }

    @staticmethod
    def _extract_referral(
        referral_result
    ):

        if not referral_result:
            return {
                "priority": "Unavailable",
                "priority_level": None,
                "recommended_action": "Unavailable",
                "reason": None,
            }

        return {
            "priority": referral_result.get(
                "priority",
                "Unavailable"
            ),
            "priority_level": referral_result.get(
                "priority_level"
            ),
            "recommended_action": referral_result.get(
                "recommended_action",
                "Unavailable"
            ),
            "reason": referral_result.get(
                "reason"
            ),
        }

    @staticmethod
    def _extract_lesions(
        findings
    ):

        if not findings:
            return {}

        allowed_keys = {
            "microaneurysms",
            "haemorrhages",
            "hard_exudates",
            "soft_exudates",
            "total",
        }

        return {
            key: findings[key]
            for key in allowed_keys
            if key in findings
        }

    # =========================================================
    # NORMALIZATION
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


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":

    engine = ClinicalReportGenerator()

    print("\n" + "=" * 60)
    print("EyeX Automated Clinical Report Engine")
    print("=" * 60)

    # ---------------------------------------------------------
    # SAMPLE INPUTS
    # ---------------------------------------------------------

    severity_result = {
        "disease": "DR",
        "severity": "Moderate DR pattern",
        "severity_level": 2,
        "basis": "Lesion analysis",
        "rationale": (
            "Multiple lesion types were detected."
        ),
        "limitations": (
            "Project-level screening estimate."
        ),
    }

    risk_result = {
        "disease": "DR",
        "risk_score": 81,
        "risk_category": "Very High",
        "priority": "Urgent Review",
        "contributing_factors": [
            "DR classification",
            "Moderate severity",
            "Lesion findings",
        ],
    }

    referral_result = {
        "disease": "DR",
        "risk_score": 81,
        "priority": "High Priority Review",
        "priority_level": 4,
        "recommended_action": (
            "Prioritize ophthalmologist review "
            "of the EyeX screening findings."
        ),
        "reason": (
            "The combined project-level screening "
            "findings indicate high review priority."
        ),
    }

    lesion_findings = {
        "microaneurysms": 12,
        "haemorrhages": 4,
        "hard_exudates": 8,
        "soft_exudates": 0,
        "total": 24,
    }

    # ---------------------------------------------------------
    # GENERATE REPORT
    # ---------------------------------------------------------

    report = engine.generate(
        disease="DR",
        classification_confidence=91.5,
        image_quality="Good",
        severity_result=severity_result,
        risk_result=risk_result,
        referral_result=referral_result,
        lesion_findings=lesion_findings,
        gradcam_available=True,
    )

    # ---------------------------------------------------------
    # PRINT STRUCTURED REPORT
    # ---------------------------------------------------------

    print("\nSTRUCTURED REPORT")
    print("-" * 60)
    print(report)

    # ---------------------------------------------------------
    # PRINT HUMAN-READABLE REPORT
    # ---------------------------------------------------------

    print("\n")
    print(engine.generate_text(report))

    print("\n" + "=" * 60)
    print("Clinical report test completed successfully.")
    print("=" * 60)