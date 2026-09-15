"""
EyeX Doctor Feedback Module

Purpose:
    Store structured ophthalmologist/doctor feedback on an
    EyeX AI-assisted screening result.

This module does NOT make a medical decision.
It records human review of the AI output.

Supported feedback:
    - AI result confirmed
    - AI result disagreed with
    - Corrected disease
    - Corrected severity
    - Doctor recommendation
    - Doctor notes

The module is designed so that feedback can later be:
    - displayed in Streamlit
    - saved to JSON/CSV/database
    - used for model monitoring
    - used to identify potential model errors
"""


class DoctorFeedbackManager:
    """
    Manages structured doctor feedback for EyeX.
    """

    SUPPORTED_DISEASES = {
        "Healthy",
        "DR",
        "Glaucoma",
        "AMD",
    }

    SUPPORTED_SEVERITIES = {
        "None detected",
        "Mild / early DR pattern",
        "Moderate DR pattern",
        "Advanced DR pattern",
        "Requires lesion assessment",
        "Requires structural assessment",
        "Requires macular assessment",
        "Lower structural risk pattern",
        "Moderate structural risk pattern",
        "Higher structural risk pattern",
        "Macular findings available",
        "Unknown",
    }

    SUPPORTED_RECOMMENDATIONS = {
        "Agree with AI assessment",
        "Agree with AI assessment with monitoring",
        "Needs further examination",
        "Refer to ophthalmologist",
        "AI assessment requires correction",
        "Other",
    }

    # =========================================================
    # CREATE FEEDBACK
    # =========================================================

    def create_feedback(
        self,
        ai_disease,
        ai_severity=None,
        ai_risk_score=None,
        ai_priority=None,
        doctor_agrees=None,
        corrected_disease=None,
        corrected_severity=None,
        recommendation=None,
        doctor_notes="",
    ):
        """
        Create a structured doctor feedback record.

        Parameters
        ----------
        ai_disease : str
            Disease predicted by EyeX.

        ai_severity : str, optional
            Severity produced by EyeX.

        ai_risk_score : float, optional
            EyeX project-level risk score.

        ai_priority : str, optional
            EyeX referral priority.

        doctor_agrees : bool, optional
            Whether the doctor agrees with the AI assessment.

        corrected_disease : str, optional
            Doctor-corrected disease if AI result is disagreed with.

        corrected_severity : str, optional
            Doctor-corrected severity.

        recommendation : str, optional
            Doctor's recommended action.

        doctor_notes : str
            Free-text doctor notes.

        Returns
        -------
        dict
            Structured feedback record.
        """

        ai_disease = self._normalize_disease(
            ai_disease
        )

        corrected_disease = self._normalize_disease(
            corrected_disease
        ) if corrected_disease is not None else None

        ai_risk_score = self._normalize_score(
            ai_risk_score
        )

        # ---------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------

        validation_errors = []

        if ai_disease not in self.SUPPORTED_DISEASES:
            validation_errors.append(
                "Unsupported AI disease."
            )

        if (
            corrected_disease is not None
            and corrected_disease not in self.SUPPORTED_DISEASES
        ):
            validation_errors.append(
                "Unsupported corrected disease."
            )

        if (
            recommendation is not None
            and recommendation not in self.SUPPORTED_RECOMMENDATIONS
        ):
            validation_errors.append(
                "Unsupported recommendation."
            )

        # ---------------------------------------------------------
        # CORRECTION STATUS
        # ---------------------------------------------------------

        if doctor_agrees is True:
            assessment_status = "Confirmed by doctor"

        elif doctor_agrees is False:
            assessment_status = "Disagreed by doctor"

        else:
            assessment_status = "Doctor review recorded"

        # ---------------------------------------------------------
        # CORRECTION DETAILS
        # ---------------------------------------------------------

        correction_made = (
            doctor_agrees is False
            and (
                corrected_disease is not None
                or corrected_severity is not None
            )
        )

        if correction_made:
            correction_status = "Correction provided"
        elif doctor_agrees is False:
            correction_status = "Disagreement recorded"
        else:
            correction_status = "No correction"

        # ---------------------------------------------------------
        # RESULT
        # ---------------------------------------------------------

        return {
            "ai_assessment": {
                "disease": ai_disease,
                "severity": ai_severity,
                "risk_score": ai_risk_score,
                "priority": ai_priority,
            },

            "doctor_review": {
                "agrees_with_ai": doctor_agrees,
                "assessment_status": assessment_status,
                "correction_status": correction_status,
                "corrected_disease": corrected_disease,
                "corrected_severity": corrected_severity,
                "recommendation": recommendation,
                "notes": self._clean_notes(
                    doctor_notes
                ),
            },

            "validation": {
                "valid": len(validation_errors) == 0,
                "errors": validation_errors,
            },

            "purpose": (
                "Human-in-the-loop review and model monitoring."
            ),

            "limitations": (
                "Doctor feedback recorded by EyeX is intended "
                "for review and model-monitoring purposes. "
                "It does not itself constitute a medical record "
                "unless appropriately reviewed, verified, and "
                "stored within a compliant clinical system."
            ),
        }

    # =========================================================
    # FEEDBACK SUMMARY
    # =========================================================

    def summarize_feedback(
        self,
        feedback
    ):
        """
        Produce a concise human-readable summary.
        """

        if not feedback:
            return "No doctor feedback available."

        ai = feedback.get(
            "ai_assessment",
            {}
        )

        doctor = feedback.get(
            "doctor_review",
            {}
        )

        lines = []

        lines.append(
            "EYEX DOCTOR FEEDBACK"
        )
        lines.append(
            "-" * 50
        )

        lines.append(
            f"AI Disease: "
            f"{ai.get('disease', 'Unavailable')}"
        )

        lines.append(
            f"AI Severity: "
            f"{ai.get('severity', 'Unavailable')}"
        )

        lines.append(
            f"AI Risk Score: "
            f"{ai.get('risk_score', 'Unavailable')}"
        )

        lines.append(
            f"AI Priority: "
            f"{ai.get('priority', 'Unavailable')}"
        )

        lines.append("")

        lines.append(
            f"Doctor Agreement: "
            f"{doctor.get('agrees_with_ai', 'Not specified')}"
        )

        lines.append(
            f"Status: "
            f"{doctor.get('assessment_status', 'Unavailable')}"
        )

        lines.append(
            f"Correction: "
            f"{doctor.get('correction_status', 'Unavailable')}"
        )

        if doctor.get(
            "corrected_disease"
        ):
            lines.append(
                f"Corrected Disease: "
                f"{doctor['corrected_disease']}"
            )

        if doctor.get(
            "corrected_severity"
        ):
            lines.append(
                f"Corrected Severity: "
                f"{doctor['corrected_severity']}"
            )

        if doctor.get(
            "recommendation"
        ):
            lines.append(
                f"Recommendation: "
                f"{doctor['recommendation']}"
            )

        if doctor.get(
            "notes"
        ):
            lines.append("")
            lines.append(
                "Doctor Notes:"
            )
            lines.append(
                doctor["notes"]
            )

        return "\n".join(lines)

    # =========================================================
    # MONITORING INFORMATION
    # =========================================================

    def get_monitoring_signal(
        self,
        feedback
    ):
        """
        Identify whether the feedback represents a potential
        model disagreement.

        This can later feed a model monitoring dashboard.
        """

        if not feedback:
            return {
                "model_disagreement": False,
                "signal": "No feedback",
            }

        doctor = feedback.get(
            "doctor_review",
            {}
        )

        agrees = doctor.get(
            "agrees_with_ai"
        )

        if agrees is False:
            return {
                "model_disagreement": True,
                "signal": "Review required",
                "reason": (
                    "Doctor disagreed with the AI assessment."
                ),
            }

        if agrees is True:
            return {
                "model_disagreement": False,
                "signal": "Confirmed",
                "reason": (
                    "Doctor agreed with the AI assessment."
                ),
            }

        return {
            "model_disagreement": False,
            "signal": "Pending interpretation",
            "reason": (
                "Doctor agreement was not explicitly recorded."
            ),
        }

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _normalize_disease(
        disease
    ):

        if disease is None:
            return None

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
    def _normalize_score(
        score
    ):

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
    def _clean_notes(
        notes
    ):

        if notes is None:
            return ""

        return str(
            notes
        ).strip()


# =============================================================
# TESTS
# =============================================================

if __name__ == "__main__":

    manager = DoctorFeedbackManager()

    print("\n" + "=" * 60)
    print("EyeX Doctor Feedback Module")
    print("=" * 60)

    # ---------------------------------------------------------
    # TEST 1 — DOCTOR AGREES
    # ---------------------------------------------------------

    feedback_1 = manager.create_feedback(
        ai_disease="DR",
        ai_severity="Moderate DR pattern",
        ai_risk_score=81,
        ai_priority="High Priority Review",
        doctor_agrees=True,
        recommendation="Refer to ophthalmologist",
        doctor_notes=(
            "AI findings reviewed. "
            "Lesion findings appear consistent with the screening result."
        ),
    )

    print("\nTEST 1 — DOCTOR AGREES")
    print(feedback_1)

    print("\nSUMMARY")
    print(manager.summarize_feedback(feedback_1))

    print("\nMONITORING")
    print(manager.get_monitoring_signal(feedback_1))

    # ---------------------------------------------------------
    # TEST 2 — DOCTOR DISAGREES
    # ---------------------------------------------------------

    feedback_2 = manager.create_feedback(
        ai_disease="DR",
        ai_severity="Moderate DR pattern",
        ai_risk_score=81,
        ai_priority="High Priority Review",
        doctor_agrees=False,
        corrected_disease="Healthy",
        corrected_severity="None detected",
        recommendation="Needs further examination",
        doctor_notes=(
            "The automated result does not appear "
            "to match the clinical assessment."
        ),
    )

    print("\nTEST 2 — DOCTOR DISAGREES")
    print(feedback_2)

    print("\nSUMMARY")
    print(manager.summarize_feedback(feedback_2))

    print("\nMONITORING")
    print(manager.get_monitoring_signal(feedback_2))

    # ---------------------------------------------------------
    # TEST 3 — PENDING REVIEW
    # ---------------------------------------------------------

    feedback_3 = manager.create_feedback(
        ai_disease="Glaucoma",
        ai_severity="Requires structural assessment",
        ai_risk_score=37,
        ai_priority="Review",
        doctor_agrees=None,
        recommendation="Needs further examination",
        doctor_notes="Awaiting complete ophthalmic examination.",
    )

    print("\nTEST 3 — PENDING REVIEW")
    print(feedback_3)

    print("\nSUMMARY")
    print(manager.summarize_feedback(feedback_3))

    print("\nMONITORING")
    print(manager.get_monitoring_signal(feedback_3))

    # ---------------------------------------------------------
    # TEST 4 — INVALID INPUT
    # ---------------------------------------------------------

    feedback_4 = manager.create_feedback(
        ai_disease="UnknownDisease",
        ai_risk_score=50,
        doctor_agrees=False,
        corrected_disease="AMD",
        recommendation="Other",
        doctor_notes="Testing validation.",
    )

    print("\nTEST 4 — VALIDATION")
    print(feedback_4)

    print("\n" + "=" * 60)
    print("All doctor-feedback tests completed.")
    print("=" * 60)