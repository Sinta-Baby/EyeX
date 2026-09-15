"""
EyeX Disease Severity Engine

Purpose:
    Converts available EyeX findings into a transparent severity assessment.

Important:
    Classification confidence is NOT treated as disease severity.
    Severity requires disease-specific clinical/structural findings.

Current support:
    - Healthy
    - Diabetic Retinopathy (DR)
    - Glaucoma
    - Age-related Macular Degeneration (AMD)

The engine is designed so lesion detection and optic-disc analysis
can be connected later without changing the overall interface.
"""

from typing import Optional, Dict, Any


class DiseaseSeverityEngine:
    """
    Rule-based disease severity engine for EyeX.

    The current version provides:
        1. Safe provisional assessment when structural findings
           are unavailable.
        2. DR severity estimation when lesion findings are supplied.
        3. Clear rationale and limitations.
    """

    SUPPORTED_DISEASES = {
        "Healthy",
        "DR",
        "Glaucoma",
        "AMD",
    }

    def __init__(self):
        pass

    def assess(
        self,
        disease: str,
        classification_confidence: Optional[float] = None,
        lesion_findings: Optional[Dict[str, Any]] = None,
        optic_disc_findings: Optional[Dict[str, Any]] = None,
        macular_findings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Assess disease severity using available disease-specific findings.

        Parameters
        ----------
        disease:
            Predicted disease from EyeX classification model.

        classification_confidence:
            Classification model confidence in percentage or decimal form.
            This is reported separately and is NOT used as severity.

        lesion_findings:
            Findings from the lesion detection module.

        optic_disc_findings:
            Findings from optic disc/cup analysis.

        macular_findings:
            Findings from future AMD/macular analysis.

        Returns
        -------
        dict
            Structured severity result.
        """

        disease = self._normalize_disease(disease)

        if disease not in self.SUPPORTED_DISEASES:
            return self._unknown_result(disease)

        confidence = self._normalize_confidence(
            classification_confidence
        )

        # ---------------------------------------------------------
        # HEALTHY
        # ---------------------------------------------------------
        if disease == "Healthy":
            return {
                "disease": disease,
                "severity": "None detected",
                "severity_level": 0,
                "confidence": confidence,
                "basis": "Disease classification",
                "rationale": (
                    "The classification model predicted a healthy retina. "
                    "No disease-specific severity grading is required "
                    "for the current result."
                ),
                "limitations": (
                    "A healthy classification does not guarantee absence "
                    "of every ocular abnormality."
                ),
            }

        # ---------------------------------------------------------
        # DIABETIC RETINOPATHY
        # ---------------------------------------------------------
        if disease == "DR":

            if lesion_findings:
                return self._assess_dr(
                    confidence,
                    lesion_findings
                )

            return {
                "disease": disease,
                "severity": "Requires lesion assessment",
                "severity_level": None,
                "confidence": confidence,
                "basis": "Disease classification only",
                "rationale": (
                    "Diabetic retinopathy severity cannot be reliably "
                    "determined from disease classification confidence "
                    "alone. Lesion-level findings are required."
                ),
                "limitations": (
                    "Microaneurysms, haemorrhages and exudates have not "
                    "yet been incorporated into this assessment."
                ),
            }

        # ---------------------------------------------------------
        # GLAUCOMA
        # ---------------------------------------------------------
        if disease == "Glaucoma":

            if optic_disc_findings:
                return self._assess_glaucoma(
                    confidence,
                    optic_disc_findings
                )

            return {
                "disease": disease,
                "severity": "Requires structural assessment",
                "severity_level": None,
                "confidence": confidence,
                "basis": "Disease classification only",
                "rationale": (
                    "Glaucoma severity requires optic-disc and "
                    "optic-cup structural measurements."
                ),
                "limitations": (
                    "Cup-to-disc ratio and other structural findings "
                    "are not currently available."
                ),
            }

        # ---------------------------------------------------------
        # AMD
        # ---------------------------------------------------------
        if disease == "AMD":

            if macular_findings:
                return self._assess_amd(
                    confidence,
                    macular_findings
                )

            return {
                "disease": disease,
                "severity": "Requires macular assessment",
                "severity_level": None,
                "confidence": confidence,
                "basis": "Disease classification only",
                "rationale": (
                    "AMD severity requires disease-specific macular "
                    "findings rather than classification confidence alone."
                ),
                "limitations": (
                    "Macular structural/lesion findings are not "
                    "currently available."
                ),
            }

    # =============================================================
    # DIABETIC RETINOPATHY
    # =============================================================

    def _assess_dr(
        self,
        confidence: Optional[float],
        findings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Provisional DR severity assessment based on detected lesion
        counts.

        Expected lesion_findings keys:
            microaneurysms
            haemorrhages
            hard_exudates
            soft_exudates

        These rules are intentionally transparent and should be treated
        as a project-level screening severity estimate, not a clinical
        diagnosis.
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

        total_lesions = ma + he + ex + se

        # ---------------------------------------------------------
        # No detected lesions
        # ---------------------------------------------------------
        if total_lesions == 0:
            severity = "No detected DR lesions"
            level = 0
            rationale = (
                "No supported diabetic-retinopathy lesion findings "
                "were supplied by the lesion analysis."
            )

        # ---------------------------------------------------------
        # Mild
        # ---------------------------------------------------------
        elif ma > 0 and he == 0 and ex == 0 and se == 0:
            severity = "Mild / early DR pattern"
            level = 1
            rationale = (
                "Microaneurysms were detected without haemorrhages "
                "or exudative lesion findings."
            )

        # ---------------------------------------------------------
        # Moderate
        # ---------------------------------------------------------
        elif total_lesions > 0 and se == 0:
            severity = "Moderate DR pattern"
            level = 2
            rationale = (
                "Multiple lesion types or haemorrhagic/exudative "
                "findings were detected without soft-exudate findings."
            )

        # ---------------------------------------------------------
        # More advanced pattern
        # ---------------------------------------------------------
        else:
            severity = "Advanced DR pattern"
            level = 3
            rationale = (
                "Multiple lesion types including soft exudative "
                "findings were detected."
            )

        return {
            "disease": "DR",
            "severity": severity,
            "severity_level": level,
            "confidence": confidence,
            "basis": "Lesion analysis",
            "rationale": rationale,
            "lesion_counts": {
                "microaneurysms": ma,
                "haemorrhages": he,
                "hard_exudates": ex,
                "soft_exudates": se,
                "total": total_lesions,
            },
            "limitations": (
                "This is a transparent project-level screening "
                "estimate based on supplied lesion findings. "
                "It is not a clinical DR staging diagnosis."
            ),
        }

    # =============================================================
    # GLAUCOMA
    # =============================================================

    def _assess_glaucoma(
        self,
        confidence: Optional[float],
        findings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Provisional glaucoma structural assessment.

        Expected:
            cup_to_disc_ratio
        """

        cdr = findings.get("cup_to_disc_ratio")

        if cdr is None:
            return {
                "disease": "Glaucoma",
                "severity": "Requires structural assessment",
                "severity_level": None,
                "confidence": confidence,
                "basis": "Classification only",
                "rationale": (
                    "Optic-disc findings were supplied, but "
                    "cup-to-disc ratio is unavailable."
                ),
                "limitations": (
                    "Glaucoma severity requires broader clinical "
                    "assessment and cannot be established from "
                    "a single measurement."
                ),
            }

        try:
            cdr = float(cdr)
        except (TypeError, ValueError):
            cdr = None

        if cdr is None:
            severity = "Requires structural assessment"
            level = None
            rationale = "Invalid cup-to-disc ratio."
        elif cdr < 0.5:
            severity = "Lower structural risk pattern"
            level = 1
            rationale = (
                "The supplied cup-to-disc ratio is below 0.50."
            )
        elif cdr < 0.7:
            severity = "Moderate structural risk pattern"
            level = 2
            rationale = (
                "The supplied cup-to-disc ratio is between "
                "0.50 and 0.69."
            )
        else:
            severity = "Higher structural risk pattern"
            level = 3
            rationale = (
                "The supplied cup-to-disc ratio is 0.70 or greater."
            )

        return {
            "disease": "Glaucoma",
            "severity": severity,
            "severity_level": level,
            "confidence": confidence,
            "basis": "Optic-disc structural analysis",
            "rationale": rationale,
            "cup_to_disc_ratio": cdr,
            "limitations": (
                "Cup-to-disc ratio alone does not diagnose or "
                "clinically stage glaucoma. Clinical examination, "
                "IOP, visual field and other assessments may be required."
            ),
        }

    # =============================================================
    # AMD
    # =============================================================

    def _assess_amd(
        self,
        confidence: Optional[float],
        findings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        AMD severity placeholder for future macular analysis.
        """

        if not findings:
            return {
                "disease": "AMD",
                "severity": "Requires macular assessment",
                "severity_level": None,
                "confidence": confidence,
                "basis": "Classification only",
                "rationale": (
                    "Macular findings are not available."
                ),
                "limitations": (
                    "AMD severity requires disease-specific "
                    "macular assessment."
                ),
            }

        return {
            "disease": "AMD",
            "severity": "Macular findings available",
            "severity_level": None,
            "confidence": confidence,
            "basis": "Macular analysis",
            "rationale": (
                "Macular findings have been supplied, but a validated "
                "AMD severity grading model has not yet been implemented."
            ),
            "macular_findings": findings,
            "limitations": (
                "AMD severity grading is not currently validated "
                "within EyeX."
            ),
        }

    # =============================================================
    # HELPERS
    # =============================================================

    @staticmethod
    def _normalize_disease(disease: str) -> str:
        if disease is None:
            return ""

        disease = str(disease).strip()

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
        confidence: Optional[float]
    ) -> Optional[float]:

        if confidence is None:
            return None

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            return None

        # Convert decimal confidence to percentage.
        if 0.0 <= confidence <= 1.0:
            confidence *= 100.0

        return round(confidence, 2)

    @staticmethod
    def _safe_count(value: Any) -> int:
        try:
            value = int(value)
            return max(value, 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _unknown_result(disease: str) -> Dict[str, Any]:
        return {
            "disease": disease,
            "severity": "Unknown",
            "severity_level": None,
            "confidence": None,
            "basis": "Unsupported disease",
            "rationale": (
                "The supplied disease is not supported by "
                "the current EyeX severity engine."
            ),
            "limitations": (
                "Add disease-specific severity logic before "
                "using this result."
            ),
        }


# =============================================================
# SIMPLE TEST
# =============================================================

if __name__ == "__main__":

    engine = DiseaseSeverityEngine()

    print("\n" + "=" * 60)
    print("EyeX Disease Severity Engine")
    print("=" * 60)

    # Test 1: Healthy
    result = engine.assess(
        disease="Healthy",
        classification_confidence=99.2
    )

    print("\nTEST 1 — HEALTHY")
    print(result)

    # Test 2: DR without lesion findings
    result = engine.assess(
        disease="DR",
        classification_confidence=91.5
    )

    print("\nTEST 2 — DR WITHOUT LESIONS")
    print(result)

    # Test 3: DR with lesion findings
    result = engine.assess(
        disease="DR",
        classification_confidence=91.5,
        lesion_findings={
            "microaneurysms": 12,
            "haemorrhages": 4,
            "hard_exudates": 8,
            "soft_exudates": 0
        }
    )

    print("\nTEST 3 — DR WITH LESIONS")
    print(result)

    # Test 4: Glaucoma without optic-disc analysis
    result = engine.assess(
        disease="Glaucoma",
        classification_confidence=88.7
    )

    print("\nTEST 4 — GLAUCOMA")
    print(result)

    # Test 5: Glaucoma with optic-disc measurement
    result = engine.assess(
        disease="Glaucoma",
        classification_confidence=88.7,
        optic_disc_findings={
            "cup_to_disc_ratio": 0.72
        }
    )

    print("\nTEST 5 — GLAUCOMA WITH C/D RATIO")
    print(result)

    # Test 6: AMD
    result = engine.assess(
        disease="AMD",
        classification_confidence=95.4
    )

    print("\nTEST 6 — AMD")
    print(result)

    print("\n" + "=" * 60)
    print("All basic severity-engine tests completed.")
    print("=" * 60)