from statistics import mean, pstdev

from app.schemas.analysis import (
    FinalDecision,
    RawEvidence,
    Verdict,
)


class ForensicService:
    """
    Converts raw ML evidence into a defensible forensic decision.

    The ML layer answers:
        "What anomalies do we see?"

    This layer answers:
        "How much should we trust those anomalies?"
    """

    # ---------------------------------------------------------
    # Initial weights
    # ---------------------------------------------------------
    #
    # These are STARTING weights.
    # We will tune them later using validation data.
    #
    DEFAULT_WEIGHTS = {
        "visual": 0.25,
        "frequency": 0.15,
        "boundary": 0.20,
        "temporal": 0.25,
        "identity": 0.15,
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        deepfake_threshold: float = 0.55,
        review_threshold: float = 0.38,
    ) -> None:

        self.weights = (
            weights.copy()
            if weights is not None
            else self.DEFAULT_WEIGHTS.copy()
        )

        self.deepfake_threshold = deepfake_threshold
        self.review_threshold = review_threshold

    # =========================================================
    # MAIN ANALYSIS
    # =========================================================

    def analyze(
        self,
        evidence: RawEvidence,
    ) -> FinalDecision:
        """
        Convert raw detector evidence into a final decision.
        """

        # -----------------------------------------------------
        # 1. Weighted manipulation score
        # -----------------------------------------------------

        manipulation_score = self._weighted_score(
            evidence
        )

        # -----------------------------------------------------
        # 2. Consistency across frames
        # -----------------------------------------------------

        consistency = self._consistency(
            evidence.frame_scores
        )

        # -----------------------------------------------------
        # 3. Agreement between independent detectors
        # -----------------------------------------------------

        agreement = self._agreement(
            evidence
        )

        # -----------------------------------------------------
        # 4. Evidence reliability
        # -----------------------------------------------------

        reliability = self._reliability(
            evidence=evidence,
            agreement=agreement,
            consistency=consistency,
        )

        # -----------------------------------------------------
        # 5. Final confidence
        # -----------------------------------------------------
        #
        # This is deliberately transparent.
        # It is NOT being claimed as a calibrated probability yet.
        #
        # We will validate/calibrate this later.
        # -----------------------------------------------------

        final_confidence = self._final_confidence(
            manipulation_score=manipulation_score,
            consistency=consistency,
            reliability=reliability,
        )

        # -----------------------------------------------------
        # 6. Final verdict
        # -----------------------------------------------------

        verdict = self._verdict(
            final_confidence=final_confidence,
            reliability=reliability,
            agreement=agreement,
            manipulation_score=manipulation_score,
        )

        # The ML pipeline already makes the calibrated three-way classification
        # from its quality-weighted frame scores.  Keep this legacy evidence
        # summary from publishing a contradictory verdict based on proxy
        # signals (for example, face-tracking stability).
        pipeline_result = evidence.pipeline_forensic_result or {}
        pipeline_verdict = pipeline_result.get("verdict")
        if pipeline_verdict in {item.value for item in Verdict}:
            verdict = Verdict(pipeline_verdict)

        # -----------------------------------------------------
        # 7. Human-readable explanation
        # -----------------------------------------------------

        reasons = self._reasons(
            evidence
        )

        # -----------------------------------------------------
        # 8. Additional metrics
        # -----------------------------------------------------

        frame_mean = self._frame_mean(
            evidence.frame_scores
        )

        frame_std = self._frame_std(
            evidence.frame_scores
        )

        return FinalDecision(
            verdict=verdict,

            manipulation_score=round(
                manipulation_score,
                4,
            ),

            evidence_reliability=round(
                reliability,
                4,
            ),

            evidence_consistency=round(
                consistency,
                4,
            ),

            detector_agreement=round(
                agreement,
                4,
            ),

            reasons=reasons,

            suspicious_segments=(
                evidence.suspicious_segments
            ),

            metrics={
                "final_confidence": round(
                    final_confidence,
                    4,
                ),

                "frame_score_mean": round(
                    frame_mean,
                    4,
                ),

                "frame_score_std": round(
                    frame_std,
                    4,
                ),

                "weights": self.weights,
            },
        )

    # =========================================================
    # 1. WEIGHTED SCORE
    # =========================================================

    def _weighted_score(
        self,
        evidence: RawEvidence,
    ) -> float:
        """
        Combine all detector scores using the configured weights.
        """

        values = {
            "visual": evidence.visual_score,
            "frequency": evidence.frequency_score,
            "boundary": evidence.boundary_score,
            "temporal": evidence.temporal_score,
            "identity": evidence.identity_score,
        }

        total_weight = sum(
            self.weights.values()
        )

        if total_weight <= 0:
            raise ValueError(
                "Evidence weights must sum to a positive value."
            )

        weighted_sum = sum(
            values[name] * self.weights[name]
            for name in values
        )

        return weighted_sum / total_weight

    # =========================================================
    # 2. FRAME SCORE HELPERS
    # =========================================================

    @staticmethod
    def _extract_frame_scores(
        frame_scores: list[dict],
    ) -> list[float]:
        """
        Extract valid numerical scores from frame-level results.
        """

        scores: list[float] = []

        for item in frame_scores:

            if not isinstance(item, dict):
                continue

            if "score" not in item:
                continue

            try:
                score = float(
                    item["score"]
                )
            except (TypeError, ValueError):
                continue

            if 0.0 <= score <= 1.0:
                scores.append(score)

        return scores

    def _frame_mean(
        self,
        frame_scores: list[dict],
    ) -> float:
        """
        Average frame-level manipulation score.
        """

        scores = self._extract_frame_scores(
            frame_scores
        )

        if not scores:
            return 0.0

        return mean(scores)

    def _frame_std(
        self,
        frame_scores: list[dict],
    ) -> float:
        """
        Population standard deviation of frame-level scores.

        Low standard deviation:
            evidence is consistent.

        High standard deviation:
            evidence is unstable.
        """

        scores = self._extract_frame_scores(
            frame_scores
        )

        if len(scores) < 2:
            return 1.0

        return pstdev(scores)

    # =========================================================
    # 3. CONSISTENCY
    # =========================================================

    def _consistency(
        self,
        frame_scores: list[dict],
    ) -> float:
        """
        Convert frame-score standard deviation into
        a normalized consistency score.

        Initial mapping:

            std = 0.00 -> consistency 1.00
            std = 0.50 -> consistency 0.00

        This threshold is intentionally configurable later.
        """

        scores = self._extract_frame_scores(
            frame_scores
        )

        # Not enough frames to establish consistency.
        if len(scores) < 2:
            return 0.40

        standard_deviation = pstdev(
            scores
        )

        consistency = 1.0 - (
            standard_deviation / 0.50
        )

        return max(
            0.0,
            min(
                1.0,
                consistency,
            ),
        )

    # =========================================================
    # 4. DETECTOR AGREEMENT
    # =========================================================

    def _agreement(
        self,
        evidence: RawEvidence,
    ) -> float:
        """
        Measure how closely the independent evidence sources agree.

        Sources:

            Visual
            Frequency
            Boundary
            Temporal
            Identity
        """

        scores = [
            evidence.visual_score,
            evidence.frequency_score,
            evidence.boundary_score,
            evidence.temporal_score,
            evidence.identity_score,
        ]

        standard_deviation = pstdev(
            scores
        )

        agreement = 1.0 - (
            standard_deviation / 0.50
        )

        return max(
            0.0,
            min(
                1.0,
                agreement,
            ),
        )

    # =========================================================
    # 5. RELIABILITY
    # =========================================================

    def _reliability(
        self,
        evidence: RawEvidence,
        agreement: float,
        consistency: float,
    ) -> float:
        """
        Estimate how trustworthy the available evidence is.

        Inputs:

            usable frames
            face quality
            tracking quality
            video quality
            detector agreement
            temporal consistency
        """

        # -----------------------------------------------------
        # Frame support
        # -----------------------------------------------------
        #
        # 30+ usable frames receives the maximum frame-support
        # contribution for this initial MVP formulation.
        #

        frame_factor = min(
            1.0,
            evidence.usable_frames / 30.0,
        )

        # -----------------------------------------------------
        # Evidence quality
        # -----------------------------------------------------

        quality = (

            0.25
            * evidence.face_quality

            + 0.20
            * evidence.tracking_quality

            + 0.15
            * evidence.video_quality

            + 0.20
            * agreement

            + 0.20
            * consistency
        )

        # -----------------------------------------------------
        # Combine frame support + quality
        # -----------------------------------------------------

        reliability = (

            0.60 * quality

            + 0.40 * frame_factor
        )

        return max(
            0.0,
            min(
                1.0,
                reliability,
            ),
        )

    # =========================================================
    # 6. FINAL CONFIDENCE
    # =========================================================

    @staticmethod
    def _final_confidence(
        manipulation_score: float,
        consistency: float,
        reliability: float,
    ) -> float:
        """
        Combine manipulation score, consistency and reliability.

        IMPORTANT:
        This is an initial transparent decision formula.
        It is NOT yet a statistically calibrated probability.
        """

        confidence = (

            manipulation_score

            * (
                0.60
                + 0.40 * consistency
            )

            * (
                0.60
                + 0.40 * reliability
            )
        )

        return max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

    # =========================================================
    # 7. FINAL VERDICT
    # =========================================================

    def _verdict(
        self,
        final_confidence: float,
        reliability: float,
        agreement: float,
        manipulation_score: float = 0.5,
    ) -> Verdict:
        """
        Convert continuous evidence into a final classification.
        """

        # High confidence or elevated manipulation signal with reasonable reliability
        if (
            (final_confidence >= self.deepfake_threshold or manipulation_score >= 0.58)
            and reliability >= 0.40
        ):
            return Verdict.LIKELY_DEEPFAKE

        # Low confidence or low manipulation signal with baseline reliability
        if (
            (final_confidence <= self.review_threshold or manipulation_score <= 0.38)
            and reliability >= 0.35
        ):
            return Verdict.REAL

        # Everything else requires review.
        return Verdict.INCONCLUSIVE

    # =========================================================
    # 8. HUMAN-READABLE REASONS
    # =========================================================

    def _reasons(
        self,
        evidence: RawEvidence,
    ) -> list[str]:
        """
        Generate the strongest evidence explanations.
        """

        candidates = [

            (
                "Visual facial anomaly",
                evidence.visual_score,
            ),

            (
                "Fine-detail / frequency anomaly",
                evidence.frequency_score,
            ),

            (
                "Face-boundary anomaly",
                evidence.boundary_score,
            ),

            (
                "Temporal facial inconsistency",
                evidence.temporal_score,
            ),

            (
                "Identity instability",
                evidence.identity_score,
            ),
        ]

        # Strongest evidence first.
        candidates.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        # Only report meaningful evidence.
        return [
            name
            for name, score in candidates
            if score >= 0.70
        ][:4]
