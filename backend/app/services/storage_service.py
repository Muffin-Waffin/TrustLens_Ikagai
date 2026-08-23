from pathlib import Path


class StorageService:
    """
    Handles file and directory storage for SynthGuard cases.

    Each case gets its own directory so that uploaded media,
    evidence and reports never get mixed between investigations.
    """

    def __init__(
        self,
        upload_dir: Path,
        processing_dir: Path,
        evidence_dir: Path,
        report_dir: Path,
    ) -> None:
        self.upload_dir = upload_dir
        self.processing_dir = processing_dir
        self.evidence_dir = evidence_dir
        self.report_dir = report_dir

    def case_upload_dir(
        self,
        case_id: str,
    ) -> Path:
        """
        Return/create the upload directory for a case.
        """

        path = self.upload_dir / case_id

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path

    def case_processing_dir(
        self,
        case_id: str,
    ) -> Path:
        """
        Return/create the processing directory for a case.

        This can later contain:
        - extracted frames
        - temporary face crops
        - intermediate model outputs
        """

        path = self.processing_dir / case_id

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path

    def case_evidence_dir(
        self,
        case_id: str,
    ) -> Path:
        """
        Return/create the evidence directory for a case.

        This will eventually contain:
        - raw evidence JSON
        - suspicious frames
        - heatmaps
        - temporal charts
        """

        path = self.evidence_dir / case_id

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path

    def case_report_dir(
        self,
        case_id: str,
    ) -> Path:
        """
        Return/create the report directory for a case.
        """

        path = self.report_dir / case_id

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path