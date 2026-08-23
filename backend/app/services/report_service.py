import json
from pathlib import Path
from typing import Any


class ReportService:
    """
    Creates and stores SynthGuard forensic reports.

    For the MVP, the primary report format is JSON because
    it is easy for the backend and frontend to consume.

    Later we can generate:
        JSON -> HTML -> PDF
    """

    def write_json_report(
        self,
        path: Path,
        case: dict[str, Any],
        evidence: dict[str, Any],
    ) -> Path:
        """
        Create a JSON forensic report.

        Parameters
        ----------
        path:
            Destination where the report should be written.

        case:
            Information about the uploaded video/case.

        evidence:
            Raw ML evidence + final forensic decision.
        """

        report = {
            "report_version": "0.1.0",

            "case": case,

            "analysis": evidence,
        }

        # Make sure the destination directory exists.
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Write formatted JSON.
        path.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

        return path