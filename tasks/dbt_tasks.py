"""
Tasks for running dbt commands.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import subprocess
import shutil

from prefect import task, get_run_logger


@dataclass
class DbtRunResult:
    """
    Result object for dbt command execution.
    """
    is_success: bool
    return_code: int
    stdout: str
    stderr: str


@task(retries=2, retry_delay_seconds=10)
def run_dbt_command(
    command: List[str],
    project_dir: str,
    profiles_dir: Optional[str] = None,
) -> DbtRunResult:
    """
    Run a dbt command and capture output.

    Args:
        command: dbt command list (e.g., ["dbt", "build", "--select", "model"])
        project_dir: Path to dbt project directory
        profiles_dir: Optional path to dbt profiles directory

    Returns:
        DbtRunResult with status and logs
    """
    logger = get_run_logger()

    dbt_path = shutil.which(command[0])
    if dbt_path is None:
        logger.error("dbt not found on PATH. Ensure dbt is installed and available.")
        return DbtRunResult(
            is_success=False,
            return_code=127,
            stdout="",
            stderr="dbt not found on PATH",
        )

    full_command = command.copy()
    if profiles_dir:
        full_command += ["--profiles-dir", profiles_dir]

    logger.info(f"Running dbt command: {' '.join(full_command)}")
    logger.info(f"Using project dir: {project_dir}")

    result = subprocess.run(
        full_command,
        cwd=project_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        logger.info("dbt command completed successfully")
    else:
        logger.error(
            f"dbt command failed with return code {result.returncode}"
        )

    return DbtRunResult(
        is_success=result.returncode == 0,
        return_code=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
