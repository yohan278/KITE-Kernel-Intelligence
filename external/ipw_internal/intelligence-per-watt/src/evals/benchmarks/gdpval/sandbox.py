# src/gdpval/sandbox.py
"""
Standalone Docker sandbox for GDPval.
- Spins up the GDPval image
- Copies reference files into /workspace
- Exposes bash/python execution helpers
- Collects deliverables from deliverable_files/
"""
from __future__ import annotations

import io
import tarfile
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator

import docker
import requests


@dataclass
class SandboxHandle:
    container: docker.models.containers.Container
    deliverable_root: Path
    workdir: str = "/workspace"

    def run_tool(self, tool: str, command: str) -> Dict[str, str | int]:
        if tool == "bash":
            cmd = ["bash", "-lc", command]
        elif tool == "python":
            cmd = ["python", "-c", command]
        else:
            raise ValueError(f"Unsupported tool: {tool}")

        result = self.container.exec_run(cmd, workdir=self.workdir, user="appuser")
        return {
            "stdout": result.output.decode("utf-8", errors="ignore"),
            "exit_code": result.exit_code,
        }

    def collect_deliverables(self) -> Dict[str, bytes]:
        files: Dict[str, bytes] = {}
        for path in self.deliverable_root.rglob("*"):
            if path.is_file():
                relative = path.relative_to(self.deliverable_root)
                files[str(relative)] = path.read_bytes()
        return files


class DockerSandbox:
    def __init__(
        self,
        *,
        image: str = "gdpval",
        workdir: str = "/workspace",
        timeout: int = 120,
    ):
        self.client = docker.from_env()
        self.image = image
        self.workdir = workdir
        self.timeout = timeout

    @contextmanager
    def run(self, reference_files: Dict[str, str]) -> Iterator[SandboxHandle]:
        with tempfile.TemporaryDirectory() as tmp:
            host_dir = Path(tmp)
            container = self._start_container(host_dir)
            try:
                self._copy_reference_files(container, reference_files)
                yield SandboxHandle(container, host_dir, self.workdir)
            finally:
                container.kill()
                container.remove(force=True)

    # internal helpers -------------------------------------------------------

    def _start_container(self, host_dir: Path):
        return self.client.containers.run(
            self.image,
            command=["/bin/bash"],
            detach=True,
            tty=True,
            working_dir=self.workdir,
            user="appuser",
            volumes={
                str(host_dir): {
                    "bind": f"{self.workdir}/deliverable_files",
                    "mode": "rw",
                }
            },
        )

    def _copy_reference_files(self, container, files: Dict[str, str]) -> None:
        for rel_path, url in files.items():
            content = requests.get(url, timeout=self.timeout).content
            target_dir = rel_path.rsplit("/", 1)[0] if "/" in rel_path else ""
            if target_dir:
                container.exec_run(
                    ["mkdir", "-p", f"{self.workdir}/{target_dir}"],
                    user="appuser",
                )
            container.put_archive(
                f"{self.workdir}/{target_dir}" if target_dir else self.workdir,
                self._tar_bytes(rel_path.split("/")[-1], content),
            )

    def _tar_bytes(self, name: str, data: bytes) -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        buf.seek(0)
        return buf.read()