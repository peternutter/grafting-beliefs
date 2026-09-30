from __future__ import annotations

import base64
import errno
import logging
import shlex
from pathlib import Path
from typing import Any, Literal, Union, overload

import yaml
from inspect_ai.util._sandbox.environment import (
    SandboxConnection,
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
)
from inspect_ai.util._sandbox.registry import sandboxenv
from inspect_ai.util._subprocess import ExecResult

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60 * 60
DEFAULT_IDLE_TIMEOUT = 10 * 60
MAX_FILE_BYTES = 512 * 1024 * 1024
WRITE_CHUNK_BYTES = 4 * 1024 * 1024
READ_CHUNK_BYTES = 4 * 1024 * 1024


def _modal():
    import modal

    return modal


class _Service:

    def __init__(self, name: str, spec: dict[str, Any], options: dict[str, Any]) -> None:
        self.name = name
        self.image: str = spec["image"]
        self.workdir: str | None = spec.get("working_dir") or spec.get("workdir")
        cmd = spec.get("command")
        self.entrypoint: list[str] = (
            shlex.split(cmd) if isinstance(cmd, str) else list(cmd) if cmd else ["sleep", "infinity"]
        )
        self.cpu: float | None = float(spec["cpus"]) if spec.get("cpus") is not None else None
        mem = spec.get("mem_limit")
        self.memory: int | None = _parse_mem(mem) if mem else None
        env = spec.get("environment") or {}
        self.env: dict[str, str] = (
            {k: str(v) for k, v in env.items()}
            if isinstance(env, dict)
            else {p.split("=", 1)[0]: p.split("=", 1)[1] for p in env if "=" in p}
        )
        self.block_network: bool = bool(options.get("block_network", False))
        self.timeout: int = int(options.get("timeout", DEFAULT_TIMEOUT))
        self.idle_timeout: int = int(options.get("idle_timeout", DEFAULT_IDLE_TIMEOUT))
        self.gpu: str | None = options.get("gpu")
        self.region: str | None = options.get("region")


def _parse_mem(v: Any) -> int:
    s = str(v).strip().lower()
    mult = {"k": 1 / 1024, "m": 1, "g": 1024}.get(s[-1:])
    return int(float(s[:-1]) * mult) if mult else max(1, int(float(s) / (1024 * 1024)))


def _parse_config(config: SandboxEnvironmentConfigType | None) -> dict[str, _Service]:
    if config is None:
        raise ValueError(
            "modal sandbox requires a config: a compose-style yaml path, or an image name string."
        )
    if isinstance(config, str) and not (config.endswith((".yaml", ".yml")) and Path(config).exists()):
        return {"default": _Service("default", {"image": config}, {})}

    path = Path(str(config))
    if not path.exists():
        raise FileNotFoundError(f"modal sandbox config not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    options = data.get("x-inspect_modal_sandbox") or {}
    services = data.get("services") or {}
    if not services:
        raise ValueError(f"modal sandbox config {path} declares no services")
    return {name: _Service(name, spec or {}, options) for name, spec in services.items()}


@sandboxenv(name="modal")
class ModalSandboxEnvironment(SandboxEnvironment):

    def __init__(self, sandbox: Any, service: _Service) -> None:
        self._sb = sandbox
        self._service = service


    @classmethod
    def config_files(cls) -> list[str]:
        return ["compose.yaml", "compose.yml", "docker-compose.yaml", "docker-compose.yml"]

    @classmethod
    def default_concurrency(cls) -> int | None:
        return 16

    @classmethod
    async def task_init(cls, task_name: str, config: SandboxEnvironmentConfigType | None) -> None:
        _parse_config(config)

    @classmethod
    async def sample_init(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        metadata: dict[str, str],
    ) -> dict[str, SandboxEnvironment]:
        modal = _modal()
        services = _parse_config(config)
        app = await modal.App.lookup.aio(f"inspect-sandbox-{_safe(task_name)}", create_if_missing=True)

        envs: dict[str, SandboxEnvironment] = {}
        for name, svc in services.items():
            image = modal.Image.from_registry(svc.image)
            sb = await modal.Sandbox.create.aio(
                *svc.entrypoint,
                app=app,
                image=image,
                workdir=svc.workdir,
                cpu=svc.cpu,
                memory=svc.memory,
                env=svc.env or None,
                block_network=svc.block_network,
                timeout=svc.timeout,
                idle_timeout=svc.idle_timeout,
                gpu=svc.gpu,
                region=svc.region,
            )
            envs[name] = cls(sb, svc)
        return envs

    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        environments: dict[str, SandboxEnvironment],
        interrupted: bool,
    ) -> None:
        for env in environments.values():
            try:
                await env._sb.terminate.aio()
            except Exception as e:
                logger.warning(f"modal sandbox terminate failed: {e}")

    @classmethod
    async def task_cleanup(
        cls, task_name: str, config: SandboxEnvironmentConfigType | None, cleanup: bool
    ) -> None:
        return None


    async def exec(
        self,
        cmd: list[str],
        input: str | bytes | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,
        timeout_retry: bool = True,
        concurrency: bool = True,
    ) -> ExecResult[str]:
        argv = list(cmd)
        if user:
            argv = ["su", user, "-c", " ".join(shlex.quote(a) for a in argv)]

        proc = await self._sb.exec.aio(
            *argv,
            workdir=cwd or self._service.workdir,
            env=env or None,
            timeout=timeout,
            text=True,
        )
        if input is not None:
            data = input.encode() if isinstance(input, str) else input
            proc.stdin.write(data)
            await proc.stdin.drain.aio()
            proc.stdin.write_eof()
            await proc.stdin.drain.aio()
        stdout = await proc.stdout.read.aio()
        stderr = await proc.stderr.read.aio()
        returncode = await proc.wait.aio()
        return ExecResult(
            success=returncode == 0, returncode=returncode, stdout=stdout or "", stderr=stderr or ""
        )


    async def write_file(self, file: str, contents: str | bytes) -> None:
        data = contents.encode() if isinstance(contents, str) else contents
        parent = str(Path(file).parent)
        if parent not in ("", "/", "."):
            mk = await self.exec(["mkdir", "-p", parent])
            if not mk.success and "File exists" not in mk.stderr:
                if "Permission denied" in mk.stderr:
                    raise PermissionError(f"Permission denied creating {parent}: {mk.stderr}")
                raise RuntimeError(f"failed to create {parent}: {mk.stderr}")
        try:
            fh = await self._sb.open.aio(file, "wb")
        except Exception as e:
            msg = str(e)
            if "Permission denied" in msg:
                raise PermissionError(f"Permission denied writing {file}: {msg}") from e
            if "Is a directory" in msg or "directory" in msg.lower():
                raise IsADirectoryError(f"{file} is a directory") from e
            raise
        try:
            for i in range(0, max(len(data), 1), WRITE_CHUNK_BYTES):
                await fh.write.aio(data[i : i + WRITE_CHUNK_BYTES])
            await fh.flush.aio()
        finally:
            await fh.close.aio()

    @overload
    async def read_file(self, file: str, text: Literal[True] = True) -> str: ...

    @overload
    async def read_file(self, file: str, text: Literal[False]) -> bytes: ...

    async def read_file(self, file: str, text: bool = True) -> Union[str, bytes]:
        probe = await self.exec(
            [
                "sh",
                "-c",
                f"if [ -d {shlex.quote(file)} ]; then echo DIR; "
                f"elif [ ! -e {shlex.quote(file)} ]; then echo MISSING; "
                f"elif [ ! -r {shlex.quote(file)} ]; then echo DENIED; "
                f"else wc -c < {shlex.quote(file)}; fi",
            ]
        )
        token = (probe.stdout or "").strip()
        if token == "MISSING":
            raise FileNotFoundError(errno.ENOENT, "No such file or directory", file)
        if token == "DIR":
            raise IsADirectoryError(errno.EISDIR, "Is a directory", file)
        if token == "DENIED":
            raise PermissionError(errno.EACCES, "Permission denied", file)
        if token.isdigit() and int(token) > MAX_FILE_BYTES:
            raise RuntimeError(f"{file} is {token} bytes, over the {MAX_FILE_BYTES} byte read limit")

        fh = await self._sb.open.aio(file, "rb")
        try:
            chunks: list[bytes] = []
            while True:
                buf = await fh.read.aio(READ_CHUNK_BYTES)
                if not buf:
                    break
                chunks.append(buf)
                if sum(map(len, chunks)) > MAX_FILE_BYTES:
                    raise RuntimeError(f"{file} exceeds the {MAX_FILE_BYTES} byte read limit")
        finally:
            await fh.close.aio()
        raw = b"".join(chunks)
        return raw.decode(errors="replace") if text else raw

    async def connection(self, *, user: str | None = None) -> SandboxConnection:
        sid = getattr(self._sb, "object_id", "?")
        return SandboxConnection(
            type="modal",
            command=f"modal shell {sid}",
            container=str(sid),
        )


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in s)[:60].strip("-") or "task"
