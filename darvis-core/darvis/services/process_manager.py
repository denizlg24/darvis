import asyncio
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import httpx


class ServiceStatus(Enum):
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    FAILED = auto()
    STOPPING = auto()


@dataclass
class ServiceConfig:
    name: str
    directory: str
    port: int
    health_endpoint: str = "/health"
    startup_timeout: float = 120.0
    health_check_interval: float = 1.0


@dataclass
class ServiceProcess:
    config: ServiceConfig
    process: Optional[subprocess.Popen] = None
    status: ServiceStatus = ServiceStatus.STOPPED
    error: Optional[str] = None


DEFAULT_SERVICES = [
    ServiceConfig(
        name="transcription",
        directory="darvis-transcription",
        port=int(os.environ.get("TRANSCRIPTION_PORT", "8001")),
    ),
    ServiceConfig(
        name="wakeword",
        directory="darvis-wakeword",
        port=int(os.environ.get("WAKEWORD_PORT", "8002")),
    ),
    ServiceConfig(
        name="tts",
        directory="darvis-tts",
        port=int(os.environ.get("TTS_PORT", "8003")),
    ),
]


class ProcessManager:

    def __init__(
        self,
        services: Optional[list[ServiceConfig]] = None,
        base_path: Optional[Path] = None
    ):
        self._services = services or DEFAULT_SERVICES
        self._base_path = base_path or self._find_base_path()
        self._processes: dict[str, ServiceProcess] = {}
        self._running = False

    def _find_base_path(self) -> Path:
        
        
        current = Path.cwd()

        
        if current.name == "darvis-core":
            return current.parent

        
        for parent in [current] + list(current.parents):
            if (parent / "darvis-transcription").exists():
                return parent

        
        return current.parent

    def _get_uv_executable(self) -> str:
        
        if sys.platform == "win32":
            return "uv"
        return "uv"

    def _get_python_executable(self) -> str:
        
        return sys.executable

    async def start_all(self) -> bool:
        
        print("[SERVICES] Starting microservices...")
        self._running = True

        all_started = True
        for config in self._services:
            success = await self._start_service(config)
            if not success:
                print(f"[SERVICES] Failed to start {config.name}")
                all_started = False

        if all_started:
            print("[SERVICES] All microservices started successfully")
        else:
            print("[SERVICES] Some services failed to start")

        return all_started

    async def _start_service(self, config: ServiceConfig) -> bool:
        
        service_dir = self._base_path / config.directory

        if not service_dir.exists():
            print(f"[{config.name.upper()}] Service directory not found: {service_dir}")
            self._processes[config.name] = ServiceProcess(
                config=config,
                status=ServiceStatus.FAILED,
                error=f"Directory not found: {service_dir}"
            )
            return False

        print(f"[{config.name.upper()}] Starting on port {config.port}...")

        try:
            
            env = os.environ.copy()

            
            env_var_name = f"{config.name.upper()}_PORT"
            env[env_var_name] = str(config.port)

            
            if sys.platform == "win32":
                
                cmd = ["uv", "run", "python", "main.py"]
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
                process = subprocess.Popen(
                    cmd,
                    cwd=str(service_dir),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags
                )
            else:
                
                cmd = ["uv", "run", "python", "main.py"]
                process = subprocess.Popen(
                    cmd,
                    cwd=str(service_dir),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=True
                )

            service_process = ServiceProcess(
                config=config,
                process=process,
                status=ServiceStatus.STARTING
            )
            self._processes[config.name] = service_process

            
            healthy = await self._wait_for_health(config)

            if healthy:
                service_process.status = ServiceStatus.RUNNING
                print(f"[{config.name.upper()}] Started successfully")
                return True
            else:
                service_process.status = ServiceStatus.FAILED
                service_process.error = "Health check timed out"
                print(f"[{config.name.upper()}] Health check failed")
                return False

        except FileNotFoundError as e:
            print(f"[{config.name.upper()}] Failed to start: uv not found. Please install uv.")
            self._processes[config.name] = ServiceProcess(
                config=config,
                status=ServiceStatus.FAILED,
                error=str(e)
            )
            return False
        except Exception as e:
            print(f"[{config.name.upper()}] Failed to start: {e}")
            self._processes[config.name] = ServiceProcess(
                config=config,
                status=ServiceStatus.FAILED,
                error=str(e)
            )
            return False

    async def _wait_for_health(self, config: ServiceConfig) -> bool:
        
        url = f"http://127.0.0.1:{config.port}{config.health_endpoint}"
        start_time = time.monotonic()

        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.monotonic() - start_time < config.startup_timeout:
                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        return True
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass

                await asyncio.sleep(config.health_check_interval)

        return False

    def stop_all(self) -> None:
        
        print("[SERVICES] Stopping microservices...")
        self._running = False

        
        for name, service_process in self._processes.items():
            self._stop_service(service_process)

        
        if sys.platform == "win32":
            self._cleanup_ports()

        print("[SERVICES] All microservices stopped")

    def _cleanup_ports(self) -> None:
        
        for config in self._services:
            self._kill_process_on_port_windows(config.port)

    def _stop_service(self, service_process: ServiceProcess) -> None:
        
        if service_process.process is None:
            return

        if service_process.process.poll() is not None:
            
            service_process.status = ServiceStatus.STOPPED
            return

        name = service_process.config.name
        port = service_process.config.port
        print(f"[{name.upper()}] Stopping (PID: {service_process.process.pid}, port: {port})...")
        service_process.status = ServiceStatus.STOPPING

        try:
            if sys.platform == "win32":
                
                result = subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(service_process.process.pid)],
                    capture_output=True,
                    text=True
                )

                
                if result.returncode != 0:
                    print(f"[{name.upper()}] taskkill by PID failed, trying port-based kill...")
                    self._kill_process_on_port_windows(port)
            else:
                
                import signal as sig_module
                try:
                    os.killpg(os.getpgid(service_process.process.pid), sig_module.SIGTERM)
                except ProcessLookupError:
                    pass

            
            try:
                service_process.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                
                print(f"[{name.upper()}] Process didn't stop gracefully, force killing...")
                try:
                    service_process.process.kill()
                    service_process.process.wait(timeout=2.0)
                except:
                    pass

                
                if sys.platform == "win32":
                    self._kill_process_on_port_windows(port)

            service_process.status = ServiceStatus.STOPPED
            print(f"[{name.upper()}] Stopped")

        except Exception as e:
            print(f"[{name.upper()}] Error stopping: {e}")
            
            try:
                service_process.process.kill()
            except:
                pass
            
            if sys.platform == "win32":
                self._kill_process_on_port_windows(port)
            service_process.status = ServiceStatus.STOPPED

    def _kill_process_on_port_windows(self, port: int) -> None:
        
        try:
            
            result = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True,
                text=True
            )

            for line in result.stdout.split('\n'):
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        if pid.isdigit():
                            print(f"[SERVICES] Killing process {pid} on port {port}")
                            subprocess.run(
                                ["taskkill", "/F", "/PID", pid],
                                capture_output=True
                            )
        except Exception as e:
            print(f"[SERVICES] Error killing process on port {port}: {e}")

    def get_status(self) -> dict[str, ServiceStatus]:
        
        return {name: sp.status for name, sp in self._processes.items()}

    def is_all_running(self) -> bool:
        
        if not self._processes:
            return False
        return all(sp.status == ServiceStatus.RUNNING for sp in self._processes.values())

    async def health_check(self) -> dict[str, bool]:
        
        results = {}

        async with httpx.AsyncClient(timeout=2.0) as client:
            for config in self._services:
                url = f"http://127.0.0.1:{config.port}{config.health_endpoint}"
                try:
                    response = await client.get(url)
                    results[config.name] = response.status_code == 200
                except:
                    results[config.name] = False

        return results
