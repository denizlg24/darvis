import json
import platform
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import psutil
from mcp.types import Tool

system_info_tool = Tool(
    name="system_info",
    description="Get current system information including date/time and system stats. Use this when the user asks about the current date, time, day of the week, or system information like CPU usage, memory, etc.",
    inputSchema={
        "type": "object",
        "properties": {
            "info_type": {
                "type": "string",
                "enum": ["datetime", "system", "all"],
                "description": "Type of information to retrieve: 'datetime' for current date/time, 'system' for hardware stats, 'all' for everything",
                "default": "all"
            }
        },
        "required": []
    }
)


def get_datetime_info() -> dict[str, Any]:
    try:
        tz = ZoneInfo("Europe/Lisbon")
        now = datetime.now(tz)
    except Exception:
        now = datetime.now()

    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "month": now.strftime("%B"),
        "year": now.year,
        "timezone": str(now.tzinfo) if now.tzinfo else "local",
        "formatted": now.strftime("%A, %B %d, %Y at %I:%M %p"),
        "iso": now.isoformat()
    }


def get_system_info() -> dict[str, Any]:
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()

    return {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_usage_percent": cpu_percent,
        "memory_total_gb": round(memory.total / (1024 ** 3), 2),
        "memory_available_gb": round(memory.available / (1024 ** 3), 2),
        "memory_used_percent": memory.percent
    }


async def execute_system_info(arguments: dict[str, Any]) -> str:
    info_type = arguments.get("info_type", "all")

    try:
        result = {}

        if info_type in ("datetime", "all"):
            result["datetime"] = get_datetime_info()

        if info_type in ("system", "all"):
            result["system"] = get_system_info()

        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": f"Failed to get system info: {str(e)}"})
