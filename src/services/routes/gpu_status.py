"""GPU status endpoint — queries nvidia-smi on the backend host."""

import subprocess
from fastapi import APIRouter

router = APIRouter()


@router.get("/gpu_status/")
async def gpu_status():
    """Return GPU memory usage, utilization, and process count for each GPU."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,uuid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return {"gpus": [], "error": result.stderr.strip()}

        # Count compute processes per GPU UUID
        process_counts: dict[str, int] = {}
        try:
            apps = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if apps.returncode == 0:
                for line in apps.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if parts and parts[0]:
                        process_counts[parts[0]] = process_counts.get(parts[0], 0) + 1
        except Exception:
            pass  # process counts remain 0; not fatal

        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            # Fields: index, name..., memory.used, memory.total, utilization.gpu, uuid
            # Name may contain commas — reconstruct from the middle section
            uuid_val = parts[-1]
            name = ", ".join(parts[1:-4])
            gpu: dict = {
                "index": int(parts[0]),
                "name": name,
                "num_processes": process_counts.get(uuid_val, 0),
            }
            error_fields = []
            try:
                gpu["memory_used_mb"] = int(parts[-4])
            except ValueError:
                gpu["memory_used_mb"] = 0
                error_fields.append("memory_used")
            try:
                gpu["memory_total_mb"] = int(parts[-3])
            except ValueError:
                gpu["memory_total_mb"] = 0
                error_fields.append("memory_total")
            try:
                gpu["utilization_pct"] = int(parts[-2])
            except ValueError:
                gpu["utilization_pct"] = -1
                error_fields.append("utilization")
            if error_fields:
                gpu["error"] = f"N/A fields: {', '.join(error_fields)}"
            gpus.append(gpu)

        return {"gpus": gpus}

    except FileNotFoundError:
        return {"gpus": [], "error": "nvidia-smi not found on backend host"}
    except subprocess.TimeoutExpired:
        return {"gpus": [], "error": "nvidia-smi timed out"}
    except Exception as e:
        return {"gpus": [], "error": str(e)}
