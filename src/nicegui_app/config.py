import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env - try current directory first, then project root
if not load_dotenv():
    project_root = Path(__file__).parent.parent.parent
    load_dotenv(project_root / ".env")

API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:8000")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8050"))
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")

# Backend API key (must match the key set on the backend)
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY", "")

# APS beam status polling (set to "true" to enable; disabled by default for external users)
ENABLE_APS_STATUS = os.getenv("ENABLE_APS_STATUS", "false").lower() in ("true", "1", "t")

# LLM Chat Assistant - Argo Gateway (Argonne's internal LLM API)
ANL_USERNAME = os.getenv("ANL_USERNAME", "")
ARGO_BASE_URL = os.getenv("ARGO_BASE_URL", "https://apps.inside.anl.gov/argoapi/v1")
ARGO_MODEL = os.getenv("ARGO_MODEL", "gpt4o")
