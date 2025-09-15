from __future__ import annotations
import time
import traceback
import uuid
from fastapi.responses import JSONResponse


def error_response(exc: Exception) -> JSONResponse:
    rid = str(uuid.uuid4())
    payload = {
        "error": {
            "message": str(exc),
            "type": exc.__class__.__name__,
            "param": None,
            "code": "internal_error",
        },
        "created": int(time.time()),
        "id": rid,
    }
    # In production you would not include traceback by default
    payload["error"]["traceback"] = traceback.format_exc(limit=2)
    return JSONResponse(status_code=500, content=payload)
