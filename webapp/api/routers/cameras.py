"""
webapp/api/routers/cameras.py
-----------------------------
Live-cams placeholder. The WiseNet source videos are .avi (no native browser
codec), and ffmpeg isn't available to transcode, so each camera is served as an
MJPEG stream (OpenCV decode -> JPEG frames) that plays in a plain <img>. This is
the "loop 5 raw videos" placeholder; the real overlay recorder is a later slice.

The stream endpoint authenticates via a `token` query param (an <img> can't send
an Authorization header) OR a normal Bearer header.
"""

import time
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import StreamingResponse

from .. import config
from ..auth.deps import TenantCtx, get_tenant_ctx, user_from_token

router = APIRouter(prefix="/cameras", tags=["cameras"])

STREAM_FPS = 12
STREAM_WIDTH = 480
_BOUNDARY = "frame"


def _overlap_for(cam_id: int) -> Optional[list]:
    for g in config.overlap_groups():
        if cam_id in g:
            return g
    return None


@router.get("")
def list_cameras(ctx: TenantCtx = Depends(get_tenant_ctx)):
    from .. import overlays  # lazy — keeps app startup fast

    cams = []
    any_overlay = False
    for i, path in enumerate(config.DEMO_CAMERA_VIDEOS):
        has_ov = overlays.has_overlay(i)
        any_overlay = any_overlay or has_ov
        cams.append({
            "cam_id": i,
            "name": f"Camera {i}",
            "available": path.exists(),
            "stream_url": f"/cameras/{i}/stream",
            "overlap_group": _overlap_for(i),
            "overlay_available": has_ov,
        })
    return {
        "cameras": cams,
        "overlap_groups": config.overlap_groups(),
        "overlay_available": any_overlay,
    }


def _mjpeg(video_path: str, cam_id: int, overlay: bool):
    import cv2  # lazy — keep app startup fast

    cap = cv2.VideoCapture(video_path)
    delay = 1.0 / STREAM_FPS
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
                ok, frame = cap.read()
                if not ok:
                    break
            # Burn recorded detection boxes/IDs in at full resolution (so the
            # stored bbox coords line up), then resize for streaming. The frame
            # just read is at POS_FRAMES - 1.
            if overlay:
                from .. import overlays  # lazy — keeps app startup fast

                idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                frame = overlays.annotate(frame, cam_id, max(idx, 0))
            h, w = frame.shape[:2]
            if w > STREAM_WIDTH:
                frame = cv2.resize(frame, (STREAM_WIDTH, int(h * STREAM_WIDTH / w)))
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok2:
                continue
            yield (
                f"--{_BOUNDARY}\r\nContent-Type: image/jpeg\r\n\r\n".encode()
                + buf.tobytes()
                + b"\r\n"
            )
            time.sleep(delay)
    finally:
        cap.release()


@router.get("/{cam_id}/stream")
def stream_camera(
    cam_id: int,
    token: Optional[str] = Query(None, description="JWT (for <img> tags)"),
    overlay: bool = Query(False, description="Burn recorded detection boxes/IDs in"),
    authorization: Optional[str] = Header(None),
):
    raw = token
    if not raw and authorization and authorization.lower().startswith("bearer "):
        raw = authorization[7:]
    if not raw:
        raise HTTPException(status_code=401, detail="Missing token")
    user_from_token(raw)  # validates or raises 401

    if cam_id < 0 or cam_id >= len(config.DEMO_CAMERA_VIDEOS):
        raise HTTPException(status_code=404, detail="Unknown camera")
    path = config.DEMO_CAMERA_VIDEOS[cam_id]
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {path.name}")

    return StreamingResponse(
        _mjpeg(str(path), cam_id, overlay),
        media_type=f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
    )
