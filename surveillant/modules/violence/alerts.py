"""
modules/violence/alerts.py
--------------------------
Side effects for violence detection (Part 11): JSON log, alert snapshot + clip,
and optional email. Stateless helpers — the violence worker owns per-camera
state (buffers, consecutive counts, email cooldown) and calls these.

SECURITY: email credentials are read from config (which loads them from the
environment / .env). NOTHING is hardcoded here. If VIOLENCE_ALERT_SENDER is
empty, email is disabled and only the JSON log + clip are produced. (The team's
old copy hardcoded a live Gmail app password — never do that; revoke it.)
"""

import json
import smtplib
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    VIOLENCE_ALERT_SENDER,
    VIOLENCE_ALERT_PASSWORD,
    VIOLENCE_ALERT_RECEIVER,
    VIOLENCE_SMTP_HOST,
    VIOLENCE_SMTP_PORT,
)


def append_violence_log(log_path: Path, entry: dict, keep: int = 200) -> None:
    """Append an event to violence_log.json, keeping the most recent `keep`."""
    try:
        existing = []
        if Path(log_path).exists():
            try:
                existing = json.loads(Path(log_path).read_text())
            except Exception:
                existing = []
        existing.append(entry)
        Path(log_path).write_text(json.dumps(existing[-keep:], indent=2))
    except Exception:
        pass


def save_snapshot_and_clip(
    alerts_dir: Path,
    cam_id: int,
    snapshot: np.ndarray,
    clip_frames: List[np.ndarray],
    fps: int = 15,
) -> Tuple[Optional[str], Optional[str]]:
    """Write an alert snapshot JPG and (best-effort) an MP4 clip. Returns their paths."""
    Path(alerts_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    img_path = None
    clip_path = None
    try:
        if snapshot is not None and snapshot.size > 0:
            img_path = str(Path(alerts_dir) / f"alert_cam{cam_id}_{ts}.jpg")
            cv2.imwrite(img_path, snapshot)
    except Exception as exc:
        print(f"[VIOLENCE] Snapshot error: {exc}")
    try:
        if clip_frames:
            h, w = clip_frames[0].shape[:2]
            clip_path = str(Path(alerts_dir) / f"clip_cam{cam_id}_{ts}.mp4")
            writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            for frm in clip_frames:
                writer.write(frm)
            writer.release()
            print(f"[VIOLENCE] Clip saved: {clip_path}")
    except Exception as exc:
        print(f"[VIOLENCE] Clip error: {exc}")
        clip_path = None
    return img_path, clip_path


def email_enabled() -> bool:
    return bool(VIOLENCE_ALERT_SENDER and VIOLENCE_ALERT_PASSWORD and VIOLENCE_ALERT_RECEIVER)


def send_email_alert(image_path: Optional[str], cam_id: int) -> None:
    """Send an alert email if SMTP creds are configured; otherwise no-op."""
    if not email_enabled():
        return
    try:
        msg = EmailMessage()
        msg["Subject"] = f"Violence Detected - Camera {cam_id}!"
        msg["From"]    = VIOLENCE_ALERT_SENDER
        msg["To"]      = VIOLENCE_ALERT_RECEIVER
        msg.set_content(f"Violence detected on camera {cam_id} at {datetime.now()}")
        if image_path and Path(image_path).exists():
            with open(image_path, "rb") as f:
                msg.add_attachment(f.read(), maintype="image", subtype="jpeg",
                                   filename="alert.jpg")
        with smtplib.SMTP_SSL(VIOLENCE_SMTP_HOST, VIOLENCE_SMTP_PORT) as smtp:
            smtp.login(VIOLENCE_ALERT_SENDER, VIOLENCE_ALERT_PASSWORD)
            smtp.send_message(msg)
        print(f"[VIOLENCE] Email alert sent for cam{cam_id}.")
    except Exception as exc:
        print(f"[VIOLENCE] Email send failed: {exc}")
