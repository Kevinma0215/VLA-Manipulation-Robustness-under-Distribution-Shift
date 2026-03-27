"""Gemini-based semantic planner. Canonical location: vla_manipulation/policy/hierarchical/gemini_planner.py"""
from __future__ import annotations
import base64
import json
import logging
from dataclasses import dataclass
from typing import List

import numpy as np

from vla_manipulation.policy.hierarchical.config import ActionType

log = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"


@dataclass
class SemanticWaypoint:
    action_type:   ActionType
    pixel_u:       int
    pixel_v:       int
    gripper_state: float  # 0=open, 1=closed

    @property
    def pixel_coords(self):
        return (self.pixel_u, self.pixel_v)

    @property
    def gripper(self):
        return self.gripper_state


class GeminiPlanner:
    """Calls Gemini API to produce a list of SemanticWaypoints from an RGB image + language command."""

    def __init__(self, api_key: str, model: str = GEMINI_MODEL):
        self.model = model
        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
        except ImportError:
            log.warning("google-genai not installed; GeminiPlanner will fail at plan() time.")
            self._client = None

    def plan(self, image: np.ndarray, command: str) -> List[SemanticWaypoint]:
        """
        Send image + command to Gemini and parse the returned JSON waypoints.
        Returns a list of SemanticWaypoint.
        """
        if self._client is None:
            raise RuntimeError("google-genai not installed.")

        import PIL.Image
        import io

        pil_img = PIL.Image.fromarray(image)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        prompt = (
            f"Task: {command}\n\n"
            "Respond with a JSON array of waypoints. Each waypoint has:\n"
            "  action_type: one of 'move', 'grasp', 'place', 'open'\n"
            "  pixel_u: integer x pixel coordinate in the image\n"
            "  pixel_v: integer y pixel coordinate in the image\n"
            "  gripper_state: 0.0 (open) or 1.0 (closed)\n\n"
            "Return ONLY the JSON array, no other text."
        )

        from google.genai import types as genai_types
        response = self._client.models.generate_content(
            model=self.model,
            contents=[
                genai_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                prompt,
            ],
        )

        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        raw = json.loads(text)

        waypoints = []
        for wp in raw:
            waypoints.append(SemanticWaypoint(
                action_type   = ActionType(wp["action_type"]),
                pixel_u       = int(wp["pixel_u"]),
                pixel_v       = int(wp["pixel_v"]),
                gripper_state = float(wp["gripper_state"]),
            ))
        return waypoints
