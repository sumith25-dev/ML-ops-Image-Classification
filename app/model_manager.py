"""
ModelManager — loads model versions from MLflow registry,
handles A/B traffic routing, and runs inference.
"""

import os
import random
import hashlib
import logging
from typing import Tuple

import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image
import requests
from io import BytesIO

import mlflow
import mlflow.pytorch

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
CLASSES = [
    "buildings", "forest", "glacier",
    "mountain", "sea", "street",
]

TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ModelManager:
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self._models: dict[str, torch.nn.Module] = {}
        self.ab_config = {
            "stable": 0.80,   # 80 % traffic to production model
            "canary": 0.20,   # 20 % traffic to new candidate
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Loading ───────────────────────────────────────────────────────────────
    def load_models(self):
        """
        Try to load from MLflow registry; fall back to a freshly initialised
        EfficientNet so the server always starts successfully (useful for local
        dev / CI without a real MLflow backend).
        """
        for alias in ("stable", "canary"):
            try:
                model_uri = f"models:/image_classifier/{alias}"
                model = mlflow.pytorch.load_model(model_uri, map_location=self.device)
                model.eval()
                self._models[alias] = model
                logger.info(f"Loaded '{alias}' model from MLflow registry.")
            except Exception as exc:
                logger.warning(
                    f"Could not load '{alias}' from MLflow ({exc}). "
                    "Using randomly-initialised EfficientNet as placeholder."
                )
                self._models[alias] = self._make_placeholder_model()

    def _make_placeholder_model(self) -> torch.nn.Module:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASSES))
        model.eval()
        return model.to(self.device)

    # ── A/B routing ───────────────────────────────────────────────────────────
    def route_ab(self, user_id: str) -> str:
        """
        Deterministic per-user routing: same user always hits same model version.
        Hash the user_id → float in [0,1) → compare against cumulative weights.
        """
        digest = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        bucket = (digest % 10_000) / 10_000.0
        cumulative = 0.0
        for version, weight in self.ab_config.items():
            cumulative += weight
            if bucket < cumulative:
                return version
        return "stable"

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, version: str, image_url: str) -> Tuple[str, float]:
        if version not in self._models:
            raise ValueError(f"Unknown model version: {version}")

        img = self._fetch_image(image_url)
        tensor = TRANSFORM(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self._models[version](tensor)
            probs  = torch.softmax(logits, dim=1)[0]
            top_idx = probs.argmax().item()

        return CLASSES[top_idx], probs[top_idx].item()

    def _fetch_image(self, url: str) -> Image.Image:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Cannot fetch image from '{url}': {exc}")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def loaded_versions(self) -> list[str]:
        return list(self._models.keys())
