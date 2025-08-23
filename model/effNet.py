"""EfficientNet-B0 classifier wrapper using PyTorch/torchvision.

Provides a small object-oriented wrapper that builds an EfficientNet-B0
backbone, replaces the classifier head to match a target number of classes,
and exposes convenience methods for saving/loading and prediction.

Usage:
	from model.model1 import EfficientNetB0Classifier
	m = EfficientNetB0Classifier(num_classes=3, pretrained=False)
	logits = m(torch.randn(1, 3, 224, 224))
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
	# Prefer torchvision native EfficientNet if available
	from torchvision.models import efficientnet_b0
except Exception:
	efficientnet_b0 = None


class EfficientNetB0Classifier(nn.Module):
	"""Wrapper around torchvision's EfficientNet-B0.

	Args:
		num_classes: number of output classes for the classifier head.
		pretrained: whether to load pretrained ImageNet weights (if available).
		device: torch device or string; defaults to CUDA if available else CPU.
	"""

	def __init__(self, num_classes: int = 2, pretrained: bool = False, device: Optional[str] = None):
		super().__init__()
		if efficientnet_b0 is None:
			raise RuntimeError("torchvision.models.efficientnet_b0 is unavailable in this environment")

		self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# Build base model
		self.model = efficientnet_b0(weights=pretrained)

		# Replace classifier head if needed
		replaced = False
		if hasattr(self.model, "classifier"):
			head = self.model.classifier
			# Typical torchvision EfficientNet classifier is nn.Sequential(..., Linear(in_feat, out_feat))
			if isinstance(head, nn.Sequential) and len(head) > 0 and isinstance(head[-1], nn.Linear):
				in_features = head[-1].in_features
				head[-1] = nn.Linear(in_features, num_classes)
				self.model.classifier = head
				replaced = True
		if not replaced:
			# Fallback: try common attribute name
			if hasattr(self.model, "fc") and isinstance(self.model.fc, nn.Linear):
				in_features = self.model.fc.in_features
				self.model.fc = nn.Linear(in_features, num_classes)
				replaced = True

		if not replaced:
			# As a last resort, attach a simple classifier
			self.model.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1280, num_classes))

		# Move model to device
		self.to(self.device)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Forward pass returning raw logits."""
		x = x.to(self.device)
		return self.model(x)

	def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
		"""Return class probabilities (softmax) for input tensor."""
		logits = self.forward(x)
		return F.softmax(logits, dim=1)

	def predict(self, x: torch.Tensor) -> torch.Tensor:
		"""Return predicted class indices."""
		probs = self.predict_proba(x)
		return torch.argmax(probs, dim=1)

	def save(self, path: str) -> None:
		"""Save state_dict to the given path."""
		torch.save(self.state_dict(), path)

	def load(self, path: str, map_location: Optional[str] = None) -> None:
		"""Load state_dict from the given path."""
		state = torch.load(path, map_location=map_location or self.device)
		self.load_state_dict(state)


__all__ = ["EfficientNetB0Classifier"]


if __name__ == "__main__":
	# Quick smoke test (does not download weights when pretrained=False)
	m = EfficientNetB0Classifier(num_classes=2, pretrained=False)
	x = torch.randn(1, 3, 224, 224)
	out = m(x)
	print("Output shape:", out.shape)

