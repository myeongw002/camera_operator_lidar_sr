"""Camera-guided local-operator LiDAR super-resolution."""

from .models.student import LidarOperatorStudent
from .models.teacher import CameraGuidedOperatorTeacher
from .models.outputs import OperatorOutput

__all__ = ["LidarOperatorStudent", "CameraGuidedOperatorTeacher", "OperatorOutput"]
