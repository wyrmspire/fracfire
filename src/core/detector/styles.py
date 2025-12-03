"""
Configurable detection styles for the Setup Detector.
"""
from dataclasses import dataclass

@dataclass
class DetectionStyle:
    name: str
    description: str
    # Add more configuration fields as needed
