"""
Canonical Constants and Class Mappings for DOTA Aerial Object Detection.
"""

DOTA_CLASSES = [
    "plane",
    "ship",
    "storage-tank",
    "baseball-diamond",
    "tennis-court",
    "basketball-court",
    "ground-track-field",
    "harbor",
    "bridge",
    "large-vehicle",
    "small-vehicle",
    "helicopter",
    "roundabout",
    "soccer-ball-field",
    "swimming-pool",
]

CLASS_TO_ID = {cls_name: i for i, cls_name in enumerate(DOTA_CLASSES)}
ID_TO_CLASS = {i: cls_name for i, cls_name in enumerate(DOTA_CLASSES)}
