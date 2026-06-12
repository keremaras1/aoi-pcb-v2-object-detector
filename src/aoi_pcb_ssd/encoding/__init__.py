# SPDX-License-Identifier: Apache-2.0
"""SSD encoding and decoding: anchor grid generation, label encoding, and prediction decoding."""

from aoi_pcb_ssd.encoding.input_encoder import SSDInputEncoder
from aoi_pcb_ssd.encoding.matching import match_by_nearest_centre
from aoi_pcb_ssd.encoding.output_decoder import decode_detections

__all__ = [
    "match_by_nearest_centre",
    "SSDInputEncoder",
    "decode_detections",
]
