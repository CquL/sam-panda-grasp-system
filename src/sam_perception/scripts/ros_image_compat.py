#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import cv2
import numpy as np
from sensor_msgs.msg import Image


_ENCODING_SPECS = {
    "rgb8": (np.uint8, 3),
    "bgr8": (np.uint8, 3),
    "mono8": (np.uint8, 1),
    "8UC1": (np.uint8, 1),
    "8UC3": (np.uint8, 3),
    "mono16": (np.uint16, 1),
    "16UC1": (np.uint16, 1),
    "16SC1": (np.int16, 1),
    "32FC1": (np.float32, 1),
}


def _byteorder_matches_message(msg, dtype):
    if np.dtype(dtype).itemsize <= 1:
        return True
    message_big_endian = bool(getattr(msg, "is_bigendian", 0))
    host_big_endian = sys.byteorder == "big"
    return message_big_endian == host_big_endian


def _reshape_image_buffer(msg, dtype, channels):
    itemsize = np.dtype(dtype).itemsize
    row_stride = int(msg.step)
    if row_stride <= 0:
        row_stride = int(msg.width) * channels * itemsize

    if row_stride % itemsize != 0:
        raise ValueError(f"Invalid row stride {row_stride} for dtype itemsize {itemsize}")

    raw = np.frombuffer(msg.data, dtype=dtype)
    if not _byteorder_matches_message(msg, dtype):
        raw = raw.byteswap().view(raw.dtype.newbyteorder("="))

    elems_per_row = row_stride // itemsize
    expected_rows = int(msg.height) * elems_per_row
    if raw.size < expected_rows:
        raise ValueError(
            f"Image buffer too small: got {raw.size} elements, expected at least {expected_rows}"
        )

    raw = raw[:expected_rows].reshape(int(msg.height), elems_per_row)
    if channels == 1:
        return raw[:, : int(msg.width)].copy()

    needed_cols = int(msg.width) * channels
    if elems_per_row < needed_cols:
        raise ValueError(
            f"Image row too small: got {elems_per_row} elements, need {needed_cols}"
        )
    return raw[:, :needed_cols].reshape(int(msg.height), int(msg.width), channels).copy()


def image_msg_to_numpy(msg, desired_encoding="passthrough"):
    encoding = str(getattr(msg, "encoding", "") or "").strip()
    if not encoding:
        raise ValueError("ROS Image message has empty encoding")
    if encoding not in _ENCODING_SPECS:
        raise NotImplementedError(f"Unsupported ROS image encoding: {encoding}")

    dtype, channels = _ENCODING_SPECS[encoding]
    image = _reshape_image_buffer(msg, dtype, channels)

    desired = str(desired_encoding or "passthrough").strip()
    if desired in ("", "passthrough", encoding):
        return image

    if encoding == "rgb8" and desired == "bgr8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "bgr8" and desired == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if encoding in ("mono8", "8UC1") and desired == "bgr8":
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if encoding in ("mono8", "8UC1") and desired == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if encoding == "rgb8" and desired == "mono8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if encoding == "bgr8" and desired == "mono8":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    raise NotImplementedError(f"Unsupported conversion: {encoding} -> {desired}")


def numpy_to_image_msg(image, encoding="bgr8", stamp=None, frame_id=""):
    if encoding not in _ENCODING_SPECS:
        raise NotImplementedError(f"Unsupported ROS image encoding: {encoding}")

    dtype, channels = _ENCODING_SPECS[encoding]
    array = np.asarray(image)
    if channels == 1:
        if array.ndim != 2:
            raise ValueError(f"Encoding {encoding} requires a 2D array, got shape {array.shape}")
    else:
        if array.ndim != 3 or array.shape[2] != channels:
            raise ValueError(
                f"Encoding {encoding} requires shape (H, W, {channels}), got {array.shape}"
            )

    array = np.ascontiguousarray(array.astype(dtype, copy=False))
    msg = Image()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    msg.height = int(array.shape[0])
    msg.width = int(array.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = 1 if sys.byteorder == "big" else 0
    msg.step = int(array.strides[0])
    msg.data = array.tobytes()
    return msg
