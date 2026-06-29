import cv2
import random
import string
import os
import numpy as np

from collections import Counter


def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frames


def count_lines(filename):
    file = open(filename, "r", encoding="utf-8")
    nonempty_lines = [line.strip("\n") for line in file if line != "\n"]
    line_count = len(nonempty_lines)
    file.close()

    return line_count


def keep_after(yield_list, check_list, check_item):
    keep = False
    for item, check in zip(yield_list, check_list):
        if check == check_item:
            keep = True
        if keep:
            yield item


def random_name(positions):
    return ''.join(random.choices(string.ascii_lowercase, k=positions))


def file_startswith(dir, filename):
    starts_with = []
    for file in os.listdir(dir):
        if file.startswith(filename):
            starts_with.append(file)

    if len(starts_with) == 0:
        return None
    elif len(starts_with) == 1:
        return starts_with[0]
    else:
        counts = np.array([Counter(f)["."] for f in starts_with])
        return starts_with[np.argmin(counts)]
