import logging
import random
import numpy as np
import os
import os.path as osp
import yt_dlp
import cv2
import math
from pathlib import Path

from scenedetect import detect, ContentDetector

from collections import deque

from optical_flow.settings import YOUTUBE8M_IDS_FILE, YOUTUBE_URL
from optical_flow.youtube8m.download_video_ids import save_ids
from optical_flow.youtube8m.utils import count_lines, file_startswith

log = logging.getLogger(__name__)


class DownloadYouTube:

    def __init__(self, ids_buffer_size=50000):
        self.ids_buffer_size = ids_buffer_size
        self.buffer = deque(maxlen=ids_buffer_size)
        self.buffer_idx = 0
        save_ids()
        self.category_count = self.__count_categories()

    def __count_categories(self):
        count = {}
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        relative_path = current_file.relative_to(project_root)
        print(current_file)
        print(project_root)
        print(relative_path)
        with open(YOUTUBE8M_IDS_FILE, "r", encoding='utf-8') as file_read:
            while line := file_read.readline():
                category = line.split(",")[0]
                count[category] = count.get(category, 0) + 1
        return count

    def __stratified_sample(self, n_lines):
        log.info("Stratified sampling")
        rem_categories = len(self.category_count)
        curr_index = 0
        idx_ids = []
        for category, count in self.category_count.items():
            select = math.ceil(rem_categories / n_lines)
            idx_ids.extend(random.sample(range(curr_index, curr_index + count), select))

            curr_index += count
            rem_categories -= 1
            n_lines -= count

        return set(idx_ids)

    def __random_sample(self, n_lines):
        idx_ids = set(random.sample(range(0, n_lines), min(n_lines, self.ids_buffer_size)))
        return idx_ids

    def __buffer_ids(self, stratify_category=True):
        log.info(f"Buffering ids, {self.ids_buffer_size}")
        self.buffer.clear()
        n_lines = count_lines(YOUTUBE8M_IDS_FILE)
        if stratify_category:
            idx_ids = self.__stratified_sample(n_lines)
        else:
            idx_ids = self.__random_sample(n_lines)

        log.info("Selecting videos in buffer")
        with open(YOUTUBE8M_IDS_FILE, "r", encoding="utf-8") as file_read:
            i = 0
            while line := file_read.readline():
                if i in idx_ids:
                    self.buffer.append(line)
                i += 1

        self.buffer_idx = 0

    def _next_video(self):
        if len(self.buffer) == 0 or self.buffer_idx == self.ids_buffer_size - 1:
            self.__buffer_ids()

        next_value = self.buffer[self.buffer_idx]
        self.buffer_idx += 1
        return next_value

    def _download_video(self, ydl_opts, id):
        name, fps = None, None
        try:
            url = f"{YOUTUBE_URL}{id}"
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(url, download=True)
                fps = info_dict['fps']
                filename = ydl.prepare_filename(info_dict)
                name = filename
        except Exception as e:
            log.info(e)

        return name, fps

    def download(self, dir="videos", height=480):
        """
        Usual heights are 240, 480, 720, 1080
        :param dir:
        :param height:
        :return:
        """
        log.info(f"Downloading 1 video")

        line = self._next_video()
        category, id = line.strip().split(",")

        # Try to download with specified height
        ydl_opts = {'outtmpl': f'{dir}/%(title)s.%(ext)s',
                    'format': f'bv*[height={height}]+ba'}
        name, fps = self._download_video(ydl_opts, id)

        # If unable then just try the best format
        if not name:
            ydl_opts['format'] = 'best'

            name, fps = self._download_video(ydl_opts, id)

        # yt_dlp may merge files into mkv so we need to double check name by reading it from disk
        if name:
            log.info(f"Download name: {name}")
            fname = os.path.basename(name)
            fname_woext, _ = os.path.splitext(fname)

            log.info(f"Fname woext {fname_woext}")
            corrected_name = file_startswith(dir, fname_woext)
            if corrected_name:
                name = osp.join(dir, corrected_name)
            else:
                name = None

        return name, fps

    @staticmethod
    def find_scenes(video_path, threshold=20.0):
        # Create our video & scene managers, then add the detector.
        log.info(f"Detecting scenes for {video_path}...")

        # Find scenes
        scene_list = detect(video_path, ContentDetector(threshold=threshold))

        # Each returned scene is a tuple of the (start, end) timecode.
        frame_scenes = [(scene[0].get_frames(), scene[1].get_frames()) for scene in scene_list]
        
        return frame_scenes

    @staticmethod
    def skip_rate(every_fps, fps):
        skip = None
        if every_fps is not None:
            skip = int(round(fps / every_fps))
            if skip <= 1:
                skip = None
        return skip

    @staticmethod
    def split_frames(video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            yield frame

        cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def mid_scene_interval(scenes, required_interval, lower_exclusion=0.15, upper_exclusion=0.85):
        n_frames = int(scenes[-1][1])
        allowed_scenes = [scene for scene in scenes if int(scene[1]) >= lower_exclusion * n_frames and
                          int(scene[0]) <= upper_exclusion * n_frames]

        sizes = [scene[1] - scene[0] for scene in allowed_scenes]
        if max(sizes) < required_interval:
            return allowed_scenes[np.argmax(np.array(sizes))]

        mid_frame = int(scenes[-1][1]) / 2
        best_closeness, best_scene = 9E10, None
        for scene in allowed_scenes:
            mid_point = (int(scene[1]) - int(scene[0])) / 2
            closeness = np.abs(mid_frame - mid_point)
            if closeness < best_closeness:
                best_closeness = closeness
                best_scene = scene

        return best_scene

    @staticmethod
    def eligible_frames(n_frames, skip, scenes):
        """
        Calculate a stable, continuous frame range within a single video scene.

        This method selects the most centrally located, continuous scene that can 
        accommodate the requested number of frames and downsampling interval. It 
        centers the extraction window within that scene and applies a 3-frame safety 
        buffer at the scene boundaries to avoid video encoding/transition artifacts 
        detrimental to optical flow computation.

        Parameters
        ----------
        n_frames : int or None
            The number of target frames required for the sequence. If None, the 
            entire video range is returned.
        skip : int or None
            The frame sampling rate (stride). For example, a skip of 2 selects every 
            other frame. If None, the entire video range is returned.
        scenes : list of tuple of (int, int)
            A list of detected scenes, where each tuple contains the start and 
            end frame indices `(start_frame, end_frame)`.

        Returns
        -------
        first_frame : int
            The calculated start frame index for the safe extraction window.
        last_frame : int
            The calculated end frame index for the safe extraction window.
            
        See Also
        --------
        mid_scene_interval : Selects the best candidate scene based on size and centrality.
        """
        if n_frames is None or skip is None:
            return int(scenes[0][0]), int(scenes[-1][1])

        required_interval = n_frames * skip
        scene = DownloadYouTube.mid_scene_interval(scenes, required_interval + 4)
        scene_0 = int(scene[0])
        scene_1 = int(scene[1])
        mid_scene = (scene_0 + scene_1) / 2
        mid_interval = required_interval / 2
        first_frame = int(max(scene_0 + 3, mid_scene - mid_interval))
        last_frame = int(min(scene_1 - 3, mid_scene + mid_interval))

        return first_frame, last_frame

    def gen_frames(self, every_fps=None, n_frames=None, width=640, height=480):
        while True:
            name, fps = self.download(height=height)
            print("Name of video: ", name)

            if not name:
                continue

            skip = DownloadYouTube.skip_rate(every_fps, fps)
            scenes = DownloadYouTube.find_scenes(name)
            start_frame, end_frame = DownloadYouTube.eligible_frames(n_frames, skip, scenes)

            next_scene = scenes[0][1]
            frame_idx = 0
            scene_idx = 0
            repeated_frames = 0
            prev_frame = None
            log.info(f"Elligible frames {start_frame}, {end_frame}")
            for frame in DownloadYouTube.split_frames(name):
                if prev_frame is not None and np.array_equal(prev_frame, frame): # Keep track of repeated frames, some video are just images
                    repeated_frames += 1
                else:
                    repeated_frames = 0

                scene_change = 0

                if next_scene == frame_idx:
                    scene_change = 1
                    scene_idx += 1
                    if scene_idx < len(scenes) - 1:
                        next_scene = scenes[scene_idx][1]
                frame_idx += 1

                if start_frame <= frame_idx <= end_frame:
                    if scene_change or (skip and (frame_idx - start_frame) % skip == 0):
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                        yield name, frame, scene_change, repeated_frames
                if frame_idx > end_frame:
                    break

                prev_frame = frame

            os.remove(name)

    @staticmethod
    def split_frames_pairs(video_path, scenes, skip=0):
        # Create buffer to store previous scenes
        buffer = deque(maxlen=skip + 1)
        next_scene = scenes[0][1]

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        scene_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if next_scene == frame_idx:
                scene_idx += 1
                if scene_idx < len(scenes) - 1:
                    next_scene = scenes[scene_idx][1]
                buffer.clear()

            if len(buffer) == skip + 1:
                yield frame, buffer[0], [buffer[i] for i in range(len(buffer)) if i > 0]

            buffer.append(frame)
            frame_idx += 1

        cap.release()
        cv2.destroyAllWindows()

    def gen_frames_pairs(self, skip_frames=0):
        while True:
            name, fps = self.download()

            if not name:
                continue

            scenes = DownloadYouTube.find_scenes(name)  # separate scenes starting from frame 0
            for frame_tuple in DownloadYouTube.split_frames_pairs(name, scenes, skip=skip_frames):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame_tuple

            os.remove(name)