import os
import os.path as osp
import logging
from PIL import Image
import shutil

from optical_flow.youtube8m.download_youtube import DownloadYouTube

log = logging.getLogger(__name__)


def download_frames(every_fps=10, n_frames=100):
    """
    Download frames at given fps rate and the provided number of frames from the YouTube 8M Dataset.
    """
    dwld = DownloadYouTube()
    frame_idx = 0
    invalid_video = ""
    for name, frame, scene_change, repeated_frames in dwld.gen_frames(every_fps=every_fps, n_frames=n_frames):
        if name == invalid_video:
            continue

        fname = os.path.basename(name)
        fname_woext, _ = os.path.splitext(fname)
        curr_path = osp.join("frames", fname_woext)
        if not osp.exists(curr_path):
            frame_idx = 0
            os.makedirs(curr_path)

        im = Image.fromarray(frame)
        im_name = osp.join(curr_path,f"{frame_idx:06d}-{scene_change}.jpg")
        im.save(im_name)

        if frame_idx == 0:
            log.info(f"Saved: {im_name}")

        if repeated_frames > 50:   # most likely video is just images
            shutil.rmtree(curr_path)
            invalid_video = name
            log.info(f"Ignoring video {curr_path} since it has too many repeated frames")
        frame_idx += 1


def stream_frames():
    """
    Stream videos instead of downloading them.
    """
    dwld = DownloadYouTube()

    for first, second, middle_frames in dwld.gen_frames():
        print(f"Training with frames: {first.shape}, {second.shape}")
        # Do whatever you want with frames - first, second, middle frames (for there to be middle frames,
        # skip frames in gen_frames must be > 0.


def main():
    if not osp.exists("logs"):
        os.mkdir("logs")
    logging.basicConfig(filename='logs/app.log',
                        format="%(asctime)s %(levelname)s-%(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    download_frames()


if __name__ == '__main__':
    main()
