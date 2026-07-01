"""CLI entrypoint for optical-flow scripts."""

import logging
import os
import os.path as osp
import sys
from pathlib import Path

import click


def _setup_app_logging():
    if not osp.exists("logs"):
        os.mkdir("logs")

    log_format = "%(asctime)s %(levelname)s-%(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(log_format, date_format))

    handlers = [stream_handler]

    if not os.environ.get("AWS_BATCH_JOB_ID"):
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(log_dir / "app.log")
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(file_handler)

    logging.basicConfig(level=logging.INFO, handlers=handlers) 


@click.group()
def cli():
    """Optical flow pipeline commands."""


@cli.command("download-video-ids")
def download_video_ids():
    """Download YouTube-8M video IDs to config/youtube8m_ids.txt."""
    _setup_app_logging()
    from optical_flow.youtube8m.download_video_ids import save_ids

    save_ids()


@cli.command("download-images")
@click.option(
    "--every-fps",
    type=int,
    default=10,
    show_default=True,
    help="Sample one frame every N frames.",
)
@click.option(
    "--n-frames",
    type=int,
    default=100,
    show_default=True,
    help="Maximum number of frames to download per video.",
)
def download_images(every_fps, n_frames):
    """Download and extract frames from YouTube-8M videos."""
    _setup_app_logging()
    from optical_flow.scripts.download_images import download_frames
    
    download_frames(every_fps=every_fps, n_frames=n_frames)


@cli.command("stream-frames")
def stream_frames():
    """Stream video frame pairs instead of saving frames to disk."""
    _setup_app_logging()
    from optical_flow.scripts.download_images import stream_frames as stream

    stream()


@cli.command("train-flownet")
@click.option(
    "--dataset",
    type=click.Choice(["YoutubeDataset", "SintelDataset"], case_sensitive=False),
    default="YoutubeDataset",
    show_default=True,
    help="Training dataset.",
)
@click.option(
    "--model",
    type=click.Choice(["FlowNetS", "FlowNetModified"], case_sensitive=False),
    default="FlowNetS",
    show_default=True,
    help="FlowNet architecture to train.",
)
def train_flownet(dataset, model):
    """Train FlowNet with self-supervised photometric loss."""
    from optical_flow.scripts.train_flownet import main

    main(dataset=dataset, model=model)


@cli.command("train-hrnet")
def train_hrnet():
    """Experimental HRNet encoder training."""
    from optical_flow.scripts.train_hrnet import main

    main()


def main():
    cli()


if __name__ == "__main__":
    main()
