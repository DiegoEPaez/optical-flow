# Optical Flow

Thesis project investigating **self-supervised optical flow** on the **YouTube-8M** dataset using FlowNet. Thesis supervisor: Michael Romanov
RomanovMikeV.

## Overview

Implements FlowNet Simple for learning optical flow without ground-truth labels via photometric consistency on video pairs.

## Features

- YouTube-8M video ID & category downloader (categories obtained from https://github.com/gsssrao/youtube-8m-videos-frames)
- FlowNet Simple (original + modified with per-level masks)
- Self-supervised multi-scale photometric loss training
- Sintel dataset support for evaluation
- HRNet backbone (experimental)

## Running the Project
Make sure you have python with poetry installed.
Also install ffmpeg.

Download the sintel dataset for benchmarking:
https://sintel.is.tue.mpg.de/downloads

First, install the environment (requires python 3.12):
poetry install

Download youtube8m videos
python src/optical_flow/scripts/download_images.py

Train flownet:
python scripts/train_flownet.py
