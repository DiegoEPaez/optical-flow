# Optical Flow

Thesis project investigating **self-supervised optical flow** on the **YouTube-8M** dataset using FlowNet.

## Overview

Implements FlowNet Simple for learning optical flow without ground-truth labels via photometric consistency on video pairs.

## Features

- YouTube-8M video ID & category downloader
- FlowNet Simple (original + modified with per-level masks)
- Self-supervised multi-scale photometric loss training
- Sintel dataset support for evaluation
- HRNet backbone (experimental)

## Running the Project
Make sure you have python with poetry installed.
Also install ffmpeg.

First, install the environment (requires python 3.12):
poetry install

The next command will download youtube8m video ids, then it will use these ids to download youtube videos and split them into frames, by scene.
The downloaded frames are the ones with continuous scenes, near the middle of the film:
python src/optical_flow/scripts/download_images.py

Train
Bashpython scripts/train_flownet.py
References

FlowNet
YouTube-8M


Python • Research/Educational
