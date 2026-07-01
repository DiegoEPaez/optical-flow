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
The project is designed to run as a containerized application, although it requires the sintel
and YouTube videos to be available. YouTube videos can be obtained by running:

python __main__.py download-images

Sintel data must be downloaded from the webpage:

https://sintel.is.tue.mpg.de/

Make sure torch and torchvision in pyproject.toml work accordingly to your setup.
Also, the project assumes the data can be read from an S3 AWS bucket, please
modify accordingly.

FlowNet
YouTube-8M


Python • Research/Educational
