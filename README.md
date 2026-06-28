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

## Structure
├── youtube8m/      # Data download tools
├── models/         # FlowNet implementations
├── scripts/        # Training & utility scripts
├── config/         # Configuration
├── settings.py
└── tests/
text## Installation

```bash
git clone https://github.com/DiegoEPaez/optical-flow.git
cd optical-flow
pip install torch torchvision opencv-python matplotlib
Usage
Download YouTube-8M IDs
Bashpython youtube8m/download_video_ids.py
Train
Bashpython scripts/train_flownet.py
References

FlowNet
YouTube-8M


Python • Research/Educational
