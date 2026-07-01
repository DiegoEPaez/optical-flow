import sys
import torch
import torch.nn.functional as F
import math
import numpy as np
import logging

from collections import deque

import matplotlib.pyplot as plt
import flow_vis

from torch.autograd import Variable
from torch.utils.data import DataLoader

import os
import os.path as osp

from optical_flow.models.flownets import FlowNetS, FlowNetModified
from optical_flow.models.util import *
from optical_flow.models.datasets import YoutubeDataset, SintelDataset

DATASETS = {
    "YoutubeDataset": YoutubeDataset,
    "SintelDataset": SintelDataset,
}

MODELS = {
    "FlowNetS": FlowNetS,
    "FlowNetModified": FlowNetModified,
}

log = logging.getLogger(__name__)

def photometric_loss(im1, im2, eps=0.01, q=0.4):
    dif = im1 - im2
    
    return ((torch.abs(dif) + eps) ** q).sum()


def multiscale_photometric(network_output, images1, images2, weights=None):
    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # FlowNet weights
    assert(len(weights) == len(network_output))

    loss = 0
    for flow, weight in zip(network_output, weights):
        upscaled_flow = one_scale(images1, flow)
        warped2_1 = warp(images2, upscaled_flow)
        loss += weight * photometric_loss(images1, warped2_1)

    return loss

def train(net, model_file, dataset=YoutubeDataset, benchmark_sintel=False):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training on device {device}")

    net.to(device)

    q = deque(maxlen=20)

    if osp.exists(model_file):
        net.load_state_dict(torch.load(model_file))
    elif not osp.exists(osp.dirname(model_file)):
        os.makedirs(osp.dirname(model_file))
    
    dataloader = DataLoader(dataset(), batch_size=10, shuffle=True, num_workers=0)

    i = 0
    while True:
        for images,_ in dataloader:
            images1, images2, flipped_images = splitted_im(images)
            images = images.to(device)
            images2 = images2.to(device)
            images1 = images1.to(device)
            flipped_images = flipped_images.to(device)
            
            flows1 = net(images)

            loss = multiscale_photometric(flows1, images1, images2)
            
            flows2 = net(flipped_images)

            loss += multiscale_photometric(flows2, images2, images1)
            
            loss.backward()
            q.append(loss)
            
            optimizer.step()
            optimizer.zero_grad()
            
            if i % 200 == 0:
                torch.save(net.state_dict(), model_file)


            if i % 1000 == 0:
                if benchmark_sintel:
                    log.info("EPE sintel: %f", epe_sintel(net))
            
            log.info("Batch %d loss %f", i, sum(q) /len(q))
            i += 1


def run(dataset=YoutubeDataset, model=FlowNetS):
    net = model()

    model_dir = osp.join("trained", f"{dataset.__name__}_{model.__name__}.pth")

    train(net, model_dir, dataset, benchmark_sintel=True)


def main(dataset="YoutubeDataset", model="FlowNetS"):
    run(dataset=DATASETS[dataset], model=MODELS[model])