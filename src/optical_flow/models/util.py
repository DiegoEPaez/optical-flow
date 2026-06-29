import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable

import os
import os.path as osp

import matplotlib.pyplot as plt
        
def warp(x, flo):
    """
    Function from repo PWC-Net:https://github.com/NVlabs/PWC-Net
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
 
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size()))
    if x.is_cuda:
        mask = mask.cuda()

    mask = nn.functional.grid_sample(mask, vgrid)

    # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

    mask[mask<0.9999] = 0
    mask[mask>0] = 1

    return output*mask


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )


def predict_flow(in_planes, flow_layers=None):
    if flow_layers:
        return nn.Conv2d(in_planes,2 * flow_layers,kernel_size=3,stride=1,padding=1,bias=False)
    else:
        return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=False)

def predict_mask(in_planes, flow_layers):
    return nn.Sequential(
        nn.Conv2d(in_planes,2 * flow_layers,kernel_size=3,stride=1,padding=1,bias=False),
        nn.Sigmoid()
    )

def predict_flow_with_mask(flows, masks):
    masked_flows = flows * masks
    out_flow = torch.zeros((flows.shape[0], 2, *flows.shape[2:]), device='cuda')
    
    for i in range(0, flows.shape[1], 2):
        out_flow = out_flow + masked_flows[:, i:i+2, :, :] 
    return out_flow

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )


def correlate(input1, input2):
    out_corr = spatial_correlation_sample(input1,
                                          input2,
                                          kernel_size=1,
                                          patch_size=21,
                                          stride=1,
                                          padding=0,
                                          dilation_patch=2)
    # collate dimensions 1 and 2 in order to be treated as a
    # regular 4D tensor
    b, ph, pw, h, w = out_corr.size()
    out_corr = out_corr.view(b, ph * pw, h, w)/input1.size(1)
    return F.leaky_relu_(out_corr, 0.1)


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]


def read_flo_file(filename, memcached=False):
    """
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    if memcached:
        filename = io.BytesIO(filename)
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)[0]
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h, w, 2))
    f.close()
    return data2d

def gather_nd(params, indices):
    '''
    4D example
    params: tensor shaped [n_1, n_2, n_3, n_4] --> 4 dimensional
    indices: tensor shaped [m_1, m_2, m_3, m_4, 4] --> multidimensional list of 4D indices
    
    returns: tensor shaped [m_1, m_2, m_3, m_4]
    
    ND_example
    params: tensor shaped [n_1, ..., n_p] --> d-dimensional tensor
    indices: tensor shaped [m_1, ..., m_i, d] --> multidimensional list of d-dimensional indices
    
    returns: tensor shaped [m_1, ..., m_1]
    '''

    out_shape = indices.shape[:-1]
    indices = indices.unsqueeze(0).transpose(0, -1) # roll last axis to fring
    ndim = indices.shape[0]
    indices = indices.long()
    idx = torch.zeros_like(indices[0], device=indices.device).long()
    m = 1
    
    for i in range(ndim)[::-1]:
        idx += indices[i] * m 
        m *= params.size(i)
    out = torch.take(params, idx)

    return out.view(out_shape)
    
def occ_grid(flo):
    B, _, H, W = flo.size()
    
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    
    # Note that x is first since flows have x component first
    grid = torch.cat((xx,yy),1).float()
    
    if flo.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid)
    
    return vgrid


def reversed_flow(warp_fwd, bflo):
    B, _, H, W = warp_fwd.size()
    
    # In flows, X component is first - this should be the case for warp_fwd and bflo; however here we flip values
    # since the gather_nd function does not know anything about this weird convention. Also values are clipped
    warp_fwdc = warp_fwd.round()
    warp_fwdc[:, 1] = torch.clip(warp_fwd[:, 0], 0, W - 1)
    warp_fwdc[:, 0] = torch.clip(warp_fwd[:, 1], 0, H - 1)
    warp_fwdc = warp_fwdc.permute(0, 2, 3, 1)
    
    # Select values specified by warp_fwd
    select = torch.stack([torch.cat([torch.zeros(B, H, W, 1, device='cuda'), warp_fwdc], dim = -1),
                          torch.cat([torch.ones(B, H, W, 1, device='cuda'), warp_fwdc], dim = -1)], dim = 1)

    batch_selection = []
    for i in range(B):
        batch_selection.append(i * torch.ones(2, H, W, 1, device='cuda'))

    select = torch.cat([torch.stack(batch_selection, dim=0), select], dim = -1)
    rev_flo = gather_nd(bflo, select)

    return  rev_flo


def occlusion_map(flo, bflo, alpha1=0.01, alpha2=0.5):
    B, _, H, W = flo.size()
    grid = occ_grid(flo)
    
    warp_fwd = grid + flo

    O_map = torch.logical_or(warp_fwd[:, 0] > W, warp_fwd[:, 0] < 0) 
    O_map = torch.logical_or(O_map, warp_fwd[:, 1] > H)
    O_map = torch.logical_or(O_map, warp_fwd[:, 1] < 0)

    rev_flo = reversed_flow(warp_fwd, bflo)
    
    ls = torch.norm(rev_flo + flo, dim=1) ** 2
    rs = alpha1 * (torch.norm(rev_flo, dim=1) ** 2 + torch.norm(flo, dim=1) **2) + alpha2
    
    O_map = torch.logical_or(O_map, ls >= rs).int()
    
    O_map = O_map.reshape(O_map.shape[0], 1, *O_map.shape[1:])
    O_map = O_map.repeat(1, 3, 1, 1)
    
    return O_map.int(), rev_flo

def epe(flo1, flo2, channel_dim=0):
    assert flo1.shape[channel_dim] == 2
    assert flo2.shape[channel_dim] == 2
    return np.linalg.norm(flo1 - flo2, axis=channel_dim).mean()


def epe_torch(flo1, flo2, channel_dim=0):
    return torch.norm(flo1 - flo2, dim=channel_dim).mean()


def multiscale_epe(flows_fwd, real_flows, weights=None):
    # Similar to multiscale_photometric
    if type(flows_fwd) not in [tuple, list]:
        flows_fwd = [flows_fwd]
    if weights is None:
        weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # FlowNet weights
    assert(len(weights) == len(flows_fwd))
    
    loss = 0
    epe_losss = []
    for flow_fwd, weight in zip(flows_fwd, weights):

        downscaled_real_flows = one_scale(flow_fwd, real_flows)
        
        epe_loss = epe_torch(flow_fwd, downscaled_real_flows, 1)
        epe_losss.append(epe_loss.detach().cpu().numpy())
        loss += epe_loss

    return loss, epe_losss[0]


def epe_sintel(net, base_sintel='mpi_sintel/training', device='cuda'):
    folders = sorted(os.listdir(f'{base_sintel}/flow'))
    epes = []
    for folder in folders:
        # Get image names
        # im_names = sorted(os.listdir(folder))
        im_names = sorted([name[:-4] for name in os.listdir(osp.join(base_sintel, 'final', folder))])
        prev_name = None
        for name in im_names:
            image = plt.imread(osp.join(base_sintel, 'final', folder, name + '.png'))
            if prev_name is None:
                prev_name = name
                prev_image = image
                continue

            # Read flow
            flo = read_flo_file(osp.join(base_sintel, 'flow', folder, prev_name + '.flo'))
            flo_ch = np.transpose(flo, (2, 0, 1))
            bflo = flo_ch.reshape(1, *flo_ch.shape)

            # Estimate flow
            ims = np.transpose(np.c_[prev_image, image], (2, 0, 1))
            ims = ims.reshape(1, *ims.shape)
            ims = torch.from_numpy(ims).to(device)
            eflo = net(ims)[0]
            eflo = one_scale(torch.from_numpy(bflo), eflo).cpu().detach().numpy()

            eflo = eflo.reshape(*eflo.shape[1:])
            epev = epe(flo_ch, eflo)
            
            epes.append(epev)
            prev_name = name
            prev_image = image
            

    return np.array(epes).mean(), np.median(np.array(epes))


def epe_base_sintel(base_sintel='mpi_sintel/training'):
    folders = sorted(os.listdir(f'{base_sintel}/flow'))
    epes = []
    for folder in folders:
        # Get image names
        # im_names = sorted(os.listdir(folder))
        im_names = sorted([name[:-4] for name in os.listdir(osp.join(base_sintel, 'final', folder))])
        prev_name = None
        for name in im_names:
            image = plt.imread(osp.join(base_sintel, 'final', folder, name + '.png'))
            if prev_name is None:
                prev_name = name
                prev_image = image
                continue

            # Read flow
            flo = read_flo_file(osp.join(base_sintel, 'flow', folder, prev_name + '.flo'))
            flo_ch = np.transpose(flo, (2, 0, 1))
            
            # Estimate flow
            eflo = np.zeros(flo_ch.shape)
            epes.append(epe(flo_ch, eflo))
            
            prev_name = name
            prev_image = image

    return np.array(epes).mean(), np.median(np.array(epes))

def epe_sintel_unopflow(net, base_sintel='mpi_sintel/training', device='cuda'):
    folders = sorted(os.listdir(f'{base_sintel}/flow'))
    epes = []
    for folder in folders:
        # Get image names
        # im_names = sorted(os.listdir(folder))
        im_names = sorted([name[:-4] for name in os.listdir(osp.join(base_sintel, 'final', folder))])
        prev_name = None
        for name in im_names:
            image = plt.imread(osp.join(base_sintel, 'final', folder, name + '.png'))
            if prev_name is None:
                prev_name = name
                prev_image = image
                continue

            # Read flow
            flo = read_flo_file(osp.join(base_sintel, 'flow', folder, prev_name + '.flo'))
            flo_ch = np.transpose(flo, (2, 0, 1))
            bflo = flo_ch.reshape(1, *flo_ch.shape)

            # Estimate flow
            im1_trans = np.transpose(prev_image, (2, 0, 1))
            im2_trans = np.transpose(image, (2, 0, 1))

            im1_trans = np.expand_dims(im1_trans, axis=0)
            im2_trans = np.expand_dims(im2_trans, axis=0)
            
            im1_torch = torch.from_numpy(im1_trans).to(device)
            im2_torch = torch.from_numpy(im2_trans).to(device)
            
            eflo = net.inference_flow(im1_torch, im2_torch)
            eflo = one_scale(torch.from_numpy(bflo), eflo).cpu().detach().numpy()
            eflo = eflo.squeeze()
            epev = epe(flo_ch, eflo)
            
            epes.append(epev)
            prev_name = name
            prev_image = image
            

    return np.array(epes).mean(), np.median(np.array(epes))

def plot_quiver(ax, flow, spacing, margin=0, **kwargs):
    """Plots less dense quiver field.

    Args:
        ax: Matplotlib axis
        flow: motion vectors
        spacing: space (px) between each arrow in grid
        margin: width (px) of enclosing region without arrows
        kwargs: quiver kwargs (default: angles="xy", scale_units="xy")
    """
    h, w, *_ = flow.shape

    nx = int((w - 2 * margin) / spacing)
    ny = int((h - 2 * margin) / spacing)

    x = np.linspace(margin, w - margin - 1, nx, dtype=np.int64)
    y = np.linspace(margin, h - margin - 1, ny, dtype=np.int64)

    flow = flow[np.ix_(y, x)]
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    kwargs = {**dict(angles="xy", scale_units="xy"), **kwargs}
    ax.quiver(x, y, u, v, **kwargs)

    ax.set_ylim(sorted(ax.get_ylim(), reverse=True))
    ax.set_aspect("equal")
    
def one_scale(output, target):
    # Got this from FlowNet
    b, _, h, w = output.size()

    target_scaled = F.interpolate(target, (h, w), mode='area')
    return target_scaled

def splitted_im(images):
    # flip images order
    images1 = images[:, :3]
    images2 = images[:, 3:]
    flipped_images = torch.cat([images2, images1], dim=1)

    return images1, images2, flipped_images

def iterate_dataloaders(loaders, iters=[2, 6, 10, 15]):
    while True:

        loaders_prop = []
        for k, loader in zip(iters, loaders):
            loaders_prop.extend(k * [loader])

        has_next = True
        while has_next:
            for gen in loaders_prop:
                try:
                    yield next(iter(gen))

                except Exception:
                    has_next = False
