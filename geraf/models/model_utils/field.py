# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import torch
import torch.nn as nn
import numpy as np

class IdentityActivation(nn.Module):
    def forward(self, x): return x

class ExpActivation(nn.Module):
    def __init__(self, max_light=None):
        super().__init__()
        self.max_light=max_light

    def forward(self, x):
        if self.max_light is not None:
            return torch.exp(torch.clamp(x, max=self.max_light))
        else:
            return torch.exp(x)

def make_predictor(feats_dim: object, output_dim: object, weight_norm: object = True, activation='sigmoid', exp_max=None) -> object:
    if activation == 'sigmoid':
        activation = nn.Sigmoid()
    elif activation == 'softplus':
        activation = nn.Softplus()
    elif activation=='exp':
        activation = ExpActivation(max_light=exp_max)
    elif activation=='none':
        activation = IdentityActivation()
    elif activation=='relu':
        activation = nn.ReLU()
    else:
        raise NotImplementedError

    run_dim = 256
    if weight_norm:
        module=nn.Sequential(
            nn.utils.weight_norm(nn.Linear(feats_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, run_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(run_dim, output_dim)),
            activation,
        )
    else:
        module=nn.Sequential(
            nn.Linear(feats_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, run_dim),
            nn.ReLU(),
            nn.Linear(run_dim, output_dim),
            activation,
        )

    return module



def get_camera_plane_intersection(pts, dirs, poses):
    """
    compute the intersection between the rays and the camera XoY plane
    :param pts:      pn,3
    :param dirs:     pn,3
    :param poses:    pn,3,4
    :return:
    """
    R, t = poses[:,:,:3], poses[:,:,3:]

    # transfer into human coordinate
    pts_ = (R @ pts[:,:,None] + t)[..., 0] # pn,3
    dirs_ = (R @ dirs[:,:,None])[..., 0]   # pn,3

    hits = torch.abs(dirs_[..., 2]) > 1e-4
    dirs_z = dirs_[:, 2]
    dirs_z[~hits] = 1e-4
    dist = -pts_[:, 2] / dirs_z
    inter = pts_ + dist.unsqueeze(-1) * dirs_
    return inter, dist, hits

def expected_sin(mean, var):
  """Compute the mean of sin(x), x ~ N(mean, var)."""
  return torch.exp(-0.5 * var) * torch.sin(mean)  # large var -> small value.

def IPE(mean,var,min_deg,max_deg):
    scales = 2**torch.arange(min_deg, max_deg)
    shape = mean.shape[:-1] + (-1,)
    scaled_mean = torch.reshape(mean[..., None, :] * scales[:, None], shape)
    scaled_var = torch.reshape(var[..., None, :] * scales[:, None]**2, shape)
    return expected_sin(torch.concat([scaled_mean, scaled_mean + 0.5 * np.pi], dim=-1), torch.concat([scaled_var] * 2, dim=-1))

def offset_points_to_sphere(points):
    points_norm = torch.norm(points, dim=-1)
    mask = points_norm > 0.999
    if torch.sum(mask)>0:
        points = torch.clone(points)
        points[mask] /= points_norm[mask].unsqueeze(-1)
        points[mask] *= 0.999
        # points[points_norm>0.999] = 0
    return points

def get_sphere_intersection(pts, dirs):
    dtx = torch.sum(pts*dirs,dim=-1,keepdim=True) # rn,1
    xtx = torch.sum(pts**2,dim=-1,keepdim=True) # rn,1
    dist = dtx ** 2 - xtx + 1
    assert torch.sum(dist<0)==0
    dist = -dtx + torch.sqrt(dist+1e-6) # rn,1
    return dist

# this function is borrowed from NeuS
def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def get_weights(sdf_fun, inv_fun, z_vals, origins, dirs):
    points = z_vals.unsqueeze(-1) * dirs.unsqueeze(-2) + origins.unsqueeze(-2) # pn,sn,3
    inv_s = inv_fun(points[:, :-1, :])[..., 0]  # pn,sn-1
    sdf = sdf_fun(points)[..., 0]  # pn,sn

    prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]  # pn,sn-1
    prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
    mid_sdf = (prev_sdf + next_sdf) * 0.5
    cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)  # pn,sn-1
    surface_mask = (cos_val < 0)  # pn,sn-1
    cos_val = torch.clamp(cos_val, max=0)

    dist = next_z_vals - prev_z_vals  # pn,sn-1
    prev_esti_sdf = mid_sdf - cos_val * dist * 0.5  # pn, sn-1
    next_esti_sdf = mid_sdf + cos_val * dist * 0.5
    prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
    next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
    alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5) * surface_mask.float()
    weights = alpha * torch.cumprod(torch.cat([torch.ones([alpha.shape[0], 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
    mid_sdf[~surface_mask]=-1.0
    return weights, mid_sdf

def get_intersection(sdf_fun, inv_fun, pts, dirs, sn0=128, sn1=9):
    """
    :param sdf_fun:
    :param inv_fun:
    :param pts:    pn,3
    :param dirs:   pn,3
    :param sn0:
    :param sn1:
    :return:
    """
    # inside_mask = torch.norm(pts, dim=-1) < 0.999 # left some margin
    max_dist = get_sphere_intersection(pts, dirs) # pn,1
    with torch.no_grad():
        z_vals = torch.linspace(0, 1, sn0) # sn0
        z_vals = max_dist * z_vals.unsqueeze(0) # pn,sn0
        weights, mid_sdf = get_weights(sdf_fun, inv_fun, z_vals, pts, dirs) # pn,sn0-1
        z_vals_new = sample_pdf(z_vals, weights, sn1, True) # pn,sn1
        weights, mid_sdf = get_weights(sdf_fun, inv_fun, z_vals_new, pts, dirs) # pn,sn1-1
        z_vals_mid = (z_vals_new[:,1:] + z_vals_new[:,:-1]) * 0.5

    return z_vals_mid, weights, mid_sdf

def offset_points_to_sphere(points):
    points_norm = torch.norm(points, dim=-1)
    mask = points_norm > 0.999
    if torch.sum(mask)>0:
        points = torch.clone(points)
        points[mask] /= points_norm[mask].unsqueeze(-1)
        points[mask] *= 0.999
        # points[points_norm>0.999] = 0
    return points

# this function is borrowed from NeuS
def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples]).to(weights)
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def sample_pdf_np(bins, weights, n_samples, det=False):
    """
    NumPy version of sample_pdf. Samples `n_samples` values given `bins` and `weights`.
    bins: (batch, n_bins), weights: (batch, n_bins)
    Returns: (batch, n_samples)
    """
    weights = weights + 1e-5  # avoid zero-division
    pdf = weights / np.sum(weights, axis=-1, keepdims=True)
    cdf = np.cumsum(pdf, axis=-1)
    cdf = np.concatenate([np.zeros_like(cdf[..., :1]), cdf], axis=-1)

    batch_size, n_bins = bins.shape
    if det:
        u = np.linspace(0.5 / n_samples, 1. - 0.5 / n_samples, n_samples)
        u = np.broadcast_to(u, (batch_size, n_samples))
    else:
        u = np.random.rand(batch_size, n_samples)

    # Invert CDF using searchsorted
    inds = np.zeros_like(u, dtype=np.int64)
    for i in range(batch_size):
        inds[i] = np.searchsorted(cdf[i], u[i], side='right')
    below = np.maximum(0, inds - 1)
    above = np.minimum(cdf.shape[-1] - 1, inds)

    # Gather values
    cdf_g0 = np.take_along_axis(cdf, below, axis=1)
    cdf_g1 = np.take_along_axis(cdf, above, axis=1)
    bins_g0 = np.take_along_axis(bins, below, axis=1)
    bins_g1 = np.take_along_axis(bins, above, axis=1)

    denom = cdf_g1 - cdf_g0
    denom = np.where(denom < 1e-5, 1.0, denom)
    t = (u - cdf_g0) / denom
    samples = bins_g0 + t * (bins_g1 - bins_g0)

    return samples
