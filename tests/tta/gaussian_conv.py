from torch import Tensor
import torch
import numpy as np

def gaussian_to_rbbox(
        xyabc: Tensor,
        scalar_div: float):
    xyabc = xyabc.clone()
    a,b,c = xyabc[..., 2:].unbind(-1)
    w,h,t = get_rbbox_shape_from_cov(a,b,c, scalar_div=scalar_div)
    wht = torch.stack((w,h,t), dim=-1)
    xyabc[..., 2:] = wht

    return xyabc

def rbbox_to_gaussian(
        xywht: Tensor,
        scalar: float):
    xywht = xywht.clone()
    w,h,t = xywht[..., 2:].unbind(-1)
    a,b,c = get_cov_elements_from_wht(w,h,t, scalar=scalar)
    abc = torch.stack((a,b,c), dim=-1)
    xywht[..., 2:] = abc

    return xywht

def get_rbbox_shape_from_cov(
        cov_a : Tensor, 
        cov_b : Tensor, 
        cov_c : Tensor,
        scalar_div: float):
    cov = torch.stack((cov_a, cov_c, cov_c, cov_b), dim=-1).reshape(-1, 2, 2)
    L, Q = torch.linalg.eigh(cov)
    eig_min, eig_max = L.unbind(-1)
    eig_max_vec_x, eig_max_vec_y = Q[..., 1].unbind(-1)

    gw = scalar_div * torch.sqrt(eig_max)
    gh = scalar_div * torch.sqrt(eig_min)
    gt = torch.atan2(eig_max_vec_y, eig_max_vec_x)

    return gw, gh, gt

def get_cov_elements_from_wht(
        box_w : Tensor,
        box_h : Tensor,
        box_t : Tensor,
        scalar: float):
    sq_sin_t = torch.sin(box_t).square()
    sq_cos_t = torch.cos(box_t).square()
    eig_w = (scalar * box_w).square()
    eig_h = (scalar * box_h).square()
    cov_a = sq_cos_t * eig_w + sq_sin_t * eig_h
    cov_b = sq_cos_t * eig_h + sq_sin_t * eig_w
    cov_c = 0.5 * (eig_w - eig_h) * torch.sin(2 * box_t)

    return cov_a, cov_b, cov_c