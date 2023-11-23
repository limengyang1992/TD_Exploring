import math

import torch
import torch.nn.functional as F
""" Layer/Module Helpers
Hacked together by / Copyright 2020 Ross Wightman
"""
import collections.abc
from itertools import repeat


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def resize_pos_embed(posemb, posemb_new, num_tokens=1):
    # Copied from `timm` by Ross Wightman:
    # github.com/rwightman/pytorch-image-models
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0,
                                                                 num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old,
                                      -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid,
                                size=(gs_new, gs_new),
                                mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3,
                                      1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def pe_check(model, state_dict, pe_key='classifier.positional_emb'):
    if (pe_key is not None and pe_key in state_dict.keys()
            and pe_key in model.state_dict().keys()):
        if model.state_dict()[pe_key].shape != state_dict[pe_key].shape:
            state_dict[pe_key] = resize_pos_embed(
                state_dict[pe_key],
                model.state_dict()[pe_key],
                num_tokens=model.classifier.num_tokens,
            )
    return state_dict