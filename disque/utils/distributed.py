import torch
import torch.distributed as dist
import lightly.utils.dist as ldist

def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def gather(x, cat_dim=0):
    return torch.cat(ldist.gather(x), cat_dim) if is_distributed() else x

def rank():
    return ldist.rank()


@torch.no_grad()
def batch_shuffle(x):
    """
    Batch shuffle, for making use of BatchNorm.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    batch_size_this = x.shape[0]
    x_all = dist.gather(x)
    batch_size_all = x_all.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # random shuffle index
    idx_shuffle = torch.randperm(batch_size_all).cuda()

    # broadcast to all gpus
    torch.distributed.broadcast(idx_shuffle, src=0)

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    # shuffled index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    return x_all[idx_this], idx_unshuffle

@torch.no_grad()
def batch_unshuffle(x, idx_unshuffle):
    """
    Undo batch shuffle.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = dist.gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # restored index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this]
