# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

import logging
import torch
from torch import nn
import torch.nn.functional as F
import math
from .until_config import PretrainedConfig

logger = logging.getLogger(__name__)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


##################################
###### LOSS FUNCTION #############
##################################
class CrossEn(nn.Module):
    def __init__(self, config=None):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss


class ArcCrossEn(nn.Module):
    def __init__(self, margin=10):
        super(ArcCrossEn, self).__init__()
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

    def forward(self, sim_matrix, scale):
        cos = torch.diag(sim_matrix)
        sin = torch.sqrt(1.0 - torch.pow(cos, 2))
        pin = cos * self.cos_m - sin * self.sin_m
        sim_matrix = sim_matrix - torch.diag_embed(cos) + torch.diag_embed(pin)
        logpt = F.log_softmax(sim_matrix / scale, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss


class CrossEn0(nn.Module):
    def __init__(self, config=None):
        super(CrossEn0, self).__init__()

    def forward(self, sim_matrix, b):
        logpt = F.log_softmax(sim_matrix[:b, :], dim=-1)
        logpt = torch.diag(logpt[:, :b])
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss


class ema_CrossEn(nn.Module):
    def __init__(self, config=None):
        super(ema_CrossEn, self).__init__()

    def forward(self, sim_matrix0, sim_matrix1):
        m, n = sim_matrix0.size()
        diag1 = torch.diag(sim_matrix1)
        diag1 = torch.diag_embed(diag1)
        sim_matrix1 = sim_matrix1 - diag1
        logpt = F.log_softmax(torch.cat([sim_matrix0, sim_matrix1], dim=-1), dim=-1)
        logpt = torch.diag(logpt[:, :n])
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss


class DC_CrossEn(nn.Module):
    def __init__(self, config=None):
        super(DC_CrossEn, self).__init__()

    def forward(self, sim_matrix0, sim_matrix1, seta=0.8):
        diag0 = torch.diag(sim_matrix0)
        diag1 = torch.diag(sim_matrix1)
        sim_matrix0 = sim_matrix0 - diag0
        sim_matrix1 = sim_matrix1 - diag1
        m, n = sim_matrix0.size()

        sim_matrix = torch.where(sim_matrix1 < seta, sim_matrix0, torch.tensor(0.0).to(sim_matrix0.device))
        sim_matrix = sim_matrix + diag0

        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss


class ema_CrossEn1(nn.Module):
    def __init__(self, config=None):
        super(ema_CrossEn1, self).__init__()

    def forward(self, sim_matrix0, sim_matrix1):
        logpt0 = F.log_softmax(sim_matrix0, dim=-1)
        logpt1 = F.softmax(sim_matrix1, dim=-1)
        sim_loss = - logpt0 * logpt1
        # diag = torch.diag(sim_loss)
        # sim_loss = sim_loss - diag
        sim_loss = sim_loss.mean()
        return sim_loss


class ema_CrossEn2(nn.Module):
    def __init__(self, config=None):
        super(ema_CrossEn2, self).__init__()

    def forward(self, sim_matrix0, sim_matrix1, lambd=0.5):
        m, n = sim_matrix1.size()

        logpt0 = F.log_softmax(sim_matrix0, dim=-1)
        logpt1 = F.softmax(sim_matrix1, dim=-1)
        logpt1 = lambd * torch.eye(m).to(logpt1.device) + (1 - lambd) * logpt1

        sim_loss = - logpt0 * logpt1
        sim_loss = sim_loss.sum() / m
        return sim_loss


class KL(nn.Module):
    def __init__(self, config=None):
        super(KL, self).__init__()

    def forward(self, sim_matrix0, sim_matrix1):
        logpt0 = F.log_softmax(sim_matrix0, dim=-1)
        logpt1 = F.softmax(sim_matrix1, dim=-1)
        kl = F.kl_div(logpt0, logpt1, reduction='mean')
        return kl


def _batch_hard(mat_distance, mat_similarity, indice=False):
    sorted_mat_distance, positive_indices = torch.sort(mat_distance + (9999999.) * (1 - mat_similarity), dim=1,
                                                       descending=False)
    hard_p = sorted_mat_distance[:, 0]
    hard_p_indice = positive_indices[:, 0]
    sorted_mat_distance, negative_indices = torch.sort(mat_distance + (-9999999.) * (mat_similarity), dim=1,
                                                       descending=True)
    hard_n = sorted_mat_distance[:, 0]
    hard_n_indice = negative_indices[:, 0]
    if (indice):
        return hard_p, hard_n, hard_p_indice, hard_n_indice
    return hard_p, hard_n


class SoftTripletLoss(nn.Module):
    def __init__(self, config=None):
        super(SoftTripletLoss, self).__init__()

    def forward(self, sim_matrix0, sim_matrix1):
        N = sim_matrix0.size(0)
        mat_sim = torch.eye(N).float().to(sim_matrix0.device)
        dist_ap, dist_an, ap_idx, an_idx = _batch_hard(sim_matrix0, mat_sim, indice=True)
        triple_dist = torch.stack((dist_ap, dist_an), dim=1)
        triple_dist = F.log_softmax(triple_dist, dim=1)
        dist_ap_ref = torch.gather(sim_matrix1, 1, ap_idx.view(N, 1).expand(N, N))[:, 0]
        dist_an_ref = torch.gather(sim_matrix1, 1, an_idx.view(N, 1).expand(N, N))[:, 0]
        triple_dist_ref = torch.stack((dist_ap_ref, dist_an_ref), dim=1)
        triple_dist_ref = F.softmax(triple_dist_ref, dim=1).detach()
        loss = (- triple_dist_ref * triple_dist).mean(0).sum()
        return loss


class MSE(nn.Module):
    def __init__(self, config=None):
        super(MSE, self).__init__()

    def forward(self, sim_matrix0, sim_matrix1):
        logpt = (sim_matrix0 - sim_matrix1)
        loss = logpt * logpt
        return loss.mean()


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def uniformity_loss(x, y):
    input = torch.cat((x, y), dim=0)
    m = input.size(0)
    dist = euclidean_dist(input, input)
    return torch.logsumexp(torch.logsumexp(dist, dim=-1), dim=-1) - torch.log(torch.tensor(m * m - m))


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, args):
        if args.world_size == 1:
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return tensor
        else:
            output = [torch.empty_like(tensor) for _ in range(args.world_size)]
            torch.distributed.all_gather(output, tensor)
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
        )


class AllGather2(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    # https://github.com/PyTorchLightning/lightning-bolts/blob/8d3fbf7782e3d3937ab8a1775a7092d7567f2933/pl_bolts/models/self_supervised/simclr/simclr_module.py#L20
    @staticmethod
    def forward(ctx, tensor, args):
        if args.world_size == 1:
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return tensor
        else:
            output = [torch.empty_like(tensor) for _ in range(args.world_size)]
            torch.distributed.all_gather(output, tensor)
            ctx.rank = args.local_rank
            ctx.batch_size = tensor.shape[0]
            return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)
        return (grad_input[ctx.rank * ctx.batch_size:(ctx.rank + 1) * ctx.batch_size], None)


class PreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def resize_token_embeddings(self, new_num_tokens=None):
        raise NotImplementedError

    @classmethod
    def init_preweight(cls, model, state_dict, prefix=None, task_config=None):
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        if prefix is not None:
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                old_keys.append(key)
                new_keys.append(prefix + key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='')

        if prefix is None and (task_config is None or task_config.local_rank == 0):
            logger.info("-" * 20)
            if len(missing_keys) > 0:
                logger.info("Weights of {} not initialized from pretrained model: {}"
                            .format(model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)))
            if len(unexpected_keys) > 0:
                logger.info("Weights from pretrained model not used in {}: {}"
                            .format(model.__class__.__name__, "\n   " + "\n   ".join(unexpected_keys)))
            if len(error_msgs) > 0:
                logger.error("Weights from pretrained model cause errors in {}: {}"
                             .format(model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)))

        return model

    @property
    def dtype(self):
        """
        :obj:`torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            # For nn.DataParallel compatibility in PyTorch 1.5
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    @classmethod
    def from_pretrained(cls, config, state_dict=None,  *inputs, **kwargs):
        """
        Instantiate a PreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        """
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            return model
        model = cls.init_preweight(model, state_dict)

        return model