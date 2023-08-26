import os
from collections import OrderedDict
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from .module_clip import CLIP, convert_weights, _PT_NAME
from .module_cross import CrossModel, Transformer as TransformerClip
from .until_module import LayerNorm, AllGather, AllGather2, CrossEn, MSE, ArcCrossEn, KL
import numpy as np
from diffusion_models import gaussian_diffusion as gd
from diffusion_models.respace import SpacedDiffusion, space_timesteps
from .diffusion import Diffusion, Diffusion_v

allgather = AllGather.apply
allgather2 = AllGather2.apply


class ResidualLinear(nn.Module):
    def __init__(self, d_int: int):
        super(ResidualLinear, self).__init__()

        self.fc_relu = nn.Sequential(nn.Linear(d_int, d_int),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = x + self.fc_relu(x)
        return x


class DiffusionRet(nn.Module):
    def __init__(self, config):
        super(DiffusionRet, self).__init__()

        self.config = config
        self.interaction = config.interaction
        self.agg_module = getattr(config, 'agg_module', 'meanP')
        self.stage = getattr(config, 'stage', 'discrimination')
        backbone = getattr(config, 'base_encoder', "ViT-B/32")

        assert backbone in _PT_NAME
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[backbone])
        if os.path.exists(model_path):
            FileNotFoundError
        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        self.clip = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)

        if torch.cuda.is_available():
            convert_weights(self.clip)  # fp16

        cross_config = SimpleNamespace(**{
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 2048,
            "max_position_embeddings": 128,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "vocab_size": 512,
            "soft_t": 0.07,
        })
        cross_config.max_position_embeddings = context_length
        cross_config.hidden_size = transformer_width
        self.cross_config = cross_config
        if self.interaction == 'wti':
            self.text_weight_fc = nn.Sequential(
                    nn.Linear(transformer_width, 2*transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(2*transformer_width, 1))
            self.video_weight_fc = nn.Sequential(
                    nn.Linear(transformer_width, 2*transformer_width), nn.ReLU(inplace=True),
                    nn.Linear(2*transformer_width, 1))

        if self.agg_module in ["seqLSTM", "seqTransf"]:
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings,
                                                          cross_config.hidden_size)
            if self.agg_module == "seqTransf":
                self.transformerClip = TransformerClip(width=transformer_width,
                                                       layers=config.num_hidden_layers,
                                                       heads=transformer_heads)
            if self.agg_module == "seqLSTM":
                self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                           batch_first=True, bidirectional=False, num_layers=1)

        self.loss_fct = CrossEn(config)
        self.loss_arcfct = ArcCrossEn(margin=10)

        if self.stage == "generation":
            self.diffusion_model = Diffusion(cross_config.hidden_size, 0.1, self.config.temp)
            self.diffusion_model_v = Diffusion_v(cross_config.hidden_size, 0.1, self.config.temp)

            for param in self.clip.parameters():
                param.requires_grad = False  # not update by gradient
            for param in self.transformerClip.parameters():
                param.requires_grad = False  # not update by gradient
            for param in self.frame_position_embeddings.parameters():
                param.requires_grad = False  # not update by gradient

            for param in self.text_weight_fc.parameters():
                param.requires_grad = False  # not update by gradient
            for param in self.video_weight_fc.parameters():
                param.requires_grad = False  # not update by gradient

        self.apply(self.init_weights)  # random init must before loading pretrain
        self.clip.load_state_dict(state_dict, strict=False)

        ## ===> Initialization trick [HARD CODE]
        new_state_dict = OrderedDict()
                
        if self.agg_module in ["seqLSTM", "seqTransf"]:
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in state_dict.items():
                    if key == "positional_embedding":
                        new_state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if self.agg_module in ["seqTransf"] and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < config.num_hidden_layers:
                            new_state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue

        self.load_state_dict(new_state_dict, strict=False)  # only update new state (seqTransf/seqLSTM/tightTransf)
        ## <=== End of initialization trick

    def forward(self, text_ids, text_mask, video, video_mask=None, idx=None, global_step=0,
                schedule_sampler=None, diffusion=None):
        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        # B x N_v x 3 x H x W - >  (B x N_v) x 3 x H x W
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        text_feat, video_feat, cls = self.get_text_video_feat(text_ids, text_mask, video, video_mask, shaped=True)

        if self.training:
            if torch.cuda.is_available():  # batch merge here
                idx = allgather(idx, self.config)
                text_feat = allgather(text_feat, self.config)
                video_feat = allgather(video_feat, self.config)
                text_mask = allgather(text_mask, self.config)
                video_mask = allgather(video_mask, self.config)
                cls = allgather(cls, self.config)
                torch.distributed.barrier()  # force sync

            idx = idx.view(-1, 1)
            idx_all = idx.t()
            pos_idx = torch.eq(idx, idx_all).float()
            sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
            logit_scale = self.clip.logit_scale.exp()
            loss, discrimination_loss, generation_loss = 0., 0., 0.

            t2v_logits, v2t_logits = self.get_similarity_logits(text_feat, cls, video_feat,
                                                                    text_mask, video_mask, shaped=True)

            discrimination_loss_t2v = self.loss_fct(t2v_logits * logit_scale)
            discrimination_loss_v2t = self.loss_fct(v2t_logits * logit_scale)
            discrimination_loss += (discrimination_loss_t2v + discrimination_loss_v2t) / 2
            loss += discrimination_loss

            if self.stage == "discrimination":
                return loss, discrimination_loss, torch.zeros_like(discrimination_loss)

            t, weights = schedule_sampler.sample(cls.shape[0], text_feat.device)
            num = self.config.num

            cls, video_feat = cls.detach(), video_feat.detach()
            t2v_logits, v2t_logits = t2v_logits.detach(), v2t_logits.detach()
            a, b = cls.size(0), video_feat.size(1)

            mask = torch.eq(idx, idx.t())
            weights_t2v = F.softmax(t2v_logits, dim=1)
            weights_v2t = F.softmax(v2t_logits, dim=1)
            weights_t2v.masked_fill_(mask, -1)
            weights_v2t.masked_fill_(mask, -1)


            video_embeds_neg = []
            video_mask_neg = []
            for b in range(weights_t2v.size(0)):
                _, neg_idx = weights_t2v[b].topk(num, largest=True, sorted=True)
                temp = [video_feat[b, :, :]]
                temp_ = [video_mask[b, :]]
                for i in neg_idx:
                    temp.append(video_feat[i, :, :])
                    temp_.append(video_mask[i, :])
                video_embeds_neg.append(torch.stack(temp, dim=0))
                video_mask_neg.append(torch.stack(temp_, dim=0))
            video_embeds_neg = torch.stack(video_embeds_neg, dim=0)
            video_mask_neg = torch.stack(video_mask_neg, dim=0)

            text_embeds_neg = []
            for b in range(weights_v2t.size(0)):
                _, neg_idx = weights_v2t[b].topk(num, largest=True, sorted=True)
                temp = [cls[b]]
                for i in neg_idx:
                    temp.append(cls[i])
                text_embeds_neg.append(torch.stack(temp, dim=0))
            text_embeds_neg = torch.stack(text_embeds_neg, dim=0)

            pos = torch.ones((cls.size(0), 1), dtype=torch.float)
            if self.config.neg == 0:
                neg = torch.zeros((cls.size(0), num), dtype=torch.float)
            else:
                neg = -torch.ones((cls.size(0), num), dtype=torch.float)
            micro = torch.cat([pos, neg], dim=1).to(cls.device)

            output = diffusion.training_losses(self.diffusion_model, micro, t, {"text_emb": cls,
                                                                                "video_emb": video_embeds_neg,
                                                                                "video_mask": video_mask_neg},
                                               temp=self.config.d_temp)
            generation_loss += output["kl_loss"]

            output = diffusion.training_losses(self.diffusion_model_v, micro, t, {"text_emb": text_embeds_neg,
                                                                                  "video_emb": video_feat,
                                                                                  "video_mask": video_mask},
                                               temp=self.config.d_temp)
            generation_loss += output["kl_loss"]

            loss += generation_loss
            return loss, discrimination_loss, generation_loss

    def get_text_feat(self, text_ids, text_mask, shaped=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])

        bs_pair = text_ids.size(0)
        cls, text_feat = self.clip.encode_text(text_ids, return_hidden=True, mask=text_mask)
        cls, text_feat = cls.float(), text_feat.float()
        text_feat = text_feat.view(bs_pair, -1, text_feat.size(-1))
        cls = cls.view(bs_pair, -1, cls.size(-1)).squeeze(1)
        return text_feat, cls

    def get_video_feat(self, video, video_mask, shaped=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.size()) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)

        bs_pair, n_v = video_mask.size()
        video_feat = self.clip.encode_image(video, return_hidden=True)[0].float()
        video_feat = video_feat.float().view(bs_pair, -1, video_feat.size(-1))
        video_feat = self.agg_video_feat(video_feat, video_mask, self.agg_module)
        return video_feat

    def get_text_video_feat(self, text_ids, text_mask, video, video_mask, shaped=False):
        if shaped is False:
            text_ids = text_ids.view(-1, text_ids.shape[-1])
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.shape) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)

        text_feat, cls = self.get_text_feat(text_ids, text_mask, shaped=True)
        video_feat = self.get_video_feat(video, video_mask, shaped=True)

        return text_feat, video_feat, cls

    def get_video_avg_feat(self, video_feat, video_mask):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        video_feat = video_feat * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_feat = torch.sum(video_feat, dim=1) / video_mask_un_sum
        return video_feat

    def get_text_sep_feat(self, text_feat, text_mask):
        text_feat = text_feat.contiguous()
        text_feat = text_feat[torch.arange(text_feat.shape[0]), torch.sum(text_mask, dim=-1) - 1, :]
        text_feat = text_feat.unsqueeze(1).contiguous()
        return text_feat

    def agg_video_feat(self, video_feat, video_mask, agg_module):
        video_feat = video_feat.contiguous()
        if agg_module == "None":
            pass
        elif agg_module == "seqLSTM":
            # Sequential type: LSTM
            video_feat_original = video_feat
            video_feat = pack_padded_sequence(video_feat, torch.sum(video_mask, dim=-1).cpu(),
                                              batch_first=True, enforce_sorted=False)
            video_feat, _ = self.lstm_visual(video_feat)
            if self.training: self.lstm_visual.flatten_parameters()
            video_feat, _ = pad_packed_sequence(video_feat, batch_first=True)
            video_feat = torch.cat(
                (video_feat, video_feat_original[:, video_feat.size(1):, ...].contiguous()), dim=1)
            video_feat = video_feat + video_feat_original
        elif agg_module == "seqTransf":
            # Sequential type: Transformer Encoder
            video_feat_original = video_feat
            seq_length = video_feat.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=video_feat.device)
            position_ids = position_ids.unsqueeze(0).expand(video_feat.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            video_feat = video_feat + frame_position_embeddings
            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            video_feat = video_feat.permute(1, 0, 2)  # NLD -> LND
            video_feat = self.transformerClip(video_feat, extended_video_mask)
            video_feat = video_feat.permute(1, 0, 2)  # LND -> NLD
            video_feat = video_feat + video_feat_original
        return video_feat

    def get_similarity_logits(self, text_feat, cls, video_feat, text_mask, video_mask, shaped=False):
        if shaped is False:
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        text_weight = self.text_weight_fc(text_feat).squeeze(2)  # B x N_t x D -> B x N_t
        text_weight.masked_fill_((1 - text_mask).to(torch.bool), float(-9e15))
        text_weight = torch.softmax(text_weight, dim=-1)  # B x N_t

        video_weight = self.video_weight_fc(video_feat).squeeze(2)  # B x N_v x D -> B x N_v
        video_weight.masked_fill_((1 - video_mask).to(torch.bool), float(-9e15))
        video_weight = torch.softmax(video_weight, dim=-1)  # B x N_v

        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)

        retrieve_logits = torch.einsum('atd,bvd->abtv', [text_feat, video_feat])
        retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, text_mask])
        retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, video_mask])

        text_sum = text_mask.sum(-1)
        video_sum = video_mask.sum(-1)

        t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
        t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_weight])

        v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
        v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, video_weight])

        retrieve_logits = (t2v_logits + v2t_logits) / 2.0

        return retrieve_logits, retrieve_logits.T

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

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


def create_gaussian_diffusion(args):
    # default params
    steps = args.diffusion_steps
    learn_sigma = False
    sigma_small = False
    predict_xstart = True
    rescale_timesteps = False
    rescale_learned_sigmas = False
    timestep_respacing = ""
    scale_beta = 1.

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps)

    loss_type = gd.LossType.KL

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps
    )