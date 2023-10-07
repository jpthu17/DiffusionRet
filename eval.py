from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import datetime
from os.path import join, exists
from diffusion_models.resample import create_named_schedule_sampler
import torch
import torch.nn.functional as F
from DiffusionRet.models.tokenization_clip import SimpleTokenizer as ClipTokenizer
from DiffusionRet.dataloaders.data_dataloaders import DATALOADER_DICT
from DiffusionRet.dataloaders.dataloader_msrvtt_retrieval import MSRVTTDataset
from DiffusionRet.models.modeling import DiffusionRet, AllGather, create_gaussian_diffusion
from DiffusionRet.models.optimization import BertAdam
from DiffusionRet.utils.metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim

from DiffusionRet.utils.comm import is_main_process, synchronize
from DiffusionRet.utils.logger import setup_logger
from DiffusionRet.utils.metric_logger import MetricLogger
import copy
allgather = AllGather.apply

global logger


def get_args(
        description='DiffusionRet: Generative Text-Video Retrieval with Diffusion Model'):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")
    parser.add_argument('--anno_path', type=str, default='data/MSR-VTT/anns', help='annotation path')
    parser.add_argument('--video_path', type=str, default='data/MSR-VTT/videos', help='video path')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')

    parser.add_argument('--epochs', type=int, default=5, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=128, help='batch size eval')

    parser.add_argument('--max_words', type=int, default=32, help='max text token number')
    parser.add_argument('--max_frames', type=int, default=12, help='max key frames')
    parser.add_argument('--video_framerate', type=int, default=1, help='framerate to sample video frame')

    parser.add_argument("--device", default='cpu', type=str, help="cpu/cuda")
    parser.add_argument("--world_size", default=1, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--distributed", default=0, type=int, help="multi machine DDP")

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--base_encoder", default="ViT-B/32", type=str, help="Choose a CLIP version")
    parser.add_argument('--agg_module', type=str, default="seqTransf", choices=["None", "seqLSTM", "seqTransf"],
                        help="choice a feature aggregation module for video.")
    parser.add_argument('--interaction', type=str, default='wti', help="interaction type for retrieval.")
    parser.add_argument('--num_hidden_layers', type=int, default=4)

    parser.add_argument("--stage", default='generation', choices=['discrimination', 'generation'], type=str)
    
    parser.add_argument('--temp', type=float, default=1)
    parser.add_argument('--d_temp', type=float, default=100)

    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")

    parser.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                        help="Noise schedule type")
    parser.add_argument("--diffusion_steps", default=50, type=int,
                        help="Number of diffusion steps (denoted T in the paper)")
    parser.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")
    parser.add_argument("--neg", default=0, type=int, choices=[-1, 0])
    parser.add_argument("--num", default=127, type=int)
    parser.add_argument("--t2v_num", default=32, type=int)
    parser.add_argument("--v2t_num", default=32, type=int)
    parser.add_argument("--t2v_temp", default=1, type=float)
    parser.add_argument("--v2t_temp", default=1, type=float)
    parser.add_argument("--t2v_alpha", default=1, type=float)
    parser.add_argument("--v2t_alpha", default=1, type=float)

    args = parser.parse_args()

    return args


def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        args.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if torch.cuda.is_available():
        torch.distributed.barrier()
    logger.info("local_rank: {} world_size: {}".format(args.local_rank, args.world_size))

    if args.batch_size % args.world_size != 0 or args.batch_size_val % args.world_size != 0:
        raise ValueError(
            "Invalid batch_size/batch_size_val and world_size parameter: {}%{} and {}%{}, should be == 0".format(
                args.batch_size, args.world_size, args.batch_size_val, args.world_size))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args


def build_model(args):
    model = DiffusionRet(args)
    if args.init_model:
        if not exists(args.init_model):
            raise FileNotFoundError
        model_state_dict = torch.load(args.init_model, map_location='cpu')
        model.load_state_dict(model_state_dict, strict=False)

    model.to(args.device)
    return model


def build_dataloader(args):
    ## ####################################
    # dataloader loading
    ## ####################################
    tokenizer = ClipTokenizer()
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    if isinstance(test_length, int):
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)
    elif len(test_length) == 2:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %dt %dv", test_length[0], test_length[1])
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d %d", len(test_dataloader[0]), len(test_dataloader[1]))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %dt %dv", val_length[0], val_length[1])

    train_dataloader, train_sampler = None, None

    return test_dataloader, val_dataloader, train_dataloader, train_sampler


def save_model(epoch, args, model, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name == "" else type_name + ".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file


def reduce_loss(loss, args):
    world_size = args.world_size
    if world_size < 2:
        return loss
    with torch.no_grad():
        torch.distributed.reduce(loss, dst=0)
        if torch.distributed.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            loss /= world_size
    return loss


def _run_on_single_gpu(args, model, t_mask_list, v_mask_list, t_feat_list, v_feat_list,
                       cls_list, diffusion, mini_batch=32):
    mini_batch = args.t2v_num

    logger.info('[finish] map to main gpu')
    batch_t_mask = torch.split(t_mask_list, mini_batch)
    batch_v_mask = torch.split(v_mask_list, mini_batch)
    batch_t_feat = torch.split(t_feat_list, mini_batch)
    batch_v_feat = torch.split(v_feat_list, mini_batch)
    batch_cls_feat = torch.split(cls_list, mini_batch)

    sim_matrix, _batch_t_feat, _batch_v_feat = [], [], []

    logger.info('[finish] map to main gpu')
    with torch.no_grad():
        for idx1, (t_mask, t_feat, cls) in enumerate(zip(batch_t_mask, batch_t_feat, batch_cls_feat)):
            each_row = []
            for idx2, (v_mask, v_feat) in enumerate(zip(batch_v_mask, batch_v_feat)):
                logits, _, *_tmp = model.get_similarity_logits(t_feat, cls, v_feat, t_mask, v_mask)
                logits = logits.cpu().detach().numpy()
                each_row.append(logits)
            each_row = np.concatenate(tuple(each_row), axis=-1)
            sim_matrix.append(each_row)

    sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

    logger.info('diffusion')
    new_t2v_matrix, new_v2t_matrix = [], []
    _sim_matrix = torch.from_numpy(sim_matrix).to(t_feat_list.device)

    t2v_sim_matrix = _sim_matrix
    v2t_sim_matrix = _sim_matrix

    batch_t2v_matrix = torch.split(t2v_sim_matrix.clone(), mini_batch)
    batch_v2t_matrix = torch.split(v2t_sim_matrix.T.clone(), mini_batch)
    all_t_feat, all_v_feat = torch.cat(batch_cls_feat, dim=0), torch.cat(batch_v_feat, dim=0)

    with torch.no_grad():
        for idx1, (t2v_sim, t_feat) in enumerate(zip(batch_t2v_matrix, batch_cls_feat)):
            video_embeds, ids, mask = [], [], []
            for b in range(t2v_sim.size(0)):
                _, neg_idx = t2v_sim[b].topk(args.t2v_num, largest=True, sorted=True)
                ids.append(neg_idx)
                temp = []
                temp_mask = []
                for i in neg_idx:
                    temp.append(all_v_feat[i])
                    temp_mask.append(v_mask_list[i])
                video_embeds.append(torch.stack(temp, dim=0))
                mask.append(torch.stack(temp_mask, dim=0))
            video_embeds = torch.stack(video_embeds, dim=0)  # b_t, b_v, -1, 512
            mask = torch.stack(mask, dim=0)

            sample = diffusion.ddim_sample_loop(
                model.diffusion_model,
                (t_feat.size(0), args.t2v_num),
                clip_denoised=True,
                model_kwargs={"text_emb": t_feat,
                              "video_emb": video_embeds,
                              "video_mask": mask},
            )
            sample = F.softmax(sample * args.t2v_temp, dim=-1)
            for i in range(t2v_sim.size(0)):
                for _i, j in enumerate(ids[i]):
                    t2v_sim[i, j] += sample[i, _i] * args.t2v_alpha
            new_t2v_matrix.append(t2v_sim)

        for idx1, (v2t_sim, v_feat, v_mask) in enumerate(zip(batch_v2t_matrix, batch_v_feat, batch_v_mask)):
            text_embeds, ids, video_embeds = [], [], []
            for b in range(v2t_sim.size(0)):
                _, neg_idx = v2t_sim[b].topk(args.v2t_num, largest=True, sorted=True)
                ids.append(neg_idx)
                temp0, temp1 = [], []
                for i in neg_idx:
                    temp1.append(all_t_feat[i])
                text_embeds.append(torch.stack(temp1, dim=0))
            text_embeds = torch.stack(text_embeds, dim=0)

            sample = diffusion.ddim_sample_loop(
                model.diffusion_model_v,
                (v_feat.size(0), args.v2t_num),
                clip_denoised=True,
                model_kwargs={"text_emb": text_embeds,
                              "video_emb": v_feat,
                              "video_mask": v_mask},
            )
            sample = F.softmax(sample * args.v2t_temp, dim=-1)
            for i in range(v2t_sim.size(0)):
                for _i, j in enumerate(ids[i]):
                    v2t_sim[i, j] += sample[i, _i] * args.v2t_alpha
            new_v2t_matrix.append(v2t_sim)

    new_t2v_matrix = torch.cat(new_t2v_matrix, dim=0)
    new_v2t_matrix = torch.cat(new_v2t_matrix, dim=0)
    new_t2v_matrix = new_t2v_matrix.cpu().numpy()
    new_v2t_matrix = new_v2t_matrix.cpu().numpy()

    return new_t2v_matrix, new_v2t_matrix


def eval_epoch(args, model, test_dataloader, device, diffusion):
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    ## below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_:
        logger.warning("Eval under the multi-sentence per video clip setting.")
        logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()
    # ----------------------------
    # 1. cache the features
    # ----------------------------
    batch_mask_t, batch_mask_v, batch_feat_t, batch_feat_v, ids_t, ids_v = [], [], [], [], [], []
    batch_cls = []

    with torch.no_grad():
        tic = time.time()
        if multi_sentence_:  # multi-sentences retrieval means: one clip has two or more descriptions.
            total_video_num = 0
            logger.info('[start] extract text+video feature')
            for batch in tqdm(test_dataloader):
                batch = tuple(t.to(device) for t in batch)
                text_ids, text_mask, video, video_mask, inds, _ = batch

                b, *_t = video.shape
                text_feat, cls = model.get_text_feat(text_ids, text_mask)
                ids_t.append(inds)
                batch_mask_t.append(text_mask)
                batch_feat_t.append(text_feat)
                batch_cls.append(cls)

                video_feat = model.get_video_feat(video, video_mask)
                batch_mask_v.append(video_mask)
                batch_feat_v.append(video_feat)

                total_video_num += b

            ids_t = torch.cat(ids_t, dim=0).squeeze()
            batch_mask_t = torch.cat(batch_mask_t, dim=0)
            batch_mask_v = torch.cat(batch_mask_v, dim=0)
            batch_feat_t = torch.cat(batch_feat_t, dim=0)
            batch_feat_v = torch.cat(batch_feat_v, dim=0)
            batch_cls = torch.cat(batch_cls, dim=0)

            _batch_feat_v, _batch_mask_v = [], []
            for i in range(len(ids_t)):
                if ids_t[i] in cut_off_points_:
                    _batch_feat_v.append(batch_feat_v[i])
                    _batch_mask_v.append(batch_mask_v[i])

            batch_feat_v = torch.stack(_batch_feat_v, dim=0)
            batch_mask_v = torch.stack(_batch_mask_v, dim=0)

            logger.info('[finish] extract text+video feature')
        else:
            logger.info('[start] extract text+video feature')
            for batch in tqdm(test_dataloader):
                batch = tuple(t.to(device) for t in batch)
                text_ids, text_mask, video, video_mask, inds, _ = batch
                video_mask = video_mask.view(-1, video_mask.shape[-1])
                text_feat, video_feat, cls = model.get_text_video_feat(text_ids, text_mask, video, video_mask)
                ids_t.append(inds)
                batch_mask_t.append(text_mask)
                batch_mask_v.append(video_mask)
                batch_feat_t.append(text_feat)
                batch_feat_v.append(video_feat)
                batch_cls.append(cls)
            ids_t = allgather(torch.cat(ids_t, dim=0), args).squeeze()
            batch_mask_t = allgather(torch.cat(batch_mask_t, dim=0), args)
            batch_mask_v = allgather(torch.cat(batch_mask_v, dim=0), args)
            batch_feat_t = allgather(torch.cat(batch_feat_t, dim=0), args)
            batch_feat_v = allgather(torch.cat(batch_feat_v, dim=0), args)
            batch_cls = allgather(torch.cat(batch_cls, dim=0), args)
            batch_mask_t[ids_t] = batch_mask_t.clone()
            batch_mask_v[ids_t] = batch_mask_v.clone()
            batch_feat_t[ids_t] = batch_feat_t.clone()
            batch_feat_v[ids_t] = batch_feat_v.clone()
            batch_cls[ids_t] = batch_cls.clone()
            batch_mask_t = batch_mask_t[:ids_t.max() + 1, ...]
            batch_mask_v = batch_mask_v[:ids_t.max() + 1, ...]
            batch_feat_t = batch_feat_t[:ids_t.max() + 1, ...]
            batch_feat_v = batch_feat_v[:ids_t.max() + 1, ...]
            batch_cls = batch_cls[:ids_t.max() + 1, ...]
            logger.info('[finish] extract text+video feature')

    toc1 = time.time()

    logger.info('{} {} {} {}'.format(len(batch_mask_t), len(batch_mask_v), len(batch_feat_t), len(batch_feat_v)))
    # ----------------------------------
    # 2. calculate the similarity
    # ----------------------------------
    logger.info('[start] calculate the similarity')
    with torch.no_grad():
        new_t2v_matrix, new_v2t_matrix = _run_on_single_gpu(args, model, batch_mask_t,
                                                                        batch_mask_v, batch_feat_t, batch_feat_v,
                                                                        batch_cls, diffusion)
        sim_matrix = new_t2v_matrix
    logger.info('[end] calculate the similarity')

    toc2 = time.time()
    logger.info('[start] compute_metrics')
    if multi_sentence_:
        new_v2t_matrix = new_v2t_matrix.T
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_ - s_ for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_)])
        new_t2v_matrix_new, new_v2t_matrix_new = [], [], []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            new_t2v_matrix_new.append(np.concatenate((new_t2v_matrix[s_:e_],
                                                  np.full((max_length - e_ + s_, sim_matrix.shape[1]), -np.inf)),
                                                 axis=0))
            new_v2t_matrix_new.append(np.concatenate((new_v2t_matrix[s_:e_],
                                                  np.full((max_length - e_ + s_, sim_matrix.shape[1]), -np.inf)),
                                                 axis=0))
        new_t2v_matrix_new = np.stack(tuple(new_t2v_matrix_new), axis=0)
        new_v2t_matrix_new = np.stack(tuple(new_v2t_matrix_new), axis=0)

        logger.info("after reshape, new t2v matrix size: {} x {} x {}".
                    format(new_t2v_matrix_new.shape[0], new_t2v_matrix_new.shape[1], new_t2v_matrix_new.shape[2]))
        logger.info("after reshape, new v2t matrix size: {} x {} x {}".
                    format(new_v2t_matrix_new.shape[0], new_v2t_matrix_new.shape[1], new_v2t_matrix_new.shape[2]))

        new_tv_metrics = tensor_text_to_video_metrics(new_t2v_matrix_new)
        new_vt_metrics = compute_metrics(tensor_video_to_text_sim(new_v2t_matrix_new))
    else:
        logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))

        new_tv_metrics = compute_metrics(new_t2v_matrix)
        new_vt_metrics = compute_metrics(new_v2t_matrix)


    logger.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

    logger.info('[end] compute_metrics')

    toc3 = time.time()
    logger.info("time profile: feat {:.1f}s match {:.5f}s metrics {:.5f}s".format(toc1 - tic, toc2 - toc1, toc3 - toc2))

    logger.info(
        "Text-to-Video: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".
            format(new_tv_metrics['R1'], new_tv_metrics['R5'], new_tv_metrics['R10'], new_tv_metrics['R50'],
                   new_tv_metrics['MR'], new_tv_metrics['MeanR']))
    logger.info(
        "Video-to-Text: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".
            format(new_vt_metrics['R1'], new_vt_metrics['R5'], new_vt_metrics['R10'], new_vt_metrics['R50'],
                   new_vt_metrics['MR'], new_vt_metrics['MeanR']))

    return new_tv_metrics['R1']


def main():
    global logger
    global best_score
    global meters

    meters = MetricLogger(delimiter="  ")
    args = get_args()
    if not exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger('tvr', args.output_dir, args.local_rank)

    args = set_seed_logger(args)

    model = build_model(args)

    test_dataloader, val_dataloader, train_dataloader, train_sampler = build_dataloader(args)

    diffusion = create_gaussian_diffusion(args)
    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)

    eval_epoch(args, model, test_dataloader, args.device, diffusion)

if __name__ == "__main__":
    main()