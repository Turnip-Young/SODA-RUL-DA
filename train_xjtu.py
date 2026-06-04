import argparse
import itertools
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_xjtu import (
    BEARING_META,
    OPEN_SET_TASKS,
    XJTUBearingDataset,
    XJTUSingleBearingDataset,
    compute_scaler,
)
from loss import advLoss, weightedAdvLoss
from model import Discriminator, backboneDiscriminator, mymodel


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def masked_mean(x, padding, keepdim=False):
    valid = (1 - padding).float()
    while valid.dim() < x.dim():
        valid = valid.unsqueeze(-1)
    denom = valid.sum(dim=1, keepdim=keepdim).clamp_min(1.0)
    return (x * valid).sum(dim=1, keepdim=keepdim) / denom


def get_rul_stage(rul):
    stage = torch.zeros_like(rul, dtype=torch.long)
    stage[(rul <= 2.0 / 3.0) & (rul > 1.0 / 3.0)] = 1
    stage[rul <= 1.0 / 3.0] = 2
    return stage


def build_source_prototypes(features, labels, padding):
    pooled_feature = F.normalize(masked_mean(features, padding), dim=1)
    pooled_label = masked_mean(labels, padding)
    source_stage = get_rul_stage(pooled_label)
    global_proto = F.normalize(pooled_feature.mean(dim=0, keepdim=True), dim=1).squeeze(0)
    prototypes = []
    for stage_id in range(3):
        stage_mask = source_stage == stage_id
        if torch.sum(stage_mask) > 0:
            proto = pooled_feature[stage_mask].mean(dim=0, keepdim=True)
            proto = F.normalize(proto, dim=1).squeeze(0)
        else:
            proto = global_proto
        prototypes.append(proto)
    return torch.stack(prototypes, dim=0)


def monotonic_consistency(pred, padding, temperature):
    valid_pair = ((1 - padding[:, 1:]) * (1 - padding[:, :-1])).float()
    violation = F.relu(pred[:, 1:] - pred[:, :-1]) * valid_pair
    denom = valid_pair.sum(dim=1).clamp_min(1.0)
    violation = violation.sum(dim=1) / denom
    return torch.exp(-violation / temperature)


def target_transfer_weight(args, s_features, s_labels, s_padding, t_features, t_out, t_padding):
    with torch.no_grad():
        prototypes = build_source_prototypes(s_features, s_labels, s_padding)
        target_feature = F.normalize(masked_mean(t_features, t_padding), dim=1)
        target_rul = masked_mean(t_out, t_padding).clamp(0.0, 1.0)
        target_stage = get_rul_stage(target_rul)
        matched_proto = prototypes[target_stage]
        distance = torch.norm(target_feature - matched_proto, p=2, dim=1)
        proto_score = torch.exp(-distance / args.proto_temp)
        mono_score = monotonic_consistency(t_out, t_padding, args.mono_temp)
        weight = proto_score * mono_score
        if args.normalize_weight:
            weight = weight / weight.mean().clamp_min(1e-6)
        return weight.clamp(args.min_weight, 1.0), prototypes, target_stage


def selective_prototype_loss(t_features, t_padding, prototypes, target_stage, target_weight):
    target_feature = F.normalize(masked_mean(t_features, t_padding), dim=1)
    matched_proto = prototypes.detach()[target_stage.detach()]
    distance = torch.sum((target_feature - matched_proto) ** 2, dim=1)
    weight = target_weight.detach()
    return torch.sum(distance * weight) / (torch.sum(weight) + 1e-6)


def collect_prediction(out, labels, seq_len, life):
    pred_sum = torch.zeros(life)
    pred_cnt = torch.zeros(life)
    data_len = len(out)
    for j in range(data_len):
        if j < seq_len - 1:
            pred_sum[: j + 1] += out[j, -(j + 1) :]
            pred_cnt[: j + 1] += 1
        elif j <= data_len - seq_len:
            start = j - seq_len + 1
            pred_sum[start : j + 1] += out[j]
            pred_cnt[start : j + 1] += 1
        else:
            valid = data_len - j
            start = life - valid
            pred_sum[start:life] += out[j, :valid]
            pred_cnt[start:life] += 1
    truth = torch.tensor([labels[j, -1] for j in range(life)], dtype=torch.float)
    pred = pred_sum / pred_cnt.clamp_min(1.0)
    return truth, pred


def evaluate_bearing(net, dataset, seq_len, device):
    net.eval()
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    outs, lbs = [], []
    with torch.no_grad():
        for data, label, padding in loader:
            data = data.to(device)
            padding = padding.to(device)
            _, out = net(data, padding)
            outs.append(out.squeeze(2).cpu())
            lbs.append(label)
    out = torch.cat(outs, dim=0)
    labels = torch.cat(lbs, dim=0)
    truth, pred = collect_prediction(out, labels, seq_len, dataset.life)
    pred = pred.clamp(0.0, 1.0)
    rmse = math.sqrt(float(torch.mean((pred - truth) ** 2)))
    mae = float(torch.mean(torch.abs(pred - truth)))
    return rmse, mae, truth, pred


def evaluate(net, target_bearings, args, scaler, device):
    metrics = []
    for bearing in target_bearings:
        ds = XJTUSingleBearingDataset(
            args.data_root, args.cache_root, bearing, args.seq_len, scaler, args.rebuild_cache
        )
        rmse, mae, _, _ = evaluate_bearing(net, ds, args.seq_len, device)
        meta = BEARING_META[bearing]
        metrics.append((bearing, meta["fault"], rmse, mae))
    mean_rmse = sum(m[2] for m in metrics) / len(metrics)
    mean_mae = sum(m[3] for m in metrics) / len(metrics)
    return mean_rmse, mean_mae, metrics


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != "cpu" else "cpu")
    if args.gpu != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    task = OPEN_SET_TASKS[args.task]
    source_bearings = task["source"]
    target_bearings = task["target"]
    scaler = compute_scaler(args.data_root, args.cache_root, source_bearings, args.rebuild_cache)
    source_data = XJTUBearingDataset(
        args.data_root, args.cache_root, source_bearings, args.seq_len, scaler, args.rebuild_cache
    )
    target_data = XJTUBearingDataset(
        args.data_root, args.cache_root, target_bearings, args.seq_len, scaler, args.rebuild_cache
    )

    net = mymodel(d_model=args.feature_dim, max_len=args.seq_len).to(device)
    d_out = Discriminator(args.seq_len).to(device)
    d_fea = backboneDiscriminator(args.seq_len, d=args.feature_dim).to(device)
    loss_fn = nn.MSELoss().to(device)
    opt = torch.optim.SGD(
        itertools.chain(net.parameters(), d_out.parameters(), d_fea.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(opt, 80, 0.5)

    best_rmse = 999.0
    os.makedirs(args.output_dir, exist_ok=True)
    print("task={}, source={}, target={}, method=type{}".format(
        args.task, source_bearings, target_bearings, args.type
    ))
    print("device={}, source_windows={}, target_windows={}".format(
        device, len(source_data), len(target_data)
    ))

    for epoch in range(args.epoch):
        net.train()
        loss1_sum, fea_sum, out_sum, weight_sum, mono_sum, proto_sum = 0, 0, 0, 0, 0, 0
        cnt = 0
        s_iter = iter(DataLoader(source_data, batch_size=args.batch_size, shuffle=True))
        t_iter = iter(DataLoader(target_data, batch_size=args.batch_size, shuffle=True))
        steps = min(len(s_iter), len(t_iter))
        for _ in range(steps):
            s_data, t_data = next(s_iter), next(t_iter)
            s_input, s_label, s_padding = [x.to(device) for x in s_data]
            t_input, _, t_padding = [x.to(device) for x in t_data]
            s_features, s_out = net(s_input, s_padding)
            t_features, t_out = net(t_input, t_padding)
            s_out = s_out.squeeze(2)
            t_out = t_out.squeeze(2)
            loss1 = loss_fn(s_out, s_label)
            loss = loss1
            loss1_sum += loss1
            cnt += 1

            if args.type in [1, 2, 3]:
                s_domain_fea = d_fea(s_features)
                t_domain_fea = d_fea(t_features)
            if args.type in [0, 2, 3]:
                s_domain_out = d_out(s_out)
                t_domain_out = d_out(t_out)

            if args.type == 0:
                out_loss = advLoss(s_domain_out.squeeze(1), t_domain_out.squeeze(1), device.type)
                out_sum += out_loss
                loss = loss + args.b * out_loss
            elif args.type == 1:
                fea_loss = advLoss(s_domain_fea.squeeze(1), t_domain_fea.squeeze(1), device.type)
                fea_sum += fea_loss
                loss = loss + args.a * fea_loss
            elif args.type == 2 and epoch >= args.warmup_epoch:
                fea_loss = advLoss(s_domain_fea.squeeze(1), t_domain_fea.squeeze(1), device.type)
                out_loss = advLoss(s_domain_out.squeeze(1), t_domain_out.squeeze(1), device.type)
                fea_sum += fea_loss
                out_sum += out_loss
                loss = loss + args.a * fea_loss + args.b * out_loss
            elif args.type == 3 and epoch >= args.warmup_epoch:
                target_weight, prototypes, target_stage = target_transfer_weight(
                    args, s_features, s_label, s_padding, t_features, t_out, t_padding
                )
                fea_loss = weightedAdvLoss(
                    s_domain_fea.squeeze(1), t_domain_fea.squeeze(1), target_weight, device.type
                )
                out_loss = weightedAdvLoss(
                    s_domain_out.squeeze(1), t_domain_out.squeeze(1), target_weight, device.type
                )
                valid_pair = (1 - t_padding[:, 1:]) * (1 - t_padding[:, :-1])
                mono_loss = torch.sum(F.relu(t_out[:, 1:] - t_out[:, :-1]) * valid_pair) / valid_pair.sum().clamp_min(1.0)
                proto_loss = selective_prototype_loss(t_features, t_padding, prototypes, target_stage, target_weight)
                fea_sum += fea_loss
                out_sum += out_loss
                weight_sum += target_weight.mean()
                mono_sum += mono_loss
                proto_sum += proto_loss
                loss = loss + args.a * fea_loss + args.b * out_loss + args.c * mono_loss + args.d * proto_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                itertools.chain(net.parameters(), d_out.parameters(), d_fea.parameters()), 2
            )
            opt.step()

        if args.scheduler:
            scheduler.step()

        mean_rmse, mean_mae, metrics = evaluate(net, target_bearings, args, scaler, device)
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            ckpt = os.path.join(args.output_dir, "xjtu_{}_type{}_net.pth".format(args.task, args.type))
            torch.save(net.state_dict(), ckpt)

        print(
            "{}/{} | rul={:.5f}, fea={:.5f}, out={:.5f}, w={:.5f}, mono={:.5f}, proto={:.5f}, rmse={:.5f}, mae={:.5f}".format(
                epoch,
                args.epoch,
                float(loss1_sum / cnt),
                float(fea_sum / cnt),
                float(out_sum / cnt),
                float(weight_sum / cnt),
                float(mono_sum / cnt),
                float(proto_sum / cnt),
                mean_rmse,
                mean_mae,
            )
        )
        detail = ", ".join(["{}({}):{:.4f}".format(b, f, r) for b, f, r, _ in metrics])
        print("  {}".format(detail))

    return best_rmse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../XJTU-SY_Bearing_Datasets")
    parser.add_argument("--cache_root", type=str, default="cache/xjtu_features")
    parser.add_argument("--output_dir", type=str, default="online")
    parser.add_argument("--task", type=str, default="c1_outer_to_c2_mixed", choices=list(OPEN_SET_TASKS.keys()))
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--feature_dim", type=int, default=24)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--b", type=float, default=0.5)
    parser.add_argument("--c", type=float, default=0.05)
    parser.add_argument("--d", type=float, default=0.01)
    parser.add_argument("--warmup_epoch", type=int, default=5)
    parser.add_argument("--proto_temp", type=float, default=1.0)
    parser.add_argument("--mono_temp", type=float, default=0.05)
    parser.add_argument("--min_weight", type=float, default=0.05)
    parser.add_argument("--normalize_weight", type=int, default=1, choices=[0, 1])
    parser.add_argument("--scheduler", type=int, default=1, choices=[0, 1])
    parser.add_argument("--type", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--rebuild_cache", type=int, default=0, choices=[0, 1])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    t0 = time.perf_counter()
    best = train(args)
    t1 = time.perf_counter()
    print("best_rmse={:.5f}, time={:.2f}s".format(best, t1 - t0))
