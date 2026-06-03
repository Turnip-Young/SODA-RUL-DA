import torch.nn as nn
import torch.nn.functional as F
from model import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import argparse
from dataset import TRANSFORMER_ALL_DATA, TRANSFORMERDATA
from torch.utils.data import DataLoader, random_split
from loss import advLoss, weightedAdvLoss
import itertools
import time


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


def target_transfer_weight(s_features, s_labels, s_padding, t_features, t_out, t_padding, return_info=False):
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
        weight = weight.clamp(args.min_weight, 1.0)
        if return_info:
            return weight, prototypes, target_stage
        return weight


def selective_prototype_loss(t_features, t_padding, prototypes, target_stage, target_weight):
    target_feature = F.normalize(masked_mean(t_features, t_padding), dim=1)
    matched_proto = prototypes.detach()[target_stage.detach()]
    distance = torch.sum((target_feature - matched_proto) ** 2, dim=1)
    weight = target_weight.detach()
    return torch.sum(distance * weight) / (torch.sum(weight) + 1e-6)


def validate():
    net.eval()
    tot = 0
    with torch.no_grad():
        for i in target_test_names:
            pred_sum, pred_cnt = torch.zeros(800), torch.zeros(800)
            valid_data = TRANSFORMERDATA(i, seq_len)
            data_len = len(valid_data)
            valid_loader = DataLoader(valid_data, batch_size=1000)
            valid_iter = iter(valid_loader)
            d = next(valid_iter)
            input, lbl, msk = d[0], d[1], d[2]
            input, msk = input.cuda(), msk.cuda()
            _, out = net(input, msk)
            out = out.squeeze(2).cpu()
            for j in range(data_len):
                if j < seq_len-1:
                    pred_sum[:j+1] += out[j, -(j+1):]
                    pred_cnt[:j+1] += 1
                elif j <= data_len-seq_len:
                    pred_sum[j-seq_len+1:j+1] += out[j]
                    pred_cnt[j-seq_len+1:j+1] += 1
                else:
                    pred_sum[data_len-seq_len+1-(data_len-j):data_len-seq_len+1] += out[j, :(data_len-j)]
                    pred_cnt[data_len-seq_len+1-(data_len-j):data_len-seq_len+1] += 1
            truth = torch.tensor([lbl[j,-1] for j in range(len(lbl)-seq_len+1)], dtype=torch.float)
            pred_sum, pred_cnt = pred_sum[:data_len-seq_len+1], pred_cnt[:data_len-seq_len+1]
            pred = pred_sum/pred_cnt
            mse = float(torch.sum(torch.pow(pred-truth, 2)))
            rmse = math.sqrt(mse/data_len)
            tot += rmse
    return tot*Rc/len(valid_list)


def train():
    minn = 999
    for e in range(epochs):
        al, tot = 0, 0
        net.train()
        random.shuffle(source_list)
        random.shuffle(target_list)
        source_iter, target_iter = iter(source_list), iter(target_list)
        loss2_sum, loss1_sum = 0, 0
        bkb_sum, out_sum = 0, 0
        weight_sum, mono_sum, proto_sum = 0, 0, 0
        cnt = 0
        s_iter = iter(DataLoader(s_data, batch_size=args.batch_size, shuffle=True))
        t_iter = iter(DataLoader(t_data, batch_size=args.batch_size, shuffle=True))
        l = min(len(s_iter), len(t_iter))
        for _ in range(l):
            s_d, t_d = next(s_iter), next(t_iter)
            s_input, s_lb, s_msk = s_d[0], s_d[1], s_d[2]
            t_input, t_msk = t_d[0], t_d[2]
            s_input, s_lb, s_msk = s_input.cuda(), s_lb.cuda(), s_msk.cuda()
            t_input, t_msk = t_input.cuda(), t_msk.cuda()
            s_features, s_out = net(s_input, s_msk)
            t_features, t_out = net(t_input, t_msk) # [bts, seq_len, feature_num]
            s_out.squeeze_(2)
            t_out.squeeze_(2)
            loss1 = Loss(s_out, s_lb)
            loss1_sum += loss1
            cnt += 1
            if args.type == 1 or args.type == 0:
                if args.type == 1:
                    s_domain = D2(s_features)
                    t_domain = D2(t_features)
                else:
                    s_domain = D1(s_out)
                    t_domain = D1(t_out)
                loss2 = advLoss(s_domain.squeeze(1), t_domain.squeeze(1), 'cuda')
                loss2_sum += loss2
                loss = loss1 + a*loss2
            elif args.type == 2:
                s_domain_bkb = D2(s_features)
                t_domain_bkb = D2(t_features)
                s_domain_out = D1(s_out)
                t_domain_out = D1(t_out)
                if e>=5:
                    fea_loss = advLoss(s_domain_bkb.squeeze(1), t_domain_bkb.squeeze(1), 'cuda')
                    out_loss = advLoss(s_domain_out.squeeze(1), t_domain_out.squeeze(1), 'cuda')
                    bkb_sum += fea_loss
                    out_sum += out_loss
                    loss = loss1 + a*fea_loss + b*out_loss
                else:
                    loss = loss1
            elif args.type == 3:
                s_domain_bkb = D2(s_features)
                t_domain_bkb = D2(t_features)
                s_domain_out = D1(s_out)
                t_domain_out = D1(t_out)
                if e >= args.warmup_epoch:
                    target_weight, prototypes, target_stage = target_transfer_weight(
                        s_features, s_lb, s_msk, t_features, t_out, t_msk, True
                    )
                    fea_loss = weightedAdvLoss(
                        s_domain_bkb.squeeze(1), t_domain_bkb.squeeze(1),
                        target_weight, 'cuda'
                    )
                    out_loss = weightedAdvLoss(
                        s_domain_out.squeeze(1), t_domain_out.squeeze(1),
                        target_weight, 'cuda'
                    )
                    valid_pair = (1 - t_msk[:, 1:]) * (1 - t_msk[:, :-1])
                    mono_loss = torch.sum(F.relu(t_out[:, 1:] - t_out[:, :-1]) * valid_pair) / valid_pair.sum().clamp_min(1.0)
                    proto_loss = selective_prototype_loss(
                        t_features, t_msk, prototypes, target_stage, target_weight
                    )
                    bkb_sum += fea_loss
                    out_sum += out_loss
                    weight_sum += target_weight.mean()
                    mono_sum += mono_loss
                    proto_sum += proto_loss
                    loss = loss1 + a*fea_loss + b*out_loss + args.c*mono_loss + args.d*proto_loss
                else:
                    loss = loss1
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(itertools.chain(net.parameters(), D1.parameters(), D2.parameters()), 2)
            opt.step()    

        rmse = validate()
        if args.type == 2:
            print("{}/{}| loss1={:.5f}, fea_loss={:.5f}, out_loss={:.5f}, rmse={:.5f}".\
                format(e, args.epoch, float(loss1_sum/cnt), float(bkb_sum/cnt), float(out_sum/cnt), rmse))
        elif args.type == 3:
            print("{}/{}| loss1={:.5f}, fea_loss={:.5f}, out_loss={:.5f}, w={:.5f}, mono={:.5f}, proto={:.5f}, rmse={:.5f}".\
                format(e, args.epoch, float(loss1_sum/cnt), float(bkb_sum/cnt), float(out_sum/cnt), float(weight_sum/cnt), float(mono_sum/cnt), float(proto_sum/cnt), rmse))
        else:    
            print("{}/{}| 1={:.5f}, 2={:.5f}, rmse={:.5f}".format(e, args.epoch, loss1, loss2_sum/cnt, rmse))
        if rmse<minn:
            minn = rmse
            print("min={}".format(minn))
            if args.type == 1:
                torch.save(net.state_dict(), "save/final/dann_"+source[-1]+target[-1]+".pth")
            elif args.type == 0:
                torch.save(net.state_dict(), "save/final/out_"+source[-1]+target[-1]+".pth")
            elif args.type == 2 :
                #torch.save(net.state_dict(), "save/final/both_"+source[-1]+target[-1]+".pth")
                torch.save(net.state_dict(), "online/"+source[-1]+target[-1]+"_net.pth")
                torch.save(D1.state_dict(), "online/"+source[-1]+target[-1]+"_D1.pth")
                torch.save(D2.state_dict(), "online/"+source[-1]+target[-1]+"_D2.pth")
            elif args.type == 3:
                torch.save(net.state_dict(), "online/soda_"+source[-1]+target[-1]+"_net.pth")
                torch.save(D1.state_dict(), "online/soda_"+source[-1]+target[-1]+"_D1.pth")
                torch.save(D2.state_dict(), "online/soda_"+source[-1]+target[-1]+"_D2.pth")
        
        if args.scheduler:
            sch.step()

    return minn


def get_score(pred, truth):
    """input must be tensors!"""
    x = pred-truth
    score1 = torch.tensor([torch.exp(-i/13)-1 for i in x if i<0])
    score2 = torch.tensor([torch.exp(i/10)-1 for i in x if i>=0])
    return int(torch.sum(score1)+torch.sum(score2))


if __name__ == "__main__":
    seed = 0
    torch.manual_seed(seed)            
    torch.cuda.manual_seed(seed)       
    torch.cuda.manual_seed_all(seed)    
    random.seed(seed)
    np.random.seed(seed)
    Rc = 130

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument("--epoch", type=int, default=240)
    parser.add_argument("--batch_size", type=int, default=128, help="batch_size")
    parser.add_argument("--seq_len", type=int, default=70)
    parser.add_argument("--source", type=str, default="FD003", help="decide source file", choices=['FD001','FD002','FD003','FD004'])
    parser.add_argument("--target", type=str, default="FD002", help="decide target file", choices=['FD001','FD002','FD003','FD004'])
    parser.add_argument("--a", type=float, default=0.1, help='hyper-param α')
    parser.add_argument("--b", type=float, default=0.5, help='hyper-param β')
    parser.add_argument("--c", type=float, default=0.05, help='monotonic consistency weight for type=3')
    parser.add_argument("--d", type=float, default=0.01, help='selective prototype consistency weight for type=3')
    parser.add_argument("--warmup_epoch", type=int, default=5, help='source-only warmup epochs before DA')
    parser.add_argument("--proto_temp", type=float, default=1.0, help='temperature for prototype transfer score')
    parser.add_argument("--mono_temp", type=float, default=0.05, help='temperature for monotonic transfer score')
    parser.add_argument("--min_weight", type=float, default=0.05, help='minimum target alignment weight')
    parser.add_argument("--normalize_weight", type=int, default=1, choices=[0,1], help='normalize target weights by batch mean')
    parser.add_argument("--scheduler", type=int, default=1, choices=[0,1], help="1 for using sheduler while 0 for not")
    parser.add_argument("--type", type=int, default=2, choices=[0,1,2,3], help="0:out only | 1:DANN | 2:backbone+output | 3:selective open-set DA")
    parser.add_argument("--train", default=1, type=int)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
    source, target = args.source, args.target
    data_root = "CMAPSS/units/"
    label_root = "CMAPSS/labels/"
    type = {0:"out_only", 1:"DANN", 2:"backbone + output", 3:"selective open-set DA"}
    seq_len, a, epochs, b = args.seq_len, args.a, args.epoch, args.b
    option_str = "source={}, target={}, a={}, b={}, epochs={}, type={}, lr={}, {}using scheduler".\
        format(source, target, a, b, epochs, type[args.type], args.lr, "" if args.scheduler else "not ")
    print(option_str)

    net = mymodel(max_len=seq_len) 
    D1 = Discriminator(seq_len)
    D2 = backboneDiscriminator(seq_len)
    if args.type == 0:
        opt = torch.optim.SGD(itertools.chain(net.parameters(), D1.parameters()), lr=args.lr)
    elif args.type == 1:
        opt = torch.optim.SGD(itertools.chain(net.parameters(), D2.parameters()), lr=args.lr)
    elif args.type == 2 or args.type == 3:
        opt = torch.optim.SGD(itertools.chain(net.parameters(), D1.parameters(), D2.parameters()), lr=args.lr)
    Loss = nn.MSELoss()
    net, Loss, D1, D2 = net.cuda(), Loss.cuda(), D1.cuda(), D2.cuda()
    sch = torch.optim.lr_scheduler.StepLR(opt, 80, 0.5)

    source_list = np.loadtxt("save/"+source+"/train"+source+".txt", dtype=str).tolist()
    target_list = np.loadtxt("save/"+target+"/train"+target+".txt", dtype=str).tolist()
    valid_list = np.loadtxt("save/"+target+"/test"+target+".txt", dtype=str).tolist()
    a_list = np.loadtxt("save/"+target+"/valid"+target+".txt", dtype=str).tolist()
    target_test_names = valid_list + a_list
    minl = min(len(source_list), len(target_list))
    s_data = TRANSFORMER_ALL_DATA(source_list, seq_len)
    t_data = TRANSFORMER_ALL_DATA(target_list, seq_len)
    t_data_test = TRANSFORMER_ALL_DATA(target_test_names, seq_len)
    if not os.path.exists('./online'):
        os.makedirs('./online')

    if args.train:
        train_time1 = time.perf_counter()
        minn = train()
        train_time2 = time.perf_counter()
        print(option_str)
        print("best = {}, train time = {}".format(minn, train_time2-train_time1))


