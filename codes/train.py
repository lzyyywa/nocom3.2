import os
import random

from torch.utils.data.dataloader import DataLoader
import tqdm
import test as test
from loss import *
from loss import KLLoss
import torch.multiprocessing
import numpy as np
import json
import math
from utils.ade_utils import emd_inference_opencv_test
from collections import Counter

from utils.hsic import hsic_normalized_cca


def cal_conditional(attr2idx, obj2idx, set_name, daset):
    def load_split(path):
        with open(path, 'r') as f:
            loaded_data = json.load(f)
        return loaded_data

    train_data = daset.train_data
    val_data = daset.val_data
    test_data = daset.test_data
    all_data = train_data + val_data + test_data
    if set_name == 'test':
        used_data = test_data
    elif set_name == 'all':
        used_data = all_data
    elif set_name == 'train':
        used_data = train_data

    v_o = torch.zeros(size=(len(attr2idx), len(obj2idx)))
    for item in used_data:
        verb_idx = attr2idx[item[1]]
        obj_idx = obj2idx[item[2]]

        v_o[verb_idx, obj_idx] += 1

    v_o_on_v = v_o / (torch.sum(v_o, dim=1, keepdim=True) + 1.0e-6)
    v_o_on_o = v_o / (torch.sum(v_o, dim=0, keepdim=True) + 1.0e-6)

    return v_o_on_v, v_o_on_o


def evaluate(model, dataset, config):
    model.eval()
    evaluator = test.Evaluator(dataset, model=None)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(
        model, dataset, config)
    test_stats = test.test(
        dataset,
        evaluator,
        all_logits,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        config
    )
    result = ""
    key_set = ["attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all", "best_seen", "best_unseen", "best_hm", "AUC"]

    for key in key_set:
        # if key in key_set:
        result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
    print(result)
    model.train()
    return loss_avg, test_stats


def save_checkpoint(state, save_path, epoch, best=False):
    filename = os.path.join(save_path, f"epoch_resume.pt")
    torch.save(state, filename)


# ========conditional train=
def rand_bbox(size, lam):
    W = size[-2]
    H = size[-1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def c2c_vanilla(model, optimizer, lr_scheduler, config, train_dataset, val_dataset, test_dataset,
                scaler):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    model.train()
    best_loss = 1e5
    best_metric = 0
    # 由于损失已交由 loss.py 中的 loss_calu 处理，此处的 Loss_fn 仅保留占位以防外部依赖
    Loss_fn = CrossEntropyLoss()
    log_training = open(os.path.join(config.save_path, 'log.txt'), 'w')

    attr2idx = train_dataset.attr2idx
    obj2idx = train_dataset.obj2idx

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_dataset.train_pairs]).cuda()

    train_losses = []

    for i in range(config.epoch_start, config.epochs):
        progress_bar = tqdm.tqdm(
            total=len(train_dataloader), desc="epoch % 3d" % (i + 1)
        )

        epoch_train_losses = []
        epoch_com_losses = []
        epoch_oo_losses = []
        epoch_vv_losses = []

        temp_lr = optimizer.param_groups[-1]['lr']
        print(f'Current_lr:{temp_lr}')
        for bid, batch in enumerate(train_dataloader):
            # ------------------------------------------------------------------
            # 修改点 1：精确解析 dataset 中返回的 6 个返回值，增加粗粒度标签
            # ------------------------------------------------------------------
            batch_img = batch[0].cuda()
            batch_verb = batch[1].cuda()
            batch_obj = batch[2].cuda()
            batch_target = batch[3].cuda()
            batch_coarse_verb = batch[4].cuda()
            batch_coarse_obj = batch[5].cuda()

            # 将用于计算的所有目标标签打包传给我们重新构建的 loss_calu
            target = [batch_img, batch_verb, batch_obj, batch_target, batch_coarse_verb, batch_coarse_obj]

            with torch.cuda.amp.autocast(enabled=True):
                # --------------------------------------------------------------
                # 修改点 2：将必需的标签一并传入 model 的 forward，用于内部截取特征并返回字典
                # --------------------------------------------------------------
                predict = model(
                    video=batch_img,
                    batch_verb=batch_verb,
                    batch_obj=batch_obj,
                    batch_coarse_verb=batch_coarse_verb,
                    batch_coarse_obj=batch_coarse_obj,
                    pairs=batch_target
                )

                # --------------------------------------------------------------
                # 修改点 3：调用第四步中的 loss_calu 统一计算 双曲HEM、对齐损失 及 欧式CE
                # --------------------------------------------------------------
                loss = loss_calu(predict, target, config)
                loss = loss / config.gradient_accumulation_steps

            # Accumulates scaled gradients.
            scaler.scale(loss).backward()

            # weights update
            if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_dataloader)):
                scaler.unscale_(optimizer)  # TODO:May be the reason for low acc on verb
                # scaler.step(prompt_optimizer)
                scaler.step(optimizer)
                scaler.update()

                # prompt_optimizer.zero_grad()
                optimizer.zero_grad()

            epoch_train_losses.append(loss.item() * config.gradient_accumulation_steps)

            # Record component losses for debugging (原有的子损失变量赋 0.0 防止报错，新体系由 total loss 统揽)
            epoch_com_losses.append(0.0)
            epoch_vv_losses.append(0.0)
            epoch_oo_losses.append(0.0)

            progress_bar.set_postfix({"train loss": np.mean(epoch_train_losses[-50:])})
            progress_bar.update()

            # break
        lr_scheduler.step()
        progress_bar.close()
        progress_bar.write(f"epoch {i + 1} train loss {np.mean(epoch_train_losses)}")
        train_losses.append(np.mean(epoch_train_losses))
        log_training.write('\n')
        log_training.write(f"epoch {i + 1} train loss {np.mean(epoch_train_losses)}\n")

        # Additional logging for component losses
        log_training.write(f"epoch {i + 1} com loss {np.mean(epoch_com_losses)}\n")
        log_training.write(f"epoch {i + 1} vv loss {np.mean(epoch_vv_losses)}\n")
        log_training.write(f"epoch {i + 1} oo loss {np.mean(epoch_oo_losses)}\n")

        if (i + 1) % config.save_every_n == 0:
            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'scaler': scaler.state_dict(),
            }, config.save_path, i)
        # if (i + 1) > config.val_epochs_ts:
        #     torch.save(model.state_dict(), os.path.join(config.save_path, f"epoch_{i}.pt"))
        key_set = ["attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all", "best_seen", "best_unseen", "best_hm",
                   "AUC"]
        if i % config.eval_every_n == 0 or i + 1 == config.epochs or i >= config.val_epochs_ts:
            print("Evaluating val dataset:")
            loss_avg, val_result = evaluate(model, val_dataset, config)
            result = ""
            for key in val_result:
                if key in key_set:
                    result = result + key + "  " + str(round(val_result[key], 4)) + "| "
            log_training.write('\n')
            log_training.write(result)
            print("Loss average on val dataset: {}".format(loss_avg))
            log_training.write('\n')
            log_training.write("Loss average on val dataset: {}\n".format(loss_avg))
            if config.best_model_metric == "best_loss":
                if loss_avg.cpu().float() < best_loss:
                    print('find best!')
                    log_training.write('find best!')
                    best_loss = loss_avg.cpu().float()
                    print("Evaluating test dataset:")
                    loss_avg, val_result = evaluate(model, test_dataset, config)
                    torch.save(model.state_dict(), os.path.join(
                        config.save_path, f"best.pt"
                    ))
                    result = ""
                    for key in val_result:
                        if key in key_set:
                            result = result + key + "  " + str(round(val_result[key], 4)) + "| "
                    log_training.write('\n')
                    log_training.write(result)
                    print("Loss average on test dataset: {}".format(loss_avg))
                    log_training.write('\n')
                    log_training.write("Loss average on test dataset: {}\n".format(loss_avg))
            else:
                if val_result[config.best_model_metric] > best_metric:
                    best_metric = val_result[config.best_model_metric]
                    log_training.write('\n')
                    print('find best!')
                    log_training.write('find best!')
                    loss_avg, val_result = evaluate(model, test_dataset, config)
                    torch.save(model.state_dict(), os.path.join(
                        config.save_path, f"best.pt"
                    ))
                    result = ""
                    for key in val_result:
                        if key in key_set:
                            result = result + key + "  " + str(round(val_result[key], 4)) + "| "
                    log_training.write('\n')
                    log_training.write(result)
                    print("Loss average on test dataset: {}".format(loss_avg))
                    log_training.write('\n')
                    log_training.write("Loss average on test dataset: {}\n".format(loss_avg))
        log_training.write('\n')
        log_training.flush()
        key_set = ["attr_acc", "obj_acc", "ub_seen", "ub_unseen", "ub_all", "best_seen", "best_unseen", "best_hm",
                   "AUC"]
        if i + 1 == config.epochs:
            print("Evaluating test dataset on Closed World")
            model.load_state_dict(torch.load(os.path.join(
                config.save_path, "best.pt"
            )))
            loss_avg, val_result = evaluate(model, test_dataset, config)
            result = ""
            for key in val_result:
                if key in key_set:
                    result = result + key + "  " + str(round(val_result[key], 4)) + "| "
            log_training.write('\n')
            log_training.write(result)
            print("Final Loss average on test dataset: {}".format(loss_avg))
            log_training.write('\n')
            log_training.write("Final Loss average on test dataset: {}\n".format(loss_avg))


import random
import os
import pprint
from opts import parser
from models.compositional_models import get_model

from loss import *
from utils.my_lr_scheduler import WarmupCosineAnnealingLR

import yaml
import shutil
from utils.get_optimizer import get_optimizer
from utils import CosineAnnealingLR
from loss import KLLoss
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


def set_seed(seed):
    """function sets the seed value
    Args:
        seed (int): seed value
    """
    seed = int(seed)
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_args(filename, args):
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    for key, group in data_loaded.items():
        for key, val in group.items():
            setattr(args, key, val)


if __name__ == "__main__":
    config = parser.parse_args()
    load_args(config.config, config)
    config.save_path = config.save_path + '/'
    print(config)
    # set the seed value
    set_seed(config.seed)

    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("training details")
    pprint.pprint(config)

    i = 0
    temp_save_path = config.save_path + str(i)
    while os.path.exists(temp_save_path):
        i = i + 1
        print(f'file {temp_save_path} already exists')
        print('exiting!')
        temp_save_path = config.save_path + str(i)
    config.save_path = temp_save_path
    print(f'file {temp_save_path} ')

    dataset_path = config.dataset_path

    use_composed_pair_loss = True if config.method == 'oadis' else False
    if config.dataset == 'sth-com':
        from dataset.com_video_dataset import CompositionVideoDataset
    else:
        raise NotImplemented

    train_dataset = CompositionVideoDataset(dataset_path,
                                            phase='train',
                                            split='compositional-split-natural',
                                            tdn_input='tdn' in config.arch,
                                            aux_input=config.aux_input,
                                            ade_input=config.ade_input,
                                            frames_duration=config.num_frames,
                                            use_composed_pair_loss=use_composed_pair_loss)

    val_dataset = CompositionVideoDataset(dataset_path,
                                          phase='val',
                                          split='compositional-split-natural',
                                          tdn_input='tdn' in config.arch,
                                          frames_duration=config.num_frames,
                                          use_composed_pair_loss=use_composed_pair_loss
                                          )

    test_dataset = CompositionVideoDataset(dataset_path,
                                           phase='test',
                                           split='compositional-split-natural',
                                           tdn_input='tdn' in config.arch,
                                           frames_duration=config.num_frames,
                                           use_composed_pair_loss=use_composed_pair_loss)

    model = get_model(train_dataset, config)
    optimizer = get_optimizer(config, model)

    lr_scheduler = CosineAnnealingLR.WarmupCosineLR(optimizer=optimizer, milestones=[config.warmup, config.epochs],
                                                    warmup_iters=config.warmup, min_ratio=1e-8)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    if config.method == 'c2c_vanilla':
        from train_models import c2c_vanilla
        train_model = c2c_vanilla
    elif config.method == 'c2c_enhance':
        from train_models import c2c_enhance
        train_model = c2c_enhance
    else:
        raise NotImplementedError

    os.makedirs(config.save_path, exist_ok=True)
    model = torch.nn.DataParallel(model).cuda()
    if config.pretrain:
        model.load_state_dict(torch.load(config.load_model), strict=False)

    config_path = os.path.join(config.save_path, "config.yml")
    shutil.copyfile(config.config, config_path)

    shutil.copytree('./models', os.path.join(config.save_path, "models"))
    shutil.copy('./train_models.py', config.save_path)
    train_model(model, optimizer, lr_scheduler, config, train_dataset, val_dataset, test_dataset, scaler)

    print("done!")