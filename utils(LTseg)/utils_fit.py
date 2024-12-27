from nets.unet_training import CE_Loss, Dice_loss, focal_loss, Boudaryloss
from tqdm import tqdm
import torch.nn.functional as F
import os

import torch
import numpy as np
import torch.nn as nn
from utils.utils import get_lr
from utils.utils_metrics import f_score
from utils.dataloader import augmentationimage as ugmentationimage
from utils.TI_loss import TI_Loss


def trainDice_loss(inputs, targets, criticals_map, smooth=0.00001):
    inputs = torch.sigmoid(inputs)

    a, b, c, d = inputs.size()
    sums = []
    for i in range(a):
        for j in range(b):
            img = inputs[i, j, :, :]
            label = targets[i, j, :, :]
            map = criticals_map[i, j, :, :]
            img = img * map * 0.3 + img
            label = label * map * 0.3 + label
            intersection = (img * label).sum()
            sums.append(1 - (2. * intersection + smooth) / (img.sum() + label.sum() + smooth))
    return sum(sums) / len(sums)

class focalpixel(nn.Module):
    def __init__(self, gamma=2, alpha=0.75):
        super(focalpixel, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, inputs, target):
        pred_sigmoid = inputs.sigmoid()
        target = target.type_as(inputs)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(inputs, target, reduction='none') * focal_weight
        return loss


def ACE_Loss(inputs, target):
    CE_loss = nn.BCEWithLogitsLoss()(inputs.float(), target.float())
    return CE_loss


def cepixel(inputs, target):
    CE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs.float(), target.float())
    return CE_loss


def dicepixel(inputs, targets, smooth=0.00001):
    inputs = torch.sigmoid(inputs)
    intersection = inputs * targets
    dice_loss_per_pixel = 1 - (2. * intersection + smooth) / (inputs + targets + smooth)
    return dice_loss_per_pixel



def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, loss_fuc, num_classes, save_dir, no_improve_count):
    total_loss = 0
    val_loss = 0
    val_f_score = 0


    pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3, ascii=True)

    model_train.train()

    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs = batch
        # print(imgs.numpy().shape,pngs.numpy().shape)
        # imgs , pngs = ugmentationimage(imgs,pngs)
        with torch.no_grad():
            imgs = imgs.cuda()  # [bsz, 3, 448, 448]
            pngs = pngs.cuda()  # tragets

        optimizer.zero_grad()
        outputs = model_train(imgs)  # [bsz, 24, 448, 448]

        # ----------------------------------
        # choose the loss fuc
        # ----------------------------------
        if loss_fuc == "BCEloss":
            loss = CE_Loss(outputs, pngs)

        elif loss_fuc == "Diceloss":
            loss = Dice_loss(outputs, pngs)

        elif loss_fuc == "TPCloss":


           

            # 计算TPCloss
            loss = trainDice_loss(outputs, pngs, criticals_map)
            # print(loss.shape)
            # 用 total_loss 进行反向传播和优化
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # total_f_score   += _f_score.item()

        pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                            'lr': get_lr(optimizer)})
        pbar.update(1)
    pbar.close()

    print('Finish Train')
    print('Start Validation')
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3, ascii=True)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
 
        imgs, pngs = batch
 
        with torch.no_grad():
            imgs = imgs.cuda()  # [bsz, 3, 448, 448]
            pngs = pngs.cuda()  # tragets

            outputs = model_train(imgs)
            # ----------------------------------
            # choose the loss fuc
            # ----------------------------------
            if loss_fuc == "BCEloss":
                loss = CE_Loss(outputs, pngs)
            elif loss_fuc == "Diceloss":
                loss = Dice_loss(outputs, pngs)
            elif loss_fuc == "TPCloss":
                loss = Dice_loss(outputs, pngs)
            val_loss += loss.item()


        pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                            'f_score': val_f_score / (iteration + 1),
                            'lr': get_lr(optimizer)})
        pbar.update(1)
    pbar.close()

    print('Finish Validation')
    loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)

    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    if epoch > 100 and epoch % 50 == 0:

        save_path = os.path.join(save_dir, f"epoch_{epoch}_weights.pth")
        torch.save(model.state_dict(), save_path)
    torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

    no_improve_count += 1
    if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        print('Save best model to best_epoch_weights.pth')
        torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
        no_improve_count = 0

    return no_improve_count
