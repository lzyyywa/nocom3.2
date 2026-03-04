import torch.nn.functional as F
import torch.nn as nn
import torch
import math

from utils.lorentz import pairwise_dist, half_aperture, oxy_angle

def dist_to_origin(x, c):
    x_time = torch.sqrt(1.0 / c + torch.sum(x**2, dim=-1, keepdim=True))
    return torch.acosh(torch.clamp(torch.sqrt(c) * x_time, min=1.0 + 1e-5)) / torch.sqrt(c)

def safe_half_aperture(p, c, K=0.1):
    d = dist_to_origin(p, c)
    val = K / torch.clamp(torch.sinh(torch.sqrt(c) * d), min=1e-5)
    val_clamped = torch.clamp(val, min=-0.9999, max=0.9999)
    return torch.asin(val_clamped)

class HierarchicalEntailmentLoss(nn.Module):
    def __init__(self, K=0.1):
        super().__init__()
        self.K = K

    def forward(self, child, parent, c):
        with torch.cuda.amp.autocast(enabled=False):
            theta = oxy_angle(parent.float(), child.float(), curv=c.float()).unsqueeze(1)               
            alpha_parent = safe_half_aperture(parent.float(), c.float(), K=self.K).unsqueeze(1)
            loss_cone = F.relu(theta - alpha_parent)
            
        mask = ~torch.isnan(loss_cone)
        if mask.sum() > 0:
            return loss_cone[mask].mean()
        else:
            return torch.tensor(0.0, device=child.device, requires_grad=True)

class DiscriminativeAlignmentLoss(nn.Module):
    # 【终极修正】：恢复硬编码的 0.07 温度，确保模态对齐足够尖锐
    def __init__(self, temperature=0.07, hard_weight=3.0):
        super().__init__()
        self.temperature = temperature
        self.hard_weight = hard_weight
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, v_hyp, t_hyp, c, mask_pos, mask_hard):
        with torch.cuda.amp.autocast(enabled=False):
            dist = pairwise_dist(v_hyp.float(), t_hyp.float(), curv=c.float())
            logits = -dist / self.temperature
            
            if self.hard_weight > 1.0:
                logits = logits + mask_hard.float() * math.log(self.hard_weight)
                
            # 屏蔽假负例，防止左右互搏
            false_negatives = mask_pos & ~torch.eye(v_hyp.size(0), dtype=torch.bool, device=v_hyp.device)
            logits.masked_fill_(false_negatives, -1e9)
            
            labels = torch.arange(v_hyp.size(0), device=v_hyp.device)
            
        loss_v2t = self.criterion(logits, labels)
        loss_t2v = self.criterion(logits.t(), labels)
        return (loss_v2t + loss_t2v) / 2.0


def loss_calu(predict, target, config):
    batch_img, batch_verb, batch_obj, batch_pair, batch_coarse_verb, batch_coarse_obj = target
    batch_verb = batch_verb.cuda()
    batch_obj = batch_obj.cuda()
    
    c_pos = predict['c_pos']
    verb_logits = predict['verb_logits']
    obj_logits = predict['obj_logits']
    
    v_hyp = predict['v_hyp']                  
    o_hyp = predict['o_hyp']                  
    t_v_hyp = predict['t_v_hyp']              
    t_o_hyp = predict['t_o_hyp']              
    coarse_v_hyp = predict['coarse_v_hyp']    
    coarse_o_hyp = predict['coarse_o_hyp']    

    ce_loss_fn = nn.CrossEntropyLoss()
    # 恢复 0.07 的 DAL 温度
    dal_loss_fn = DiscriminativeAlignmentLoss(temperature=0.07, hard_weight=3.0)
    hem_loss_fn = HierarchicalEntailmentLoss(K=0.1)

    loss_cls_verb = ce_loss_fn(verb_logits, batch_verb)
    loss_cls_obj = ce_loss_fn(obj_logits, batch_obj)
    
    mask_verb = (batch_verb.unsqueeze(1) == batch_verb.unsqueeze(0))
    mask_obj = (batch_obj.unsqueeze(1) == batch_obj.unsqueeze(0))
    
    hard_verb = mask_obj & ~mask_verb
    # 调用时不再需要传入动态 temp
    loss_dal_verb = dal_loss_fn(v_hyp, t_v_hyp, c_pos, mask_pos=mask_verb, mask_hard=hard_verb)
    
    hard_obj = mask_verb & ~mask_obj
    loss_dal_obj = dal_loss_fn(o_hyp, t_o_hyp, c_pos, mask_pos=mask_obj, mask_hard=hard_obj)
    
    loss_dal = loss_dal_verb + loss_dal_obj

    loss_hem_v2fv = hem_loss_fn(child=v_hyp, parent=t_v_hyp, c=c_pos)
    loss_hem_fv2cv = hem_loss_fn(child=t_v_hyp, parent=coarse_v_hyp, c=c_pos)
    loss_hem_o2fo = hem_loss_fn(child=o_hyp, parent=t_o_hyp, c=c_pos)
    loss_hem_fo2co = hem_loss_fn(child=t_o_hyp, parent=coarse_o_hyp, c=c_pos)
    loss_hem = loss_hem_v2fv + loss_hem_fv2cv + loss_hem_o2fo + loss_hem_fo2co

    w_cls = getattr(config, 'w_cls', 3.0)
    w_dal = getattr(config, 'w_dal', 0.5)
    w_hem = getattr(config, 'w_hem', 0.1)

    total_loss = w_cls * (loss_cls_verb + loss_cls_obj) + \
                 w_dal * loss_dal + \
                 w_hem * loss_hem

    loss_dict = {
        'loss_cls_verb': loss_cls_verb.item(),
        'loss_cls_obj': loss_cls_obj.item(),
        'loss_dal': loss_dal.item(),
        'loss_hem': loss_hem.item()
    }

    return total_loss, loss_dict

class KLLoss(nn.Module):
    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        self.error_metric = error_metric

    def forward(self, prediction, label, mul=False):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label, 1)
        loss = self.error_metric(probs1, probs2)
        if mul:
            return loss * batch_size
        else:
            return loss

def hsic_loss(input1, input2, unbiased=False):
    pass

class Gml_loss(nn.Module):
    pass