import torch

def get_optimizer_vm(cfg,model):
    comp_param=[]
    video_en_param=[]
    for name, param in model.named_parameters():
        if 'video_encoder' in name:
            video_en_param.append(param)
        else:
            comp_param.append(param)
    optimizer = torch.optim.Adam([
        {'params': comp_param, 'lr': cfg.com_lr,'weight_decay': cfg.com_wd},
        {'params': video_en_param, 'lr': cfg.ve_lr,'weight_decay': cfg.ve_wd}],
        lr=cfg.ve_lr, eps=1e-8,weight_decay=cfg.ve_wd)  

    return optimizer

def get_optimizer_vlm(cfg,model):
    vision_no_wd=[]
    vision_with_wd=[]

    prompt_param = []
    c2c_with_wd=[]
    c2c_no_wd = []
    for name, param in model.named_parameters():
        if 'video_encoder' in name:
            if 'temporal_embedding' in name or 'ln_post' in name:
                vision_no_wd.append(param)
            elif 'Adapter' in name or 'clip_proj' in name:
                vision_with_wd.append(param)
                
    if cfg.text_encoding_manner=='composition':
        for name, param in model.named_parameters():
            if 'dfsp' in name:
                c2c_with_wd.append(param)
        optimizer = torch.optim.AdamW([
            {'params': model.prompt_learner.parameters(), 'lr': cfg.text_lr, 'weight_decay': cfg.text_wd},
            {'params': vision_with_wd, 'lr': cfg.visual_lr, 'weight_decay': cfg.visual_wd},
            {'params': vision_no_wd, 'lr': cfg.visual_lr, 'weight_decay': 0.0}, ],
            betas=(0.9, 0.999), lr=cfg.visual_lr, eps=1e-8,
            weight_decay=cfg.visual_wd) 
            
    elif cfg.text_encoding_manner=='component':
        for name, param in model.verb_prompt_learner.named_parameters():
            prompt_param.append(param)
        for name, param in model.obj_prompt_learner.named_parameters():
            if 'token_embedding' not in name:
                prompt_param.append(param)
        for name, param in model.named_parameters():
            # 【终极修正】：坚决把 scale 踢出去，只留 c2c 和 cls_temp，锁死双曲空间尺度！
            if 'c2c' in name or 'cls_temp' in name or name == 'c':
                c2c_with_wd.append(param)
                
        optimizer = torch.optim.AdamW([
            {'params':  prompt_param, 'lr': cfg.text_lr, 'weight_decay': cfg.text_wd},
            {'params': vision_with_wd, 'lr': cfg.visual_lr, 'weight_decay': cfg.visual_wd},
            {'params': vision_no_wd, 'lr': cfg.visual_lr, 'weight_decay': 0.0},

            {'params': c2c_with_wd, 'lr': cfg.visual_lr, 'weight_decay': cfg.visual_wd},
            {'params': c2c_no_wd, 'lr': cfg.visual_lr, 'weight_decay': 0.0},],
            betas=(0.9, 0.999), lr=cfg.visual_lr, eps=1e-8,
            weight_decay=cfg.visual_wd) 
    else:
        raise NotImplementedError
    return optimizer

def get_optimizer(cfg,model):
    if cfg.framework=='vm':
        return get_optimizer_vm(cfg,model)
    elif cfg.framework=='vlm':
        return get_optimizer_vlm(cfg,model)