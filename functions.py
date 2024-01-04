def get_lr(optimizer):
    for g in optimizer.param_groups:
        return g['lr']

def set_lr(optimizer, new_lr):
    for g in optimizer.param_groups:
        g['lr'] = new_lr