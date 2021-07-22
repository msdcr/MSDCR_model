
'''Some handy functions definition'''
from train import args
import torch

def save_checkpoint(model,model_dir):
    torch.save(model.state_dict(),model_dir)

def resume_checkpoint(model,model_dir,device_id):
    state_dict=torch.load(model_dir,
                          map_location=lambda storage,loc:storage.cuda(device=device_id))
    model.load_state_dict(state_dict)

## Hyper parameters

def use_cuda(enabled,device_id=0):
    if enabled:
        assert torch.cuda.is_available(), torch.cuda.set_device(device_id)

def use_optimizer(network, params):
    if params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), lr=params['lr'])
    elif params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(), lr=params['lr'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(), lr=params['lr'])

    return optimizer

