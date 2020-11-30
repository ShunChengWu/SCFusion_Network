import os
import random
import numpy as np
import torch
import argparse
from shutil import copyfile

if __name__ == '__main__':
    os.sys.path.append('./src')

from src.config import Config
from src.SceneInpainting import SceneInpainting

def main(mode=None):
    r"""starts the model

    Args:
        mode (int): 1: train, 2: test, 3: eval 4: trace 5: sample, reads from config file if not specified
    """

    config = load_config(mode)

    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    

    # init device
    if torch.cuda.is_available() and len(config.GPU) > 0:
        # torch.cuda.set_device(config.GPU[0]) # set default device
        config.DEVICE = torch.device("cuda")
        # config.DEVICE = torch.device("cuda:" + str(config.GPU[0]))
    else:
        config.DEVICE = torch.device("cpu")

    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False   # cudnn auto-tuner
    torch.backends.cudnn.deterministic=True
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    if config.MODE == 4: # trace
        config.GPU = [config.GPU[0]] # disable parallel in tracing
    model = SceneInpainting(config)
    
    # model training
    if config.MODE == 1:
        model.load()
        config.print()
        print('\nstart training...\n')
        model.train()
        
    # model test
    elif config.MODE == 2:
        if model.load(config.LOADBEST) is False:
            raise Exception('\nCannot find saved model!\n')
        print('\nstart testing...\n')
        model.test()

    # trace mode
    elif config.MODE == 3:
        print('Trace model to ScriptModule')
        #os.environ['CUDA_VISIBLE_DEVICES'] = str(config.GPU[0]) # disable parallel in tracing
        
        
        if model.load(config.LOADBEST) is False:
            raise Exception('\nCannot find saved model!\n')
        print('\nstart tracing...\n')
        model.trace()
    # sample
    elif config.MODE == 4:
        print('Trace model to ScriptModule')
        if model.load(config.LOADBEST) is False:
            raise Exception('\nCannot find saved model!\n')
        print('Sample!\n')
        model.sample(it=None, save=True)

def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml', help='configuration file name. Relative path under given path (default: config.yml)')
    parser.add_argument('--loadbest', type=int, default=1,choices=[0,1], help='1: load best model or 0: load checkpoints. Only works in non training mode.')

    # test mode
    if mode == 2:
        parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
        parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')
        parser.add_argument('--output', type=str, help='path to the output directory')

    args = parser.parse_args()
    config_path = os.path.abspath(args.config)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        # raise RuntimeError('Targer config file does not exist. {}' & config_path)
        copyfile('./config.yml.example', config_path)

    # load config file
    config = Config(config_path)
    
    config.LOADBEST = args.loadbest

    # train mode
    if mode is not None:
        config.MODE = mode
    
    return config
    
if __name__ == "__main__":
    main()
