import os
import yaml

class Config(dict):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.load(self._yaml, Loader=yaml.FullLoader)
            self._dict['PATH'] = os.path.dirname(config_path)

    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]

        if DEFAULT_CONFIG.get(name) is not None:
            return DEFAULT_CONFIG[name]

        return None

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------')
        print('')

# TODO: add new variablesto default config 
DEFAULT_CONFIG = {
    'MODE': 1,                      # 1: train, 2: test, 3: eval
    'GATED': 0,		                # 0: vanilla conv 2: Gated conv
    'MASK': 3,                      # 1: random block, 2: half, 3: external, 4: (external, random block), 5: (external, random block, half)
    'SEED': 10,                     # random seed
    'GPU': [0],                     # list of gpu ids
    'DEBUG': 0,                     # turns on debugging mode
    'VERBOSE': 0,                   # turns on verbose mode in the output console
    'CLASS_NUM': 14,    
    'DISCRIMINATIVE': 1,            # use discriminator
    'PRED_DF': 0,                   # use predict distance field branch
    'PRED_COM': 0,                  # use predict completion branch
    'PRED_SEM': 0,                  # use predict semantic branch
    
    'TRAIN_BASE_FOLDERS': [],       # base training folder 
    'TEST_BASE_FOLDERS': [],        # base testing folder 
    'VALID_BASE_FOLDERS': [],       # base validation folder 
    'SUBFOLDERS': ['train','gt','mask'], # subfolders will be appended to base folder
    'DATASET_PORTION': 1.0,          # the portion of data to use

    'LR_G': 0.0001,                  # learning rate for generator
    'LR_D': 0.0001,                  # learning rate for discriminator
    
    'BETA_G': 0.5,                   # adam optimizer beta1 for generators
    'BETA_D': 0.5,                   # adam optimizer beta1 for discriminators
    'BETA2': 0.9,                   # adam optimizer beta2
    'BATCH_SIZE': 8,                # input batch size for training
    'BATCH_FACTOR': 1,              # Increase batch size by this factor by accumulate prediction multiple times and do gradient.  https://research.fb.com/wp-content/uploads/2017/06/imagenet1kin1h5.pdf
    'MAX_ITERS': 2e6,               # maximum number of iterations to train the model
    'PADDING': 'zeros',             # zeros, circular, constant, replication
    'PAD_VALUE': 0,                  # only used when padding mode is constant

    'MASK_LOSS': 0,                 # mask out the loss from unknown regions
    'GAN_LOSS': 'nsgan',            # nsgan | lsgan | hinge
    'INPAINT_ADV_LOSS_WEIGHT': 0.01,# adversarial loss weight

    'SAVE_INTERVAL': 1000,          # how many iterations to wait before saving model (0: never)
    'SAMPLE_INTERVAL': 1000,        # how many iterations to wait before sampling (0: never)
    'SAMPLE_SIZE': 12,              # number of images to sample
    'EVAL_INTERVAL': 0,             # how many iterations to wait before model evaluation (0: never)
    'LOG_INTERVAL': 10,             # how many iterations to wait before logging training status (0: never)
}

if __name__ == '__main__':
    pass
