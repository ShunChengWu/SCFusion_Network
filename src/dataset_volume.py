if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import random
from config import Config

def scene_model_id_pair(path, dataset_portion=1.0, shuffle=False):
    '''
    Load sceneId, model names
    '''

    scene_name_pair = []  # full path of the objs files

    model_path = os.path.abspath(path)
#    models = os.listdir(model_path)

    foo = [1]
    for root, dirs, files in os.walk(os.path.abspath(model_path)):
        diff = root[len(model_path)::]
        folder_name = ''
        for file in files:
            if(os.path.isdir(file)):
                folder_name = file
            else:
                scene_name_pair.extend([(model_path, os.path.join(diff,folder_name, file)) for file__ in foo])
     
    if shuffle is True:
        random.shuffle(scene_name_pair)

    num_models = len(scene_name_pair)
    portioned_scene_name_pair = scene_name_pair[int(num_models *
                                                    (1-dataset_portion)):]
    if not shuffle is True:
        portioned_scene_name_pair = sorted(portioned_scene_name_pair)
    return portioned_scene_name_pair

def checkEnd(x):
    return x + '/' if x[-1] != '/' else x

def checkStart(x):
    return x[1:] if x[0] == '/' else x

class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_base_folders:list, folder_names:list, data_portion:float, shuffle=False):
        super(Dataset, self).__init__()
        self.folder_names = folder_names
        self.shuffle = shuffle
   
        if len(input_base_folders) == 0:
            raise RuntimeError('input base folder has size 0')
        
        paths = dict()
        for name in folder_names:
            paths[name] = list()
        
        for base_folder in input_base_folders:
            data = scene_model_id_pair(os.path.join(base_folder, folder_names[0]), data_portion)
            for d in data:
                for name in folder_names:    
                    paths[name].append( checkEnd(base_folder) + name + d[1] )
        self.len = len(paths[folder_names[0]])
        self.paths = paths
        
    def __len__(self):
        return self.len 

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index][0])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name[1])

    def load_item(self, index):
        # [print('load path:',self.paths[name][index]) for name in self.folder_names]
        return [torch.from_numpy(np.load(self.paths[name][index])) for name in self.folder_names]

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True,
                shuffle=self.shuffle,
            )

            for item in sample_loader:
                yield item
    
    

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from config import Config
    
    config = Config('../config.yml.example')
    # config.DATASET_PORTION = 0.2
    dataset = Dataset(config, config.TRAIN_BASE_FOLDERS, config.SUBFOLDERS)
    
    items = dataset.__getitem__(0)
    volume = items[0]
    gt = items[1]
    mask = items[2] if len(items) >2 else None
    if mask is not None:
        print('volume', volume.shape, 'gt',gt.shape, 'mask', mask.shape)
    else:
        print('volume', volume.shape, 'gt',gt.shape)
    
    train_loader = DataLoader(
            dataset=dataset,
            batch_size=2,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )
    
    print('Go through all data...')
    for items in train_loader:
        volume, gt, mask = items
    print('done!')
