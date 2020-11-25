import torch
import os
import numpy as np
from DataLoader import CustomDataLoader
from dataset_volume import Dataset
from config import Config
from utils import Progbar, create_dir, formatString, get_label_name_list
from metrics_ssc import Metric_IoU, Metric_PR, Metric_F1
from torch.utils.tensorboard import SummaryWriter
from models_scene_inpainting_forknet import SceneInpaintingForknetModel
from meshing import occupancy_meshing, write_ply

class SceneInpainting():
    def __init__(self, config):
        self.config = config
        model_name = self.config.NAME if not self.config.NAME is None else 'SceneInapinting'

        self.model_name = model_name
        self.model = SceneInpaintingForknetModel(config,model_name).to(config.DEVICE)
        
        self.metric_iou = Metric_IoU()
        self.metric_pr = Metric_PR()

        self.train_dataset = Dataset(config.TRAIN_BASE_FOLDERS, config.SUBFOLDERS, config.DATASET_PORTION)        
        
        if config.MODE == 2: 
            # build test set
            self.val_dataset = Dataset(config.TEST_BASE_FOLDERS, config.SUBFOLDERS, config.DATASET_PORTION)
        else:
            # build val set
            self.val_dataset = Dataset(config.VALID_BASE_FOLDERS, config.SUBFOLDERS, config.DATASET_PORTION, 
                                   shuffle = False)
        self.sample_iterator = self.val_dataset.create_iterator(1)#config.SAMPLE_SIZE

        self.samples_path = os.path.join(config.PATH, model_name, 'samples')
        self.results_path = os.path.join(config.PATH, model_name, 'results')
        
        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + self.model_name + '.dat')
        
        if self.config.MODE == 1:
            self.writter = SummaryWriter(  #TODO: maybe make logging of a model with same name to be at one file
                os.path.join(config.PATH, "logs", model_name))
            print('log path: ', config.PATH)
        else:
            self.writter = None
        # import sys
        # sys.exit()
        
    def load(self, best=False):
        return self.model.load(best)

    def save(self):
        self.model.save()

    def train(self):
        drop_last =True
        train_loader = CustomDataLoader(
            config = self.config,
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=drop_last,
            shuffle=True,
            #worker_init_fn=lambda worker_id: np.random.seed(config.SEED+worker_id)
        )
        
        epoch = 1
        keep_training=True
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = int(len(self.train_dataset)/self.config.BATCH_SIZE)*self.config.BATCH_SIZE if drop_last is True else  len(self.train_dataset) 
        
        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return
        
        if self.model.iteration >= max_iteration:
            keep_training = False
            print('Read maximum training iteration (',max_iteration,').')

        progbar = Progbar(total, width=20, stateful_metrics=['Misc/epo', 'Misc/it'])
        loader = iter(train_loader)
        # import sys
        # sys.exit()
        if self.model.iteration > 0:
            print('\n Resuming dataloader to last iteration...')
            iteration=0
            while True:                
                iter_local = 0
                for idx in loader.IndexIter():
                    progbar.add(self.config.BATCH_SIZE, silent=True)
                    iter_local += 1
                    iteration += 1
                    if iteration == self.model.iteration:
                        break
                if iteration == self.model.iteration:
                        break
                epoch+=1
                progbar = Progbar(total, width=20, stateful_metrics=['Misc/epo', 'Misc/it'])
                loader = iter(train_loader)
        iou=precision=recall=-1
        while(keep_training):
            print('\n\nTraining epoch: %d' % epoch)
            for items in loader:
                self.model.train()
                with torch.no_grad():
                    items= self.cuda(*items)
                    volume = items[0]
                    gt = items[1].long()
                    mask = items[2] if len(items) == 3 else None
                
                optimized, logs, pred  = self.model.process(volume,gt,mask)
                
                if optimized or iou == -1:
                    ious, *_= self.metric_iou(pred, gt, self.config.CLASS_NUM)
                    precisions, recalls, *_ = self.metric_pr(pred, gt, self.config.CLASS_NUM)    
                    iou = ious.detach().mean().item()
                    precision = precisions.detach().mean().item()
                    recall = recalls.detach().mean().item()
                    
                # calculate metrics                                
                logs.append(("IoU/mean", iou))
                logs.append(("Precision/mean", precision))
                logs.append(("Recall/mean", recall))
                logs.append(("F1Score/mean", Metric_F1(precision,recall)))
                
                # logs, com_dec, full_dec = self.model.process(volume,gt)
                iteration = self.model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break
                
                logs = [
                    ("Misc/epo", int(epoch)),
                    ("Misc/it", int(iteration)),
                ] + logs
                
                progbar.add(len(volume), values=logs \
                            if self.config.VERBOSE else [x for x in logs if not x[0].startswith('Loss')])
                
                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    for i in range(self.config.CLASS_NUM):
                        iou = ious[i].item()
                        precision = precisions[i].item()
                        recall = recalls[i].item()
                        
                        label_name_list = get_label_name_list(self.config.CLASS_NUM)
                        
                        name = 'IoU/' + str(i) + '_' + label_name_list[i]
                        logs.append((name, iou))
                        name = 'Precision/' + str(i) + '_' + label_name_list[i]
                        logs.append((name, precision))
                        name = 'Recall/' + str(i) + '_' + label_name_list[i]
                        logs.append((name, recall))
                        name = 'F1Score/' + str(i) + '_' + label_name_list[i]
                        
                            
                        logs.append((name, Metric_F1(precision,recall) ))
                    self.log(logs, iteration)

                # # sample model at checkpoints
                # if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                #     self.sample()

                # # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0:
                    print('\nstart eval...\n')
                    self.eval()

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()
            epoch+=1
            progbar = Progbar(total, width=20, stateful_metrics=['Misc/epo', 'Misc/it'])
            loader = iter(train_loader)
            
        if self.config.DEBUG == 0:
            if self.config.EVAL_INTERVAL:
                # self.eval()
                self.test()
            if self.config.SAVE_INTERVAL:
                self.save()
        print('\nDone!')
        
    def eval(self, write_log = True, write_file = False):
        self.model.eval()
        
        val_loader = CustomDataLoader(
            config = self.config,
            dataset=self.val_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )
        
        total = len(self.val_dataset)
        progbar = Progbar(total, width=20, stateful_metrics=['Misc/it'])
        
        classes = "{:>5.5}\t".format('Metrics')
        label_name_list = get_label_name_list(self.config.CLASS_NUM)        
        for name in label_name_list:
            classes += '{:>5.5}\t'.format(name)
        classes += '{:>5.5}'.format('Mean')
        
        means = dict()
        sums = dict()
        k_iou_mean = ('IoU', 'TEST_IoU/mean')
        k_pre_mean = ('Prec', 'TEST_Precision/mean')
        k_rec_mean = ('Recall', 'TEST_Recall/mean')
        k_f1_mean  = ('F1', 'TEST_F1Score/mean')
        if self.config.MASK > 0:
            k_iou_known   = ('IoUK','TEST_IoU/ssc_known')
            k_iou_unknown = ('IoUUK', 'TEST_IoU/ssc_unknown')
            k_rec_known   = ('RecK', 'TEST_Recall/ssc_known')
            k_rec_unknown  = ('RecUK', 'TEST_Recall/ssc_unknown')
            k_pre_known   = ('PrecK', 'TEST_Precision/ssc_known')
            k_pre_unknown = ('PrecUK', 'TEST_Precision/ssc_unknown')
            k_f1_known    = ('F1K', 'TEST_F1Score/ssc_known')
            k_f1_unknown  = ('F1UK', 'TEST_F1Score/ssc_unknown')
            
        sums[k_iou_mean[1]] = torch.zeros(self.config.CLASS_NUM)
        sums[k_pre_mean[1]] = torch.zeros(self.config.CLASS_NUM)
        sums[k_rec_mean[1]] = torch.zeros(self.config.CLASS_NUM)
        sums[k_f1_mean[1]]  = torch.zeros(self.config.CLASS_NUM)
        if self.config.MASK > 0:
            sums[k_iou_known[1]] = torch.zeros(self.config.CLASS_NUM)
            sums[k_iou_unknown[1]] = torch.zeros(self.config.CLASS_NUM)
            sums[k_rec_known[1]] = torch.zeros(self.config.CLASS_NUM)
            sums[k_rec_unknown[1]] = torch.zeros(self.config.CLASS_NUM)
            sums[k_pre_known[1]] = torch.zeros(self.config.CLASS_NUM)
            sums[k_pre_unknown[1]] = torch.zeros(self.config.CLASS_NUM)
            sums[k_f1_known[1]]  = torch.zeros(self.config.CLASS_NUM)
            sums[k_f1_unknown[1]]  = torch.zeros(self.config.CLASS_NUM)
        # sums_per_class = torch.zeros([self.config.CLASS_NUM])
        # norm_per_class = torch.zeros([self.config.CLASS_NUM])
        
        logs = []
        counter = 0
        for items in val_loader:
            items= self.cuda(*items)
            volume = items[0]
            gt = items[1].long()
            mask = items[2] if len(items) == 3 else None
            pred = self.model(volume,mask)
            
            # for i in range(self.config.CLASS_NUM):
            #     sums_per_class[i] += (gt == i).sum()
            
            # metrics
            pred = pred.detach()                
            ious, inters, unions, *_ = self.metric_iou(pred, gt, self.config.CLASS_NUM)
            precisions, recalls, *_ =  \
                self.metric_pr(pred, gt, self.config.CLASS_NUM)
            
            sums[k_iou_mean[1]] += ious
            sums[k_pre_mean[1]] += precisions
            sums[k_rec_mean[1]] += recalls
            sums[k_f1_mean[1]]  += Metric_F1(precisions, recalls)

            iou = ious.mean().item()
            precision = precisions.mean().item()
            recall = recalls.mean().item()
            f1 = Metric_F1(precision, recall)
            means['IoU/ssc'] = iou
            means['Precision/ssc'] = precision
            means['Recall/ssc'] = recall
            means['F1/ssc'] = f1
            # logs.append(("f1Score", f1))
            
            counter += 1

            if self.config.MASK > 0:
                # Known Region
                ious_known, inters_known, unions_known, *_ = \
                    self.metric_iou(pred, gt, self.config.CLASS_NUM, mask)
                ious_unknown, inters_unknown, unions_unknown, *_ = \
                    self.metric_iou(pred, gt, self.config.CLASS_NUM, torch.logical_not(mask))
                precisions_known, recalls_known, *_ = \
                    self.metric_pr(pred, gt, self.config.CLASS_NUM, mask)
                precisions_unknown, recalls_unknown, *_ = \
                    self.metric_pr(pred, gt, self.config.CLASS_NUM, torch.logical_not(mask))
                    
                means['IoU/known']    = ious_known.mean().item()
                means['IoU/unknown']  = ious_unknown.mean().item()
                means['Prec/known']   = precisions_known.mean().item()
                means['Prec/unknown'] = precisions_unknown.mean().item()
                means['Rec/known']    = recalls_known.mean().item()
                means['Rec/unknown']  = recalls_unknown.mean().item()
                means['F1/known']     = Metric_F1(precisions_known.mean().item(), recalls_known.mean().item())
                means['F1/unknown']   = Metric_F1(precisions_unknown.mean().item(), recalls_known.mean().item())

                sums[k_iou_known[1]] += ious_known
                sums[k_pre_known[1]] += precisions_known
                sums[k_rec_known[1]] += recalls_known
                sums[k_f1_known[1]]  += Metric_F1(precisions_known, recalls_known)
                
                sums[k_iou_unknown[1]] += ious_unknown
                sums[k_pre_unknown[1]] += precisions_unknown
                sums[k_rec_unknown[1]] += recalls_unknown
                sums[k_f1_unknown[1]]  += Metric_F1(precisions_unknown, recalls_unknown)
                
            for key, item in means.items():
                logs += [(key, item)]
            progbar.add(len(volume), values=logs if self.config.VERBOSE else [x for x in logs if not x[0].startswith('Loss')])
            # break  
                 
        # Calculate the average of metrics of each class
        for key, item in sums.items():
            sums[key] = (item + 1e-12) / (counter + 1e-12)
            
        # Calculate weights for each class base on per class frequency 
        # for i in range(self.config.CLASS_NUM):
        # norm_per_class = torch.abs(1.0 / (torch.log(sums_per_class)+1))
        # print('sums_per_class\n',sums_per_class)
        # print('norm_per_class\n',norm_per_class)
            
        logs = []
        means = dict()
        
        for key, value in sums.items():
            means[key] = value
            # means[key+'w'] = value * norm_per_class

        getString = lambda means, name_pair: "{}:\t{}".format(name_pair[0], formatString(means,name_pair[1]) )
        # getString_w = lambda means, name_pair: "{}:\t{}".format(name_pair[0]+'w', formatString(means,name_pair[1]+'w') )

        print('')
        print(classes)
        print(getString(means,k_iou_mean))
        print(getString(means,k_pre_mean))
        print(getString(means,k_rec_mean))
        print(getString(means,k_f1_mean))
        
        # print(getString_w(means,k_iou_mean))
        if self.config.MASK > 0:
            print(getString(means, k_iou_known))
            print(getString(means, k_pre_known))
            print(getString(means, k_rec_known))
            print(getString(means, k_f1_known))
            print(getString(means, k_iou_unknown))
            print(getString(means, k_pre_unknown))
            print(getString(means, k_rec_unknown))
            print(getString(means, k_f1_unknown))
            
        if write_file:
            create_dir(self.results_path)
            with open(os.path.join(self.results_path, 'result.txt'), 'w+') as f:
                f.write('{}\n'.format(classes))
                f.write('{}\n'.format(getString(means,k_iou_mean)))
                f.write('{}\n'.format(getString(means,k_pre_mean)))
                f.write('{}\n'.format(getString(means,k_rec_mean)))
                f.write('{}\n'.format(getString(means,k_f1_mean)))
                if self.config.MASK > 0:
                    f.write('{}\n'.format(getString(means, k_iou_known)))
                    f.write('{}\n'.format(getString(means, k_pre_known)))
                    f.write('{}\n'.format(getString(means, k_rec_known)))
                    f.write('{}\n'.format(getString(means, k_f1_known)))
                    f.write('{}\n'.format(getString(means, k_iou_unknown)))
                    f.write('{}\n'.format(getString(means, k_pre_unknown)))
                    f.write('{}\n'.format(getString(means, k_rec_unknown)))
                    f.write('{}\n'.format(getString(means, k_f1_unknown)))

        for key, item in means.items():
            logs += [(key, item.mean().item())]

        if self.config.LOG_INTERVAL and write_log:
            for i in range(self.config.CLASS_NUM):
                iou = sums[k_iou_mean[1]][i].item()
                precision = sums[k_pre_mean[1]][i].item()
                recall = sums[k_rec_mean[1]][i].item()
                
                label_name_list = get_label_name_list(self.config.CLASS_NUM)
                name = 'TEST_IoU/' + str(i) + '_' + label_name_list[i]
                logs.append((name, iou))
                name = 'TEST_Precision/' + str(i) + '_' + label_name_list[i]
                logs.append((name, precision))
                name = 'TEST_Recall/' + str(i) + '_' + label_name_list[i]
                logs.append((name, recall))
                name = 'TEST_F1Score/' + str(i) + '_' + label_name_list[i]
                logs.append(( name, Metric_F1(precision,recall) ))
            # import sys
            # sys.exit()
            self.log(logs, self.model.iteration)
        return logs, means
          
    def test(self):
        self.eval(write_log = False, write_file =  True)  
        print('\nEnd test....')
    
    
    
        
    def sample(self, it=None, save=False):
        # do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            raise RuntimeError('To sample, val_dataset must be given.')
        self.model.eval()
        
        indices = [i for i in range(len(self.val_dataset))]
        indices = np.array(indices)
        np.random.shuffle(indices)

        iteration=0
        if self.config.SAMPLE_SIZE >0:
            iter_max = self.config.SAMPLE_SIZE 
        else:
            iter_max = 100
            
        for i in range(iter_max):
            # items = next(self.sample_iterator)
            items = self.val_dataset.__getitem__(indices[i])
            items = [np.expand_dims(item, axis=0) for item in items]
            items = [torch.from_numpy(item) for item in items]
            
            cuda_items = self.cuda(*items)
            # print('type(cuda_items)',type(cuda_items))
            volume = cuda_items[0]
            gt = cuda_items[1]
            mask = cuda_items[2] if len(items) == 3 else None
    
            # iteration = self.model.iteration
            # if it is not None:
            #     iteration = it
            
            # if self.config.MASK > 0:
            #     inputs = (volume * torch.logical_not(mask))
            # else:
            inputs = volume
            outputs = self.model(volume, mask)
            
            outputs_merged = torch.argmax(outputs, dim=1)
            
            # if self.config.MASK > 0:
            #     outputs_merged[masks > 0] = 0
            
            path = os.path.join(self.samples_path, self.model_name)
            create_dir(path)
            
            for b in range(gt.size(0)):
                def Meshing(path, volume, label, threshold):
                    vertices, faces, colors = occupancy_meshing(self.config.CLASS_NUM, volume[b,:,:,:],
                                                                label[b,:,:,:],threshold=threshold)
                    if save:
                        print('saving...')
                        # print('vertices.shape:',vertices.shape)
                        # print('faces.shape:',faces.shape)
                        # print('colors.shape:',colors.shape)
                        write_ply(path,vertices,faces,colors)
                        print('\nsaving sample ' + path)
                    return vertices, faces, colors
                
                try:
                    name = os.path.join(path, str(iteration).zfill(5) + '_' + str(b) + "_pd.ply")
                    vertices_pd, faces_pd, colors_pd = Meshing(name, outputs_merged, outputs_merged,0.5)
                    
                    # name = os.path.join(path, str(iteration).zfill(5) + '_' + str(b) + "_mask.ply")
                    # vertices_pd, faces_pd, colors_pd = Meshing(name, mask, mask,0.5)
                    
                    name = os.path.join(path, str(iteration).zfill(5) + '_' + str(b) + "_in.ply")
                    vertices_in, faces_in, colors_in = Meshing(name,inputs, torch.zeros_like(inputs).long(),0.0)
                    
                    name = os.path.join(path, str(iteration).zfill(5) + '_' + str(b) + "_gt.ply")
                    vertices_gt, faces_gt, colors_gt = Meshing(name,gt, gt,0.5)
                except:
                    print('error occured when computing meshes at',b)
                    
            iteration+=1
            print('iteration:',iteration)
        return 
        
    def log(self, logs, iteration):
        # # Binary
        # with open(self.log_file, 'a') as f:
        #     f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))
            
        # Tensorboard
        if self.writter is not None:
            for i in logs:
                if not i[0].startswith('Misc'):
                    self.writter.add_scalar(i[0], i[1], iteration)

    def cuda(self, *args):
        return [item.to(self.config.DEVICE) for item in args]

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()
    
    def test_run(self):
        batch=2
        # class_num=14
        volume = torch.rand(batch,64,64,64)
        full_gt = torch.randint(0, self.config.CLASS_NUM,size=(batch,64,64,64))
        volume = volume.to(self.config.DEVICE)
        full_gt = full_gt.to(self.config.DEVICE)
        max_iter=1e3
        iter=0
        while iter < max_iter:
            logs = self.model.process(volume, full_gt)
            print(logs)
            iter+=1
            
    def trace(self):
        self.model.eval()
        volume = torch.rand(1,64,64,64)
        # full_gt = torch.randint(0, self.class_num,size=(1,64,64,64))
        mask = (torch.rand(1,64,64,64) > 0.3).float()
        volume = volume.to(self.config.DEVICE)
        mask = mask.to(self.config.DEVICE)
        
        traced_script_module = torch.jit.trace(self.model, (volume, mask))
        
        saving_pth = os.path.join(self.config.PATH,self.model.name,self.model.name+'.pt')
        
        traced_script_module.save(saving_pth)
        print("Model saved to {}!".format(saving_pth))
    
if __name__ == '__main__':    
    pass
