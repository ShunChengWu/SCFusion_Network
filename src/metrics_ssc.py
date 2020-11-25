import torch
import torch.nn as nn
import numpy as np 


# NYU14_name_list = ['Unknown', 'Bed', 'Books', 'Ceiling', 'Chair',
#                   'Floor', 'Furniture', 'Objects', 'Picture',
#                   'Sofa', 'Table', 'TV', 'Wall', 'Window'
#                  ]
# Label11_name_list = ["None", "Ceiling", "Floor", "Wall", "Window",
#                      "Chair", "Bed", "Sofa", "Desk","TV","Furniture","Objects"]

	
class Metric_IoU_base(nn.Module):
    """
    Given two [n,class] tensors, calculating per class IoU and average IoU
    """
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-6
        self.register_buffer('output', torch.ones([], requires_grad=False))
        self.register_buffer('intersections', torch.ones([], requires_grad=False))
        self.register_buffer('unions', torch.ones([], requires_grad=False))
        # self.register_buffer('valid', torch.ones([], requires_grad=False))
    def __call__(self, pred, gt):
        class_num = pred.size(-1)
        self.output = torch.ones([])
        self.valid = torch.ones([])
        
        self.intersections = torch.zeros([class_num])
        self.unions = torch.zeros([class_num])
        self.output = torch.zeros([class_num])
        # self.valid = torch.zeros([class_num])

        for class_n in range(class_num):
            pred_c = pred[:, class_n].flatten()
            gt_c   = gt[:, class_n].flatten()            
            intersection = (pred_c & gt_c).float().sum() # if one of them is 0: 0
            union = (pred_c | gt_c).float().sum() # if both 0: 0        
            iou = (intersection+self.epsilon)/(union+self.epsilon)

            
            iou = iou.unsqueeze(-1)
            intersection = intersection.unsqueeze(-1)
            union = union.unsqueeze(-1)
            # valid = gt_c.sum() > 0
            # valid = valid.unsqueeze(-1)
            
            self.intersections[class_n] = intersection
            self.unions[class_n] = union
            self.output[class_n] = iou
            # self.valid[class_n] = valid

        return self.output, self.intersections, self.unions#, self.valid
    
class Metric_PR_Base(nn.Module):
    """
    Given two [n,class] tensors, calculating per class accuracy
    """
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-6
        self.register_buffer('precisions', torch.ones([], requires_grad=False))
        self.register_buffer('recalls', torch.ones([], requires_grad=False))
        # self.register_buffer('valid', torch.ones([], requires_grad=False))
        self.register_buffer('intersection', torch.ones([], requires_grad=False))
        self.register_buffer('gt_sum', torch.ones([], requires_grad=False))
        self.register_buffer('pred_sum', torch.ones([], requires_grad=False))
    def __call__(self, pred, gt):
        class_num = pred.size(-1)
        self.precisions = torch.zeros([class_num])
        self.recalls = torch.zeros([class_num])
        # self.valid = torch.zeros([class_num])
        self.intersection = torch.zeros([class_num])
        self.gt_sum = torch.zeros([class_num])
        self.pred_sum = torch.zeros([class_num])

        for class_n in range(class_num):
            pred_c = pred[:, class_n].flatten()
            gt_c   = gt[:, class_n].flatten()            
            intersection = (pred_c & gt_c).float().sum() # if one of them is 0: 0
            gt_sum = gt_c.sum()
            pred_sum = pred_c.sum()
            r = (intersection+self.epsilon)/(gt_sum+self.epsilon)
            p = (intersection+self.epsilon)/(pred_sum+self.epsilon)

            r= r.unsqueeze(-1)
            p= p.unsqueeze(-1)
            # valid = gt_c.sum() > 0
            # valid = valid.unsqueeze(-1)
            
            self.precisions[class_n] = p
            self.recalls[class_n] = r
            # self.valid[class_n] = valid 
            self.intersection[class_n] += intersection.cpu()
            self.gt_sum[class_n] += gt_sum.cpu()
            self.pred_sum[class_n] += pred_sum.cpu()
            
        return self.precisions, self.recalls, self.intersection, \
            self.gt_sum, self.pred_sum
        
        
class Metric_IoU(nn.Module):
    def __init__(self):
        super().__init__()
        self.iou = Metric_IoU_base()

    def __call__(self, pred_, gt_, class_num, mask_=None):
        """
        

        Parameters
        ----------
        pred_ : TYPE
            Shape=[batch, class, ...]
        gt_ : TYPE
            Shape=[batch, ...]
        class_num : TYPE
            Number of classes
        mask_ : TYPE, optional
            Shape=[batch, ...]. Larger than 0 are the region to be masked

        Returns
        -------
            IoU, Intersections, Unions, Valid

        """
        volume = torch.argmax(pred_, dim=1)
        volume = volume.clamp(0, class_num-1)
        gt = gt_.clamp(0, class_num-1)
        
        if mask_ is not None:
            volume = volume[mask_ == 0]
            gt = gt[mask_ == 0]
        
        # print('arg\n', volume)
        volume = torch.nn.functional.one_hot(volume, class_num).view(-1,class_num)
        # print('onehot volume\n', volume)
        gt = torch.nn.functional.one_hot(gt,class_num).view(-1,class_num)
        # print('onehot gt\n', gt)
        return self.iou(volume,gt)
    
class Metric_PR(nn.Module):
    def __init__(self):
        super().__init__()
        self.acc = Metric_PR_Base()

    def __call__(self, pred_, gt_, class_num, mask_=None):
        """
        

        Parameters
        ----------
        pred_ : TYPE
            Shape=[batch, class, ...]
        gt_ : TYPE
            Shape=[batch, ...]
        class_num : TYPE
            Number of classes
        mask_ : TYPE, optional
            Shape=[batch, ...]. Larger than 0 are the region to be masked

        Returns
        -------
            Accuracy, Recall, Valid

        """
        volume = torch.argmax(pred_, dim=1)
        volume = volume.clamp(0, class_num-1)
        gt = gt_.clamp(0, class_num-1)
        
        if mask_ is not None:
            volume = volume[mask_ == 0]
            gt = gt[mask_ == 0]
        
        volume = torch.nn.functional.one_hot(volume, class_num).view(-1,class_num)
        gt = torch.nn.functional.one_hot(gt,class_num).view(-1,class_num)
        return self.acc(volume,gt)
    
def Metric_F1(precision, recall):
    return 2 * precision * recall / (precision + recall)
    
        
def test_basic():
    batch=1
    class_num=3
    volume_pred = torch.rand(batch,class_num,1,1,2)
    full_gt = torch.randint(0, class_num,size=(batch,1,1,2))
    
    volume_pred = torch.FloatTensor([[0.1,0.1,0.7],[1,0,0], [0.2,0.8,0.0], [0,0,1], [0,1,0]]) #2,0,1,2,1
    full_gt = torch.LongTensor([[2],[1],[2],[1],[1]])
    print('volume_pred', volume_pred)
    print('gt', full_gt)
    
    # Semantic 
    print("semantic: ")
    iou_ = Metric_IoU()
    acc_ = Metric_PR()
    iou, inters, unions = iou_(volume_pred, full_gt,class_num = class_num)
    print('expect intersections: [0, 1, 1], got', inters,'mean',inters.mean())
    print('expect unions: [1, 4, 3], got', unions,'mean',unions.mean())
    print('expect IoU: [0, 0.25, 0.33], got', iou,'mean',iou.mean())
    acc, recall = acc_(volume_pred, full_gt,class_num = class_num)
    print('expect Accuracy: [0, 0.33, 0.5], got', acc,'mean',acc.mean())
    print('expect Recall: [0, 0.5, 0.5], got', recall,'mean',recall.mean())
    
    # Completion
    print("completion: ")
    iou_ = Metric_IoU()
    acc_ = Metric_PR()
    iou, inters, unions = iou_(volume_pred, full_gt,class_num = 2)
    print('expect intersections: [0, 4], got', inters, 'mean',inters.mean())
    print('expect unions: [1, 5], got', unions,'mean',unions.mean())
    print('expect IoU: [0, 0.8], got', iou,'mean',iou.mean())
    acc, recall = acc_(volume_pred, full_gt,class_num = 2)
    print('expect Accuracy: [0, 0.8], got', acc,'mean',acc.mean())
    print('expect Recall: [0, 0.8], got', recall,'mean',recall.mean())
    print('recall mean: ', recall.mean())
    
def test():
    from torch.utils.data import DataLoader
    from dataset_volume import Dataset
    from config import Config
    config = Config('../config.yml.example')
    config.BATCH_SIZE=1
    train_dataset = Dataset(config, '../example_data/train', '../example_data/gt','../example_data/mask')
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=1,
        drop_last=True,
        shuffle=False
    )
    
    iou_ = Metric_IoU()
    acc_ = Metric_PR()
    
    for items in train_loader:
        volume,gt,mask = items
        # name = items
        
        # gt = torch.from_numpy(gt)
        gt2 = torch.nn.functional.one_hot(gt,14).view(-1,14)
        for i in range(14):
            print(gt2[:,i].sum().item())
        
        volume_pred = torch.rand(config.BATCH_SIZE, config.CLASS_NUM,config.DATA_DIMS[0],config.DATA_DIMS[1],
                                  config.DATA_DIMS[2])
        
        # Semantic 
        iou, inter, union= iou_(volume_pred,gt, config.CLASS_NUM)
        acc, recall, inter2, gt_sum, pred_sum = acc_(volume_pred, gt,config.CLASS_NUM)
        
        
        print("Semantic")
        print("iou:",iou, "mean", iou.mean())
        print("inter:",inter, "mean", inter.mean())
        print("union:", union, "mean", union.mean())
        print("acc:",acc, "acc", acc.mean())
        print("recall:",recall, "mean", recall.mean())
        
        # Completion
        iou, inter, union= iou_(volume_pred,gt, 2)
        acc, recall, inter, gt_sum, pred_sum = acc_(volume_pred, gt, 2)
        print("Completion")
        print("iou:",iou, "mean", iou.mean())
        print("inter:",inter, "mean", inter.mean())
        print("union:", union, "mean", union.mean())
        print("acc:",acc, "acc", acc.mean())
        print("recall:",recall, "mean", recall.mean())
        print('\n\n')
        break
            
if __name__ == "__main__":
    test()
