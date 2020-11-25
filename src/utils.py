if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import os
import sys
import time
import random
import numpy as np
from PIL import Image
import torch.nn as nn
import math
import torch

NYU14_name_list = ['Unknown', 'Bed', 'Books', 'Ceiling', 'Chair',
                  'Floor', 'Furniture', 'Objects', 'Picture',
                  'Sofa', 'Table', 'TV', 'Wall', 'Window'
                 ]
Label11_name_list = ["None", "Ceiling", "Floor", "Wall", "Window",
                     "Chair", "Bed", "Sofa", "Desk","TV","Furniture","Objects"]

def get_label_name_list(label_num):
    if label_num == 14:
        return NYU14_name_list
    elif label_num == 12:
        return Label11_name_list
    else:
        raise NotImplementedError('')
    


def formatString(means:list,name:str):
        def numpy_to_string(x:np):
            string = ''
            for n in x:
                string += '%5.3f\t' % n
            return string
        return '{}{}'.format(numpy_to_string(means[name].numpy()), '%5.3f' % means[name].mean().item())

def cal_gan_from_op(x:nn.Module):
    if x is torch.nn.LeakyReLU or x.__class__.__name__ == 'LeakyReLU':
        # print(hasattr(activation, 'negative_slope'))
        return nn.init.calculate_gain('leaky_relu',x.negative_slope)
    if x is torch.nn.Sigmoid or x.__class__.__name__ == 'Sigmoid':
        return nn.init.calculate_gain('sigmoid')
    if x is torch.nn.ReLU()  or x.__class__.__name__ == 'ReLU':
        return nn.init.calculate_gain('relu')
    if x is torch.nn.Tanh  or x.__class__.__name__ == 'Tanh':
        return nn.init.calculate_gain('tanh')
    if x is torch.nn.Conv1d  or x.__class__.__name__ == 'Conv1d':
        return nn.init.calculate_gain('conv1d')
    if x is torch.nn.Conv2d  or x.__class__.__name__ == 'Conv2d':
        return nn.init.calculate_gain('conv2d')
    if x is torch.nn.Conv3d  or x.__class__.__name__ == 'Conv3d':
        return nn.init.calculate_gain('conv3d')
    if x is torch.nn.ConvTranspose1d or x.__class__.__name__ == 'ConvTranspose1d':
        return nn.init.calculate_gain('conv_transpose1d')
    if x is torch.nn.ConvTranspose2d or x.__class__.__name__ == 'ConvTranspose2d':
        return nn.init.calculate_gain('conv_transpose2d')
    if x is torch.nn.ConvTranspose3d or x.__class__.__name__ == 'ConvTranspose3d':
        return nn.init.calculate_gain('conv_transpose3d')
    raise NotImplementedError('x.__class__.__name__:',x.__class__.__name__)

def print_params(named_parameters):
    for name, param in named_parameters:
        if param.requires_grad:
            print(name, param.data.sum(), param.grad.sum())

def mean_with_mask(x, mask):
        return (x * mask).sum() / (mask.sum()+1e-6)        

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    

def create_mask(width, height, mask_width, mask_height, x=None, y=None):
    mask = np.zeros((height, width))
    mask_x = x if x is not None else random.randint(0, width - mask_width)
    mask_y = y if y is not None else random.randint(0, height - mask_height)
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    return mask

def maskImage(image, mask, merge=False, toFloat=False, inverse=True):
    if inverse:
        mask = 1- mask
    if merge:
        if toFloat:
            return (image * (1-mask).float()) + mask
        else:
            return (image * (1-mask)) + mask
    else:
        if toFloat:
            return (image * (1-mask).float())
        else:
            return (image * (1-mask))

def stitch_images(inputs, *outputs, img_per_row=2):
    gap = 5
    columns = len(outputs) + 1

    width, height = inputs[0][:, :, 0].shape
    img = Image.new('RGB', (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img


def imshow(img, title=''):
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.axis('off')
    plt.imshow(img, interpolation='none')
    plt.show()


def imsave(img, path):
    im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
    im.save(path)


def get_pad_same(dilation,kernel_size, stride=1,shape_in=1,shape_out=1):
        # return tuple ((int(0.5 * (stride*(shape-1)-shape+dilation*(kernel_size-1)+1)),
        #           int(0.5 * (stride*(shape-1)-shape+dilation*(kernel_size-1)+1)),
        #           int(0.5 * (stride*(shape-1)-shape+dilation*(kernel_size-1)+1))))
    # return int(0.5 * (dilation*(kernel_size-1)))
    return int(math.floor(0.5 * (stride*(shape_out-1)+1-shape_in+dilation*(kernel_size-1))))
        

class Progbar(object):
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=25, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None, silent=False):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if not silent:
                if self._dynamic_display:
                    sys.stdout.write('\b' * prev_total_width)
                    sys.stdout.write('\r')
                else:
                    sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            if not silent:
                sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            if not silent:
                sys.stdout.write(info)
                sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'
                if not silent:
                    sys.stdout.write(info)
                    sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None,silent=False):
        self.update(self._seen_so_far + n, values,silent=silent)
        
def volumeToPointCloud(volume:torch.Tensor):
    if volume.dim() == 4:        
        batch = volume.size(0)
        X = volume.size(1)
        Y = volume.size(2)
        Z = volume.size(3)
        output = torch.zeros([batch,  X*Y*Z, 3])
        counter=0;
        for b in range(batch):
            for x in range(X):
                for y in range(Y):
                    for z in range(Z):
                        if volume[b,x,y,z] >= 0:
                            output[b, counter,:] = torch.FloatTensor([
                                x*100,y*100,z*100
                                ])
                            # print(output[b, counter,:])
                            counter+=1
                        else:
                            output[b, counter,:] = torch.FloatTensor([
                                0,0,0
                                ])
                            counter+=1
        return output
    else:
        return None
    
from torch.utils.tensorboard import SummaryWriter
        
if __name__ == '__main__':   
    x = torch.rand(1,30,30,30) 
    x *=2
    x -=1

    # print(x)
    vertices = volumeToPointCloud(x)
    # print(vertices)
    writter = SummaryWriter( 'testlog')
    writter.add_mesh('mesh', vertices)
    writter.add_scalar('scalar', 1.0)
    writter.close()
