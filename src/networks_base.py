if __name__ == '__main__' and __package__ is None:
    from os import sys#, path
    sys.path.append('../')

import torch
import torch.nn as nn
from utils import get_pad_same
from utils import cal_gan_from_op

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()
        self.param_inited = False
    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier_normal | kaiming | orthogonal | xavier_unifrom
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        
        def init_func(m):
            classname = m.__class__.__name__
            # print('classname',classname)            
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_unifrom':
                    nn.init.xavier_uniform_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError()

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.init_apply(init_func)
    def getParamList(self,x):
        return list(x.parameters())
    def init_apply(self, fn):
        # print(self.__class__.__name__,'\t', id(self), '\t', self.param_inited)
        for m in self.children():
            if hasattr(m, 'param_inited'):
                # print('\t [Is BaseModule]children', m.__class__.__name__,'\t',m.param_inited)
                if m.param_inited is False:
                    m.init_apply(fn)
                    # print('\t\t init children')
            else:
                # print('\t [tNot BaseModule]children', m.__class__.__name__)
                m.apply(fn)
        if self.param_inited is False:
            # print('\tinit myself')
            fn(self)
            self.param_inited=True
        return self
    
class mySequential(nn.Sequential, BaseNetwork):
    def __init__(self, *args):
        super(mySequential, self).__init__(*args)
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
    
class ConvSame(BaseNetwork):
    def __init__(self, in_channels,out_channels,kernel_size,stride=1,dilation=1, 
             groups=1, bias=True, padding_mode='zeros', padding_value=0,
             activation=None, norm=None, conv_layer=nn.Conv3d
             ):
        super(ConvSame, self).__init__()
        padding = get_pad_same(dilation, kernel_size)
        
        # print('padding',padding)
        pad = None
        if padding > 0:
            if padding_mode == 'zeros' or padding_mode == 'circular': # default conv padding
                pass
            elif padding_mode == 'constant':
                pad = nn.ConstantPad3d(padding, padding_value)
                padding=0
            elif padding_mode == 'replicate':
                pad = nn.ReplicationPad3d(padding)
                padding=0
            else:
                raise RuntimeError('Import padding method is not supported')
            
        
        if conv_layer is GatedConv:
            blocks=[]
            if pad is not None:
                blocks.append(
                    mySequential(pad)
                    )
            blocks.append(mySequential(
                conv_layer(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, 'zeros',
                 norm, activation)
            ))
            # self.ops = conv_layer(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, 'zeros',
                 # norm, activation)
            self.ops = mySequential(*blocks)
        else:
            blocks=[]
            
            if pad is not None:
                blocks.append(
                    mySequential(pad)
                    )
            
            blocks.append(mySequential(
                conv_layer(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, 'zeros')
            ))
            if norm is not None:
                blocks.append(mySequential(
                    norm(out_channels,track_running_stats=False)
                ))
            if activation is not None:
                blocks.append(mySequential(
                    activation
                ))
            self.ops = mySequential(*blocks)
    def forward(self, x):
        return self.ops(x)
    
# def ConvSame(in_channels,out_channels,kernel_size,stride=1,dilation=1, 
#              groups=1, bias=True, padding_mode='zeros',
#              activation=None, norm=None, conv_layer=nn.Conv3d):
#     padding = get_pad_same(dilation, kernel_size)
    
#     if conv_layer is GatedConv:
#         return conv_layer(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, 'zeros',
#              norm, activation)
        
#     blocks=[]
#     blocks.append(mySequential(
#         conv_layer(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, 'zeros')
#     ))
#     if norm is not None:
#         blocks.append(mySequential(
#             norm(out_channels,track_running_stats=False)
#         ))
#     if activation is not None:
#         blocks.append(mySequential(
#             activation
#         ))
#     return mySequential(*blocks)

class ConvTransposeSame(BaseNetwork):
    def __init__(self, in_channels,out_channels,kernel_size,stride,dilation,
                      output_padding=0, activation=None, norm=None, bias=True, padding_mode='zeros', padding_value=0):
        super(ConvTransposeSame, self).__init__()
        padding = get_pad_same(dilation, kernel_size)

        # print('convT:', padding)
        pad = None        
        # if padding > 0:
        #     if padding_mode == 'zeros' or padding_mode == 'circular': # default conv padding
        #         pass
        #     elif padding_mode == 'constant':
        #         pad = nn.ConstantPad3d(padding, padding_value)
        #         padding=0
        #     elif padding_mode == 'replication':
        #         pad = nn.ReplicationPad3d(padding)
        #         padding=0
        #     else:
        #         raise RuntimeError('Import padding method is not supported')
        
        blocks=[]
        if pad is not None:
            blocks.append(
                mySequential(pad)
                )
        blocks.append(mySequential(
            nn.ConvTranspose3d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size,
                              stride=stride,
                              dilation=dilation,
                              padding=padding,
                              bias=bias,
                              output_padding=output_padding)
        ))
        if norm is not None:
            blocks.append(mySequential(
                norm(out_channels,track_running_stats=False)
            ))
        if activation is not None:
            blocks.append(mySequential(
                activation
            ))
        self.ops = mySequential(*blocks)
    def forward(self, x):
        return self.ops(x)
    
# def ConvTransposeSame(in_channels,out_channels,kernel_size,stride,dilation,
#                       output_padding=0, activation=None, norm=None):
#     pad = get_pad_same(dilation, kernel_size)
#     blocks=[]
#     blocks.append(mySequential(
#         nn.ConvTranspose3d(in_channels=in_channels, 
#                           out_channels=out_channels, 
#                           kernel_size=kernel_size,
#                           stride=stride,
#                           dilation=dilation,
#                           padding=pad,
#                           output_padding=output_padding)
#     ))
#     if norm is not None:
#         blocks.append(mySequential(
#             norm(out_channels,track_running_stats=False)
#         ))
#     if activation is not None:
#         blocks.append(mySequential(
#             activation
#         ))
#     return mySequential(*blocks)

class ResSSC(BaseNetwork):
    '''
    x -> y1 = convertion? convert(x):x -> y2 = Convs(y1)
    y = res_add? res_add(x) + y2 : y1 + y2    
    '''
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, stride=1, dilation=1,
                 residual_blocks=1,
                 activation=torch.nn.LeakyReLU(0.2, inplace=True),
                 res_add = True,
                 norm = nn.BatchNorm3d,
                 conv_layer=nn.Conv3d,
                 bias = True, padding_mode='zeros', padding_value=0,
                 ):
        super(ResSSC, self).__init__()
        self.activation=activation
        self.norm = norm(out_channels,track_running_stats=False) if norm is not None else None
        #TODO: delete me or delete BatchNorm3D after test
        if in_channels != out_channels:
            blocks = []
            blocks.append(mySequential(
                    ConvSame(in_channels, out_channels, kernel_size, stride, dilation, conv_layer=conv_layer, bias=bias, padding_mode='zeros', padding_value=padding_value)
            ))
            
            if self.norm is not None:
                blocks.append(mySequential(
                    norm(out_channels,track_running_stats=False)
                ))
            if self.activation is not None:
                blocks.append(mySequential(
                    self.activation
                ))
                
            self.conversion = mySequential(*blocks)
            residual_blocks -= 1
        else:
            self.conversion=None
            
        if res_add:
            self.res = ConvSame(in_channels, out_channels, 1, 1, 1, conv_layer=conv_layer, bias=bias, padding_mode='zeros', padding_value=padding_value)
        else:
            self.res=None
            
        blocks = []
        for n in range(residual_blocks):
            blocks.append(mySequential(
                ConvSame(out_channels, out_channels, kernel_size, stride,dilation, conv_layer=conv_layer, bias=bias, padding_mode='zeros', padding_value=padding_value)
            ))
            
            if n + 1 < residual_blocks:
                if norm is not None:
                    blocks.append(mySequential(
                        norm(out_channels,track_running_stats=False),
                    ))
                if self.activation is not None:
                    blocks.append(mySequential(
                        activation
                    ))
        self.middle = mySequential(*blocks)
        
        
    def forward(self, x):
        if self.res is not None:
            res = self.res(x)
        else:
            res = x
        if self.conversion is not None:
            c = self.conversion(x)
        else:
            c = x
        x = self.middle(c)
        
        if self.res is not None:
            x = res+x
        else:
            x = c + x
            
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x
        
class AddWithBNActivator(BaseNetwork):
    def __init__(self, activator, norm):
        super(AddWithBNActivator, self).__init__()
        self.activator = activator
        self.norm = norm
    def forward(self, x, y):
        out = torch.add(x,y)
        if self.norm is not None:
            out = self.norm(out)
        if self.activator is not None:
            out = self.activator(out)
        return out
    
class ASPP(BaseNetwork): #Atrous Spatial Pyramid Pooling
    def __init__(self, in_channels,out_channels,kernel_size:list,dilation:list,
                 activation=None, norm=None, conv_layer=nn.Conv3d):
        super(ASPP, self).__init__()
        
        
        blocks=[]
        for i in range(len(kernel_size)):
            blocks.append(
                ConvSame(in_channels,out_channels,kernel_size=kernel_size[i],
                         stride=1,dilation=dilation[i],bias=False,
                         activation=activation,norm=norm,conv_layer=conv_layer)
            )
        blocks.append(    
            mySequential(
                nn.AdaptiveAvgPool3d((None,None,None)),
                ConvSame(in_channels,out_channels,1,1,1,bias=False,norm=norm,conv_layer=conv_layer,activation=activation
             ))
        )
            
        self.blocks = blocks
        self.conv = ConvSame(out_channels*len(blocks),out_channels,kernel_size=1,
                         stride=1,dilation=1,bias=False,
                         activation=activation,norm=norm,conv_layer=conv_layer)
        
    def forward(self,x):
        y = [block(x) for block in self.blocks ]
        # [print(r.shape) for r in y]
        y = torch.cat(tuple(y), dim=1)
        y = self.conv(y)
        return y
    
class GatedConv(BaseNetwork):
    """
    Gated Convlution layer with activation (default activation:LeakyReLU)
    Params: same as conv3d
    Input: The feature from last layer "I"
    Output:\phi(f(I))*\sigmoid(g(I))
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 norm=None, 
                 activation=torch.nn.LeakyReLU(0.2, inplace=True),
                 init_weights=True):
        super(GatedConv, self).__init__()
        self.activation = activation
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.mask_conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.sigmoid = torch.nn.Sigmoid()
        self.norm = norm(out_channels) if norm is not None else None

        if init_weights:
            if self.activation is not None:
                nn.init.xavier_normal_(self.conv.weight.data, gain=cal_gan_from_op(self.activation))
            else:
                nn.init.xavier_normal_(self.conv.weight.data, gain=cal_gan_from_op(self.conv))
            nn.init.xavier_uniform_(self.mask_conv.weight.data, gain=cal_gan_from_op(torch.nn.Sigmoid))
            if bias:
                nn.init.constant_(self.conv.bias.data, 0.0)
                nn.init.constant_(self.mask_conv.bias.data, 0.0)
            self.param_inited = True
            
    def gated(self, mask):
        #return torch.clamp(mask, -1, 1)
        return self.sigmoid(mask)
    def forward(self, input):
        x = self.conv(input)
        mask = self.mask_conv(input)
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            x = x * self.gated(mask)
        if self.norm is not None:
            return self.norm(x)
        else:
            return x
    
if __name__ == "__main__":
    from config import Config
    from torchviz import make_dot
    config = Config('../config.yml.example')
    
    g = GatedConv(1, 1, 3)
    # x = torch.rand(config.BATCH_SIZE,1,64,64,64)    
    # aspp = ASPP(1,6,[1,3,3],[1,3,5],activation=torch.nn.LeakyReLU(0.2, inplace=True),norm=nn.BatchNorm3d)    
    # y = aspp(x)
    # graph = make_dot(y)
    # graph.view()
