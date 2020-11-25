if __name__ == '__main__' and __package__ is None:
    from os import sys#, path
    sys.path.append('../')

import torch
import torch.nn as nn
from networks_base import BaseNetwork, ConvSame, ConvTransposeSame, ResSSC, AddWithBNActivator, mySequential, GatedConv

        
class EncoderForkNet(BaseNetwork):
    def __init__(self, config, in_channels, out_channels, init_weights=True, conv_layer = nn.Conv3d, norm=nn.InstanceNorm3d):
        super(EncoderForkNet, self).__init__()
        activation=torch.nn.LeakyReLU(0.2, inplace=True)
        # norm = nn.InstanceNorm3d #nn.BatchNorm3d
        # norm=None
        
        self.encoder = mySequential(
            ConvSame(in_channels, 16, 7, 2, 1,activation=activation,norm=norm,conv_layer=conv_layer, bias=False, padding_mode=config.PADDING, padding_value=config.PAD_VALUE),# h1_base
            ResSSC(16, 32,residual_blocks=3, res_add=False,norm=norm,conv_layer=conv_layer, bias=False, padding_mode=config.PADDING, padding_value=config.PAD_VALUE), #h1_0, h1_1, h1_2, h2_0
            ConvSame(32, 64, 3, 1, 1,activation=activation,norm=norm,conv_layer=conv_layer, bias=False, padding_mode=config.PADDING, padding_value=config.PAD_VALUE), # h2_1
        )
        self.h2_1_1 = mySequential(
            ConvSame(64, 64, 3, 1, 1,activation=activation,norm=norm,conv_layer=conv_layer, bias=False, padding_mode=config.PADDING, padding_value=config.PAD_VALUE), # h2_1
        )
        self.h2_2 = ConvSame(64, 64, 3, 1, 1) # h2_1
        if norm is not None:
            self.h3_0 = AddWithBNActivator(activator=activation, norm=norm(64,track_running_stats=False))
        else:
            self.h3_0 = AddWithBNActivator(activator=activation, norm=None)
        
        self.h4_0 = mySequential(
            ConvSame(64,64,3,1,1,activation=activation,norm=norm,conv_layer=conv_layer, bias=False, padding_mode=config.PADDING, padding_value=config.PAD_VALUE), #h3_1
            ResSSC(64, 64, res_add=False,activation=activation,norm=norm,conv_layer=conv_layer, bias=False, padding_mode=config.PADDING, padding_value=config.PAD_VALUE), # h4_0
        )
        self.h5_0 = mySequential(
            ConvSame(64,64,3,1,2,activation=activation,norm=norm,conv_layer=conv_layer, bias=False, padding_mode=config.PADDING, padding_value=config.PAD_VALUE), #h4_1
            ResSSC(64, 64, res_add=False,activation=activation,norm=norm,conv_layer=conv_layer, bias=False, padding_mode=config.PADDING, padding_value=config.PAD_VALUE), # h5_0
        )
        self.h6_0 = mySequential(
            ConvSame(64,64,3,1,2,activation=activation,norm=norm,conv_layer=conv_layer, bias=False, padding_mode=config.PADDING, padding_value=config.PAD_VALUE), #h5_1
            ResSSC(64, 64, res_add=False,activation=activation,norm=norm,conv_layer=conv_layer, bias=False, padding_mode=config.PADDING, padding_value=config.PAD_VALUE), # h6_0
        )
        
        self.h7_1 = ConvTransposeSame(192, 64, 4, 2, 1, bias=False, padding_mode=config.PADDING, padding_value=config.PAD_VALUE)
        self.h7 = mySequential( # h7_2, h7_3, h7_4
            ConvSame(64+in_channels, 128, 1, 1, 1,activation=activation,norm=norm,conv_layer=conv_layer, bias=False, padding_mode=config.PADDING, padding_value=config.PAD_VALUE), # h7_2
            ConvSame(128, 128, 1, 1, 1,activation=activation, norm=norm,conv_layer=conv_layer, bias=False, padding_mode=config.PADDING, padding_value=config.PAD_VALUE), #h7_3
            ConvSame(128, out_channels, 1, 1, 1,conv_layer=conv_layer, bias=False, padding_mode=config.PADDING, padding_value=config.PAD_VALUE), #h7_4
            torch.nn.Softmax(1)
        )
        
        if init_weights:
            self.h7_1.init_weights('xavier_normal', nn.init.calculate_gain('conv_transpose3d'))
            self.init_weights('xavier_normal', nn.init.calculate_gain('leaky_relu',0.2))
            
    def forward(self, x):
        h2_1 = self.encoder(x)
        h2_1_1 = self.h2_1_1(h2_1)
        h2_2 = self.h2_2(h2_1)
        h3 = self.h3_0(h2_2,h2_1_1)
        h4_0 = self.h4_0(h3)
        h5_0 = self.h5_0(h4_0)
        h6_0 = self.h6_0(h5_0)
        
        # print('h2_1:',h2_1.shape)
        # print('h2_1_1:',h2_1_1.shape)
        # print('h2_2:',h2_2.shape)
        # print('h3:',h3.shape)
        # print('h4_0:',h4_0.shape)
        # print('h5_0:',h5_0.shape)
        # print('h6_0:',h6_0.shape)
        
        y =torch.cat((h4_0,h5_0,h6_0),1)
        # print('y',y.shape)
        y = self.h7_1(y)
        # print('y',y.shape)
        y = self.h7(torch.cat((y, x),1))
        # print('y',y.shape)
        return y
    
class EncoderForkNet2(BaseNetwork):
    def __init__(self, in_channels, out_channels=256, init_weights=True, norm_layer=None):
        super(EncoderForkNet2, self).__init__()
        activation=torch.nn.LeakyReLU(0.2, inplace=True)
        # norm=None
        self.encoder= mySequential(
            ConvSame(in_channels, 32, 4, 2, 1,activation=activation,norm=norm_layer), # h1
            ConvSame(32, 128,4,2,1,activation=activation,norm=norm_layer), #h2
            ConvSame(128, out_channels, 4, 2, 1,activation=activation,norm=norm_layer), #h3
        )
        if init_weights:
            self.encoder.init_weights('xavier_normal', nn.init.calculate_gain('leaky_relu',0.2))
            self.init_weights('xavier_normal')
    def forward(self, x):
        y = self.encoder(x)
        return y
    
class GeneratorForkNet(BaseNetwork):
    def __init__(self, in_channels, out_channels, mode, init_weights=True):
        super(GeneratorForkNet, self).__init__()
        self.mode = mode
        self.relu_inplace= torch.nn.ReLU(inplace=True)
        self.relu= torch.nn.ReLU(inplace=False)
        self.softmax = torch.nn.Softmax(1)
        self.tanh = torch.nn.Tanh()
        norm = nn.InstanceNorm3d #nn.BatchNorm3d
        # norm=None
        h4_input_shape = 128
        h5_input_shape = 32
        if mode == 0: #df
            pass
        elif mode == 1: #com
            pass
        elif mode == 2: #sem
            h4_input_shape *= 2
            h5_input_shape *= 2
        else:
            raise NotImplementedError()
            
        self.h3 = mySequential(
            ConvTransposeSame(in_channels,128,1,2,4, output_padding=1,activation=self.relu,norm=norm),
        )
        self.h4 = mySequential(
            ConvTransposeSame(h4_input_shape,32,1,2,4, output_padding=1,activation=self.relu,norm=norm),
        )
        self.h5 = mySequential(
            ConvTransposeSame(h5_input_shape,out_channels,1,2,4, output_padding=1),
        )

        if mode == 1:
            self.refine = mySequential(
                ResSSC(out_channels,16,3,1,1,residual_blocks=3,res_add=False,
                        activation=torch.nn.ReLU(inplace=True), norm=norm), #res1
                ConvSame(16, out_channels,3,1,1), # res_1_post
                self.softmax, # stage2
            )
        if mode == 2:
            self.res = mySequential(
                ConvSame(out_channels+2, 16, 3, 1, 1,activation=self.relu_inplace,norm=norm),
                ConvSame(16, out_channels, 3, 1, 1,activation=self.relu_inplace),                
            )
            if norm is not None:
                self.res1 = AddWithBNActivator(torch.nn.ReLU(inplace=True),norm(out_channels*2))
            else:
                self.res1 = AddWithBNActivator(torch.nn.ReLU(inplace=True),None)
            
            self.res_post = mySequential(
                ConvSame(out_channels,out_channels,3,1,1),
                torch.nn.Softmax(1)
            )
        
        if init_weights:
            self.init_weights()
    def forward(self, x, h3_=None,h4_=None,h5_=None):
        h3 = self.h3(x) # b 256 16 16 16
        if self.mode == 0: # df
            h4 = self.h4(h3) # b 32 32 32 32
            h5 = self.h5(h4) # b out_channels 64 64 64
            return self.tanh(h5)
        if self.mode == 1: # com
            h4 = self.h4(h3) # b 32 32 32 32
            h5 = self.h5(h4) # b out_channels 64 64 64
            stage1 = self.softmax(h5)
            stage2 = self.refine(stage1)
            return stage1, stage2, h3, h4, h5
        if self.mode == 2:
            h4 = self.h4(torch.cat((h3,h3_),1)) # b 32 32 32 32
            h5 = self.h5(torch.cat((h4,h4_),1)) # b out_channels 64 64 64
            res = self.res(torch.cat((h5,h5_),1))
            res1 = self.res1(h5, res)
            stage1 = self.res_post(res1)
            return stage1
        
class GeneratorRefineForkNet(BaseNetwork):
    def __init__(self, out_channels, init_weights=True):
        super(GeneratorRefineForkNet, self).__init__()
        relu_inplace= torch.nn.ReLU(inplace=True)
        norm = nn.InstanceNorm3d #nn.BatchNorm3d
        self.refine = mySequential(
                ResSSC(out_channels,16,3,1,1,3,activation=relu_inplace,res_add=False,norm=norm), # res2
                ResSSC(16,16,3,1,1,2,activation=relu_inplace,res_add=False,norm=norm), # res3
                ResSSC(16,16,3,1,1,2,activation=relu_inplace,res_add=False,norm=norm), # res4
                ConvSame(16,out_channels,3,1,1),
                torch.nn.Softmax(1)
        )
        if init_weights:
            self.init_weights()
    def forward(self, x):
        return self.refine(x)
    
class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True, 
                 conv_layer = nn.Conv3d):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = mySequential(
            spectral_norm(conv_layer(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = mySequential(
            spectral_norm(conv_layer(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = mySequential(
            spectral_norm(conv_layer(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = mySequential(
            spectral_norm(conv_layer(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = mySequential(
            spectral_norm(conv_layer(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )
        if init_weights:
            self.conv1.init_weights('xavier_normal', nn.init.calculate_gain('leaky_relu',0.2))
            self.conv2.init_weights('xavier_normal', nn.init.calculate_gain('leaky_relu',0.2))
            self.conv3.init_weights('xavier_normal', nn.init.calculate_gain('leaky_relu',0.2))
            self.conv4.init_weights('xavier_normal', nn.init.calculate_gain('leaky_relu',0.2))
            self.conv5.init_weights('xavier_normal', nn.init.calculate_gain('conv3d'))

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)
        return outputs, [conv1, conv2, conv3, conv4, conv5]
def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module
    



def TEST_encoder_forknet(config):
    config.BATCH_SIZE = 1
    input_size = 2
    output_size= 12
    x = torch.rand(config.BATCH_SIZE,input_size,64,64,64)
    
    # encoder_forknet = EncoderForkNet(1, 14, norm=None,conv_layer=GatedConv)
    # y = encoder_forknet(x)
    
    # encoder_forknet = EncoderForkNet(2, 14, norm=nn.BatchNorm3d,conv_layer=GatedConv)
    # y = encoder_forknet(x)
    
    encoder_forknet = EncoderForkNet(config, input_size, output_size, norm=nn.InstanceNorm3d,conv_layer=GatedConv)
    encoder_forknet.eval()
    y = encoder_forknet(x)
    
    # encoder_forknet = EncoderForkNet(2, 12, norm=nn.InstanceNorm3d, conv_layer=nn.Conv3d)
    # y = encoder_forknet(x)
    
    print('input shape:',x.shape)
    print('output shape:',y.shape)
    graph = make_dot(y)
    graph.view()
    
    
def TEST_ForkNet(config):
    config.BATCH_SIZE=1
    code_dim = 256
    encoder1 = EncoderForkNet(1,config.CLASS_NUM)
    encoder2 = EncoderForkNet2(config.CLASS_NUM,code_dim)
    generator_df = GeneratorForkNet(code_dim,1,0)
    generator_com = GeneratorForkNet(code_dim,2,1)
    generator_sem = GeneratorForkNet(code_dim,config.CLASS_NUM,2)
    generator_sem_ref = GeneratorRefineForkNet(config.CLASS_NUM)
    
    x_in = torch.rand(config.BATCH_SIZE,1,64,64,64)
    y_in = torch.rand(config.BATCH_SIZE,config.CLASS_NUM,64,64,64)
    code_in = torch.rand(config.BATCH_SIZE,code_dim, 8, 8, 8)
    
    
    y = encoder1(x_in)
    y = encoder2(y.detach())
    df = generator_df(y.detach())
    with torch.no_grad():
        com_dec, com_dec_ref, h3, h4, h5 = generator_com(y.detach())
    sem_suf = generator_sem(y.detach(),h3,h4,h5)
    sem_full = generator_sem_ref(sem_suf)
    
    graph = make_dot(y)
    graph.view()
    
if __name__ == "__main__":
    from torchviz import make_dot
    from config import Config
    config = Config('../config.yml.example')
    
    # TEST_generator(config)
    TEST_encoder_forknet(config)
    # TEST_ForkNet(config)
 
