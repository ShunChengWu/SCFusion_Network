if __name__ == '__main__' and __package__ is None:
    from os import sys
    sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
from networks_scene_inpainting import EncoderForkNet, EncoderForkNet2, GeneratorForkNet, GeneratorRefineForkNet, Discriminator
from networks_base import GatedConv
from loss import AdversarialLoss, LogNLLLoss, SoftF1Loss
from loss_scene import ReconstructionLoss
from config import Config
from model_base import BaseModel

def preprocess_input(x, mask = None):
    x[x<=0] = 0
    x = x * 2 -1
    if mask is not None:
        x[mask==1] = -1
        
class SceneInpaintingForknetModel(BaseModel):
    def __init__(self,config, name):
        super(SceneInpaintingForknetModel,self).__init__(name, config)
        self.class_num = config.CLASS_NUM
        self.use_mask  = config.MASK > 0
        self.use_discriminator = config.DISCRIMINATIVE > 0

        self.predict_df  = self.config.PRED_DF>0        
        self.predict_com = self.config.PRED_COM>0 or self.config.PRED_SEM>0
        self.predict_sem = self.config.PRED_SEM>0
        
        # Normalization Methods
        if config.NORM == 0:
            norm = None
        elif config.NORM == 1:
            norm = nn.BatchNorm3d
        elif config.NORM == 2:
            norm = nn.InstanceNorm3d
        else:
            raise NotImplementedError
            
        # Gated or original Conv
        if config.GATED == 0:
            conv_layer = nn.Conv3d
        else:
            conv_layer = GatedConv
        
        # Mask with mask as an additional input
        if self.use_mask:
            encoder = EncoderForkNet(self.config, 2, config.CLASS_NUM,norm=norm,conv_layer=conv_layer)
        else:
            encoder = EncoderForkNet(self.config, 1, config.CLASS_NUM,norm=norm,conv_layer=conv_layer)
            
        self.has_fork = self.predict_df or self.predict_com or self.predict_sem
        if self.has_fork:
            encoder2 = EncoderForkNet2(config.CLASS_NUM,256)
        if self.predict_df:
            generator_df = GeneratorForkNet(256, 1, 0)
        if self.predict_com:
            generator_com = GeneratorForkNet(256, 2, 1)
        if self.predict_sem:
            generator_sem = GeneratorForkNet(256, config.CLASS_NUM, 2)
            generator_sem_ref = GeneratorRefineForkNet(config.CLASS_NUM)
            
        if self.use_discriminator:
            discriminator_ssc = Discriminator(in_channels=config.CLASS_NUM, use_sigmoid=config.GAN_LOSS != 'hinge')
            # if self.use_mask:
            #     discriminator_ssc = Discriminator(in_channels=config.CLASS_NUM+1, use_sigmoid=config.GAN_LOSS != 'hinge')
            # else:
            #     discriminator_ssc = Discriminator(in_channels=config.CLASS_NUM, use_sigmoid=config.GAN_LOSS != 'hinge')
            
        
        if len(config.GPU) > 1:
            encoder = nn.DataParallel(encoder, config.GPU)
            if self.use_discriminator:
                discriminator_ssc = nn.DataParallel(discriminator_ssc, config.GPU)
            if self.has_fork:
                encoder2 = nn.DataParallel(encoder2, config.GPU)
            if self.predict_df:
                generator_df = nn.DataParallel(generator_df, config.GPU)
            if self.predict_com:
                generator_com = nn.DataParallel(generator_com, config.GPU)
            if self.predict_sem:
                generator_sem = nn.DataParallel(generator_sem, config.GPU)
                generator_sem_ref = nn.DataParallel(generator_sem_ref, config.GPU)
            
        self.add_module('encoder', encoder)
        if self.use_discriminator:
            self.add_module('discriminator_ssc', discriminator_ssc)
        if self.has_fork:
            self.add_module('encoder2',encoder2)
        if self.predict_df:
            self.add_module('generator_df',generator_df)
        if self.predict_com:
            self.add_module('generator_com',generator_com)
        if self.predict_sem:
            self.add_module('generator_sem',generator_sem)
            self.add_module('generator_sem_ref',generator_sem_ref)
        
        self.l1_loss = nn.L1Loss()
        self.reconstruction_loss = ReconstructionLoss()
        if self.use_discriminator:
            self.adversarial_loss = AdversarialLoss(type=config.GAN_LOSS)
        
        self.ssc_optimizer = optim.Adam(
            [{'params': encoder.parameters()}],
            lr = float(config.LR_G),
            betas=(config.BETA_G, config.BETA2)
        )
        self.ssc_optimizer.zero_grad()
        
        if self.has_fork:
            params = list(encoder2.parameters())
            if self.predict_df:
                params += list(generator_df.parameters())
            if self.predict_com:
                params += list(generator_com.parameters())
            if self.predict_sem:
                params += list(generator_sem.parameters())
                params += list(generator_sem_ref.parameters())
                
            self.fork_optimizer =optim.Adam(
                params=params,
                lr = float(config.LR_G),
                betas=(config.BETA_G, config.BETA2)
            )
            self.fork_optimizer.zero_grad()
        if self.use_discriminator:
            self.ssc_dis_optimizer = optim.Adam(
                params=discriminator_ssc.parameters(),
                lr = float(config.LR_D),
                betas=(config.BETA_D, config.BETA2)
            )
            self.ssc_dis_optimizer.zero_grad()

    def process_ssc(self, input_, gt_, mask_=None):
        gen_ssc_loss = 0
        dis_ssc_loss = None
        
        gt_ssc_ = torch.clamp(gt_, 0.0, self.class_num-1).long()
        gt_ssc = torch.nn.functional.one_hot(gt_ssc_,self.class_num).type(torch.float32).permute(0,4,1,2,3)
        weight_sem = self.calculateWeight(gt_ssc_, self.class_num)

        volume = input_.unsqueeze(1).float()        
        if self.use_mask:
            if mask_ is None:
                raise Exception('Must cannot be None in MASK is enabled.')
            mask = mask_.unsqueeze(1).float()
            pred_ssc = self.encoder(torch.cat((volume,mask),1))
        else:
            pred_ssc = self.encoder(volume)
        
        
        if self.config.SSC_LOSS == 'lognll':
            if self.config.MASK_LOSS > 0:
                assert mask_ is not None
                # set unknown region in gt to be the ignore_index
                ignore_index = self.class_num+1
                gt_masked = gt_ssc_.clone()
                gt_masked[(mask_==1)*(gt_masked==0)] = ignore_index
                #print('ignore_index:',ignore_index)
                
                logNLLLoss = LogNLLLoss(weight=weight_sem, ignore_index=ignore_index)
                sem_ssc_loss = logNLLLoss(pred_ssc, gt_masked.long())
            else:
                logNLLLoss = LogNLLLoss(weight=weight_sem, ignore_index=-100)
                sem_ssc_loss = logNLLLoss(pred_ssc, gt_ssc_.long())
        elif self.config.SSC_LOSS == 'softf1':
            sem_ssc_loss = SoftF1Loss(pred_ssc, gt_ssc.long(), weight=weight_sem)
        else:
            raise Exception('Unknown loss type for ssc!')
        
        # sem_ssc_loss = self.reconstruction_loss(pred_ssc,gt_ssc,weight_sem)
        if torch.isnan(sem_ssc_loss):
            raise Exception('sem_ssc_loss loss is nan!')
        gen_ssc_loss += sem_ssc_loss
        
        if self.use_discriminator:
            dis_ssc_loss = 0
            ## discriminator_sem loss ##
            # if self.use_mask:
            #     dis_ssc_input_real = torch.cat((gt_ssc,mask),1)
            #     dis_ssc_input_fake = torch.cat((pred_ssc.detach(),mask),1) 
            # else:
            #     dis_ssc_input_real = gt_ssc
            #     dis_ssc_input_fake = pred_ssc.detach()
            dis_ssc_input_real = gt_ssc
            dis_ssc_input_fake = pred_ssc.detach()
                
            dis_ssc_real, _ = self.discriminator_ssc(dis_ssc_input_real)                    # in: [rgb(3)]
            dis_ssc_fake, _ = self.discriminator_ssc(dis_ssc_input_fake)                    # in: [rgb(3)]
            dis_ssc_real_loss = self.adversarial_loss(dis_ssc_real, True, True)
            dis_ssc_fake_loss = self.adversarial_loss(dis_ssc_fake, False, True)
            dis_ssc_loss += (dis_ssc_real_loss + dis_ssc_fake_loss) / 2
            
            ## generator adversarial loss ##
            # if self.use_mask:
            #     gen_ssc_input_fake = torch.cat((pred_ssc,mask),1)
            # else:
            #     gen_ssc_input_fake = pred_ssc
            gen_ssc_input_fake = pred_ssc
            
            gen_ssc_fake, _ = self.discriminator_ssc(gen_ssc_input_fake)                    # in: [rgb(3)]
            gen_ssc_gan_loss = self.adversarial_loss(gen_ssc_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
            gen_ssc_loss += gen_ssc_gan_loss
            
        return gen_ssc_loss, dis_ssc_loss, pred_ssc

    def process_df(self, input_,gt_,code,mask_=None):
        gen_df_loss = 0
        dis_df_loss = 0
        input_s = input_.unsqueeze(1)
        if self.use_mask:
            input_known = input_ * (1-mask_.unsqueeze(1)).float()
        
        
        pred_df = self.generator_df(code)
        
        if self.use_mask:
            # only compare known area
            df_l1_loss = self.l1_loss(pred_df, input_known) * self.config.L1_LOSS_WEIGHT
        else:
            df_l1_loss = self.l1_loss(pred_df, input_s) * self.config.L1_LOSS_WEIGHT
        if torch.isnan(df_l1_loss):
            print('code.sum():',code.sum())
            print('pred_df.sum():',pred_df.sum())
            raise Exception('df_l1_loss loss is nan!')
        gen_df_loss += df_l1_loss
        
        if self.use_discriminator:
            # discriminator_df loss
            if self.use_mask:
                dis_df_input_real = input_known
            else:
                dis_df_input_real = input_s
            dis_df_input_fake = pred_df.detach()
            dis_df_real, _ = self.discriminator_df(dis_df_input_real)                    # in: [rgb(3)]
            dis_df_fake, _ = self.discriminator_df(dis_df_input_fake)                    # in: [rgb(3)]
            dis_df_real_loss = self.adversarial_loss(dis_df_real, True, True)
            dis_df_fake_loss = self.adversarial_loss(dis_df_fake, False, True)
            dis_df_loss += (dis_df_real_loss + dis_df_fake_loss) / 2
            
            # generator adversarial loss
            ## df
            gen_df_input_fake = pred_df
            gen_df_fake, _ = self.discriminator_df(gen_df_input_fake)                    # in: [rgb(3)]
            gen_df_gan_loss = self.adversarial_loss(gen_df_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
            gen_df_loss += gen_df_gan_loss
        return gen_df_loss, dis_df_loss, pred_df
    
    def process_com(self, input_,gt_,code,mask_=None):
        gen_com_loss = 0
        dis_com_loss = 0
        gt_com_ = torch.clamp(gt_, 0.0,1.0).long()
        gt_com = torch.nn.functional.one_hot(gt_com_,2).type(torch.float32).permute(0,4,1,2,3)
        weight_com = self.calculateWeight(gt_com_,2)
        if self.use_mask:
            mask = mask_.unsqueeze(1)
            mask_known = (1-mask).float()
            mask_known_ = (1-mask_).float()
            weight_com_unknown = self.calculateWeight(gt_com_[mask_>0],2)
            weight_com_known = self.calculateWeight(gt_com_[mask_==0],2)
            input_occupancy_ = input_>=0

        pred_com, pred_com_ref, h3, h4, h5= self.generator_com(code)
            
        if self.use_mask:
            com_ssc_loss=0
            com_scc_ref_loss=0
            com_nllloss_known = LogNLLLoss(weight=weight_com_known)
            com_nllloss_unknown = LogNLLLoss(weight=weight_com_unknown)
            p_known = (mask_==0).float().sum()/mask_.numel()
            
            com_ssc_loss += com_nllloss_known(pred_com*mask_known, (input_occupancy_*mask_known_).long()) * p_known  # known
            com_ssc_loss += com_nllloss_unknown(pred_com*mask, (gt_com_*mask_).long()) * (1.0-p_known) #unknown
            
            com_scc_ref_loss += com_nllloss_known(pred_com_ref*mask_known, (input_occupancy_*mask_known_).long()) * p_known  # known
            com_scc_ref_loss += com_nllloss_unknown(pred_com_ref*mask, (gt_com_*mask_).long()) * (1.0-p_known) #unknown
        else:
            com_nllloss = LogNLLLoss(weight=weight_com)
            com_ssc_loss = com_nllloss(pred_com, gt_com_)
            com_scc_ref_loss = com_nllloss(pred_com_ref, gt_com_)
            # com_ssc_loss = self.reconstruction_loss(pred_com, gt_com,weight_com)
            # com_scc_ref_loss = self.reconstruction_loss(pred_com_ref, gt_com,weight_com)
            
        if torch.isnan(com_ssc_loss):
            for name, param in self.encoder.named_parameters():
                if param.requires_grad:
                    print(name, param.data.sum(), param.grad.sum())
            print('code.sum():',code.sum())
            print('pred_com.sum():',pred_com.sum())
            raise Exception('com_ssc_loss loss is nan!')
        if torch.isnan(com_scc_ref_loss):
            print('code.sum():',code.sum())
            print('pred_com_ref.sum():',pred_com_ref.sum())
            raise Exception('com_scc_ref_loss loss is nan!')
        gen_com_loss += com_ssc_loss + com_scc_ref_loss
        
        if self.use_discriminator:
            # discriminator_com loss
            if self.use_mask:
                dis_com_input_real = gt_com*mask + input_occupancy_ * mask_known
            else:
                dis_com_input_real = gt_com
            dis_com_input_fake = pred_com.detach()
            dis_com_real, _ = self.discriminator_com(dis_com_input_real)                    # in: [rgb(3)]
            dis_com_fake, _ = self.discriminator_com(dis_com_input_fake)                    # in: [rgb(3)]
            dis_com_real_loss = self.adversarial_loss(dis_com_real, True, True)
            dis_com_fake_loss = self.adversarial_loss(dis_com_fake, False, True)
            dis_com_loss += (dis_com_real_loss + dis_com_fake_loss) / 2
            
            # generator adversarial loss
            ## com
            gen_com_input_fake = pred_com
            gen_com_fake, _ = self.discriminator_com(gen_com_input_fake)                    # in: [rgb(3)]
            gen_com_gan_loss = self.adversarial_loss(gen_com_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
            gen_com_loss += gen_com_gan_loss
        return gen_com_loss, dis_com_loss, pred_com, pred_com_ref, h3, h4, h5
    
    def process_sem(self, input_, gt_, code, h3, h4, h5, mask_=None):
        # gen_sem_loss = 0
        dis_sem_loss = 0
        
        gt_sem_ = torch.clamp(gt_, 0.0, self.class_num-1)
        # gt_sem = torch.nn.functional.one_hot(gt_sem_,self.class_num).type(torch.float32).permute(0,4,1,2,3)
        weight_sem = self.calculateWeight(gt_sem_, self.class_num)
        
        pred_sem_part = self.generator_sem(code, h3, h4, h5)
        pred_sem_full = self.generator_sem_ref(pred_sem_part)
        
        logNLLLoss = LogNLLLoss(weight=weight_sem)
        sem_ssc_loss_part = logNLLLoss(pred_sem_part, gt_sem_.long())
        sem_ssc_loss_full = logNLLLoss(pred_sem_full, gt_sem_.long())
        # sem_ssc_loss_part = self.reconstruction_loss(pred_sem_part, gt_sem, weight_sem)
        # sem_ssc_loss_full = self.reconstruction_loss(pred_sem_full, gt_sem, weight_sem)
        
        if torch.isnan(sem_ssc_loss_part):
            print('code.sum():',code.sum())
            print('pred_sem_part.sum():',pred_sem_part.sum())
            raise Exception('sem_ssc_loss_part loss is nan!')
            
        if torch.isnan(sem_ssc_loss_full):
            print('code.sum():',code.sum())
            print('pred_sem_full.sum():',pred_sem_full.sum())
            raise Exception('sem_ssc_loss_full loss is nan!')
        
        # gen_sem_loss += sem_ssc_loss_part + sem_ssc_loss_full
        
        # if self.use_discriminator:
        #     # discriminator_sem loss
        #     dis_sem_input_real = gt_sem
        #     dis_sem_input_fake = pred_sem.detach()
        #     dis_sem_real, _ = self.discriminator_sem(dis_sem_input_real)                    # in: [rgb(3)]
        #     dis_sem_fake, _ = self.discriminator_sem(dis_sem_input_fake)                    # in: [rgb(3)]
        #     dis_sem_real_loss = self.adversarial_loss(dis_sem_real, True, True)
        #     dis_sem_fake_loss = self.adversarial_loss(dis_sem_fake, False, True)
        #     dis_sem_loss += (dis_sem_real_loss + dis_sem_fake_loss) / 2
            
        #     # generator adversarial loss
        #     gen_sem_input_fake = pred_sem
        #     gen_sem_fake, _ = self.discriminator_sem(gen_sem_input_fake)                    # in: [rgb(3)]
        #     gen_sem_gan_loss = self.adversarial_loss(gen_sem_fake, True, False) * self.config.INPAINT_ADV_LOSS_WEIGHT
        #     gen_sem_loss += gen_sem_gan_loss
        # else:
        #     pred_sum = torch.zeros([1])
            
        return sem_ssc_loss_part, sem_ssc_loss_full, dis_sem_loss, pred_sem_part, pred_sem_full

    def train_ssc(self, input_,gt_,mask_):
        gen_ssc_loss, dis_ssc_loss, pred_sem_ssc = self.process_ssc(input_,gt_,mask_)
        
        # # optimize
        # if self.use_discriminator:
        #     dis_ssc_loss.backward()
        #     self.ssc_dis_optimizer.step()
            
        # gen_ssc_loss.backward()
        # self.ssc_optimizer.step()
        
        # if self.use_discriminator:
        #     self.ssc_dis_optimizer.zero_grad()
        # self.ssc_optimizer.zero_grad()
        
        return gen_ssc_loss, dis_ssc_loss, pred_sem_ssc.detach()
    
    def train_generator(self, input_,gt_,mask_):            
        with torch.no_grad():
            _, _, pred_sem_ssc = self.process_ssc(input_,gt_,mask_) # second pass after optimization
        code = self.encoder2(pred_sem_ssc)

        gen_df_loss=gen_com_loss=gen_sem_loss=0
        fork_loss=0
        pred_com=pred_com_ref=pred_sem_surf=pred_sem_full=None
        if self.predict_df:
            gen_df_loss, dis_df_loss, *_ = self.process_df(input_, gt_, code,mask_)
            fork_loss += gen_df_loss
        if self.predict_sem:
            gen_com_loss, dis_com_loss, pred_com, pred_com_ref, h3, h4, h5 = self.process_com(input_,gt_,code,mask_)                
            fork_loss += gen_com_loss

            if self.predict_sem:
                sem_ssc_loss_part, sem_ssc_loss_full, dis_sem_loss, pred_sem_surf, pred_sem_full = \
                    self.process_sem(input_,gt_,code,h3,h4,h5,mask_)
                gen_sem_loss = sem_ssc_loss_part.detach() + sem_ssc_loss_full.detach()
                fork_loss += sem_ssc_loss_part
                fork_loss += sem_ssc_loss_full
                
        if self.has_fork:
            self.fork_optimizer.zero_grad()
            fork_loss.backward()
            self.fork_optimizer.step()
                 
        return gen_df_loss, gen_com_loss, gen_sem_loss, pred_com, pred_com_ref, pred_sem_surf, pred_sem_full

    def process(self, input_, gt_, mask_ = None):
        self.iteration +=1       
        if self.use_mask:
            if mask_ is None:
                raise Exception('Mask was specified as Use but given mask is None')
                
                
        #TODO: test whether this works
        if self.config.MASK_LOSS > 0:
            assert mask_ is not None
            preprocess_input(input_,mask_)
            
                
        # input_[input_<0] = 0

        # Generator loss        
        gen_ssc_loss, dis_ssc_loss, pred_sem_ssc = self.train_ssc(input_,gt_,mask_)
        logs=[("Loss/ssc_gen_loss", gen_ssc_loss.detach().item())]
        if dis_ssc_loss is not None:
            logs+=[("Loss/ssc_dis_loss", dis_ssc_loss.detach().item())]

        # Forks
        gen_df_loss = gen_com_loss = gen_sem_loss = 0
        if self.has_fork:
            gen_df_loss, gen_com_loss, gen_sem_loss, \
                pred_com, pred_com_ref, pred_sem_surf, pred_sem_full = self.train_generator(input_,gt_,mask_)
                     
            if self.predict_df:    
                logs += [("Loss/df_loss", gen_df_loss.detach().item())]
            if self.predict_com:
                logs += [("Loss/com_loss", gen_com_loss.detach().item())]
            if self.predict_sem:
                logs += [("Loss/sem_loss", gen_sem_loss.detach().item())]

        optimized = self.backward(gen_ssc_loss, dis_ssc_loss, gen_df_loss + gen_com_loss + gen_sem_loss)

        if self.predict_sem:
            return optimized, logs, pred_sem_full.data
        else:
            return optimized, logs, pred_sem_ssc.data
    def forward(self, input_, mask_=None):
        #TODO: test whether this works
        if self.config.MASK_LOSS > 0:
            assert mask_ is not None
            preprocess_input(input_,mask_)
            
        if self.use_mask:
            if mask_ is None:
                raise Exception('input mask cannot be None when MASK is enabled.')
            pred_sem_ssc = self.encoder(torch.cat((input_.unsqueeze(1).float(),mask_.unsqueeze(1).float()),1))
        else:
            pred_sem_ssc = self.encoder(input_.unsqueeze(1).float())
        if self.predict_sem:
            code = self.encoder2(pred_sem_ssc)
            pred_com, pred_com_ref, h3, h4, h5= self.generator_com(code)
            pred_sem_part= self.generator_sem(code, h3, h4, h5)
            pred_sem_full = self.generator_sem_ref(pred_sem_part)
            return pred_sem_full
        return pred_sem_ssc
    
    def backward(self, gen_ssc_loss, dis_ssc_loss, fork_loss):
        if self.use_discriminator:
            dis_ssc_loss = dis_ssc_loss / self.config.BATCH_FACTOR
            dis_ssc_loss.backward()
        
        gen_ssc_loss = gen_ssc_loss / self.config.BATCH_FACTOR
        gen_ssc_loss.backward()
        
        if self.has_fork:
            fork_loss = fork_loss / self.config.BATCH_FACTOR
            fork_loss.backward()
            
        if (self.iteration) % self.config.BATCH_FACTOR == 0: 
            self.optimize()
            self.resetGrad()
            return True
        else:
            return False

    def optimize(self):
        if self.use_discriminator:
            self.ssc_dis_optimizer.step()
        self.ssc_optimizer.step()
        if self.has_fork:
            self.fork_optimizer.step()
            
    def resetGrad(self):
        if self.use_discriminator:
            self.ssc_dis_optimizer.zero_grad()
        self.ssc_optimizer.zero_grad()
        if self.has_fork:
            self.fork_optimizer.zero_grad()
    
    def calculateWeight(self, x, class_num):
        x = torch.nn.functional.one_hot(x, class_num).view(-1,class_num).float()
        # batch_mean = torch.mean(x, dim=(0)) #TODO: forknet use mean. but maybe inverse log is better?
        # inverse = 1.0 / (batch_mean + 1.0)
        # weight = inverse / inverse.sum()
        # print('\nweight',weight)
        
        batch_mean = torch.sum(x, dim=(0))
        weight2 = torch.abs(1.0 / (torch.log(batch_mean)+1)) # +1 to prevent 1 /log(1) = inf
        # print('batch_mean',batch_mean)
        # print('weight2', weight2)
        
        return weight2
        
if __name__ == '__main__':   
    config = Config('../config.yml.example')
    
    config.MASK=0
    config.GATED=0
    config.PRED_DF=0
    config.PRED_COM=0
    config.PRED_SEM=0
    config.NORM=0
    config.DISCRIMINATIVE=0
    config.LR_G = 0.0001
    
    batch=2
    class_num=config.CLASS_NUM
    volume = torch.rand(batch,64,64,64)
    full_gt = torch.randint(0, class_num,size=(batch,64,64,64))
    mask = (torch.rand(batch,64,64,64) > 0.3).int()
    
    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")

    volume = volume.to(config.DEVICE)
    full_gt = full_gt.to(config.DEVICE)
    mask = mask.to(config.DEVICE)
    
    # # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    # cv2.setNumThreads(0)


    # # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    # np.random.seed(config.SEED)
    # random.seed(config.SEED)
    
    
    model = SceneInpaintingForknetModel(config, "TEST_RUN").to(config.DEVICE)
    
    # model.save()
    # model.load()
    
    max_iter=1e5
    iter=0
    
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.deterministic=True # slower but less memory comsumption
    
    
    while iter < max_iter:
        _, logs, *_ = model.process(volume, full_gt, mask)# train
        # model(volume, mask_=mask) # forward
        print(logs)
        iter+=1
