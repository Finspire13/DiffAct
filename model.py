import copy
import math
import torch
import random
import numpy as np
import time as Time
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d

# Modified from DiffusionDet and pytorch-diffusion-model

########################################################################################

def get_timestep_embedding(timesteps, embedding_dim): # for diffusion model
    # timesteps: batch,
    # out:       batch, embedding_dim
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb

def swish(x):
    return x * torch.sigmoid(x)

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def normalize(x, scale): # [0,1] > [-scale, scale]
    x = (x * 2 - 1.) * scale
    return x

def denormalize(x, scale): #  [-scale, scale] > [0,1]
    x = ((x / scale) + 1) / 2  
    return x

######################################################################################

class ASDiffusionModel(nn.Module):
    def __init__(self, encoder_params, decoder_params, diffusion_params, num_classes, device):
        super(ASDiffusionModel, self).__init__()

        self.device = device
        self.num_classes = num_classes

        timesteps = diffusion_params['timesteps']
        betas = cosine_beta_schedule(timesteps)  # torch.Size([1000])
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = diffusion_params['sampling_timesteps']
        assert self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = diffusion_params['ddim_sampling_eta']
        self.scale = diffusion_params['snr_scale']

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        ################################################################

        self.detach_decoder = diffusion_params['detach_decoder']
        self.cond_types = diffusion_params['cond_types']

        self.use_instance_norm = encoder_params['use_instance_norm']
        if self.use_instance_norm:
            self.ins_norm = nn.InstanceNorm1d(encoder_params['input_dim'], track_running_stats=False)

        decoder_params['input_dim'] = len([i for i in encoder_params['feature_layer_indices'] if i not in [-1, -2]]) * encoder_params['num_f_maps']
        if -1 in encoder_params['feature_layer_indices']: # -1 means "video feature"
            decoder_params['input_dim'] += encoder_params['input_dim']
        if -2 in encoder_params['feature_layer_indices']: # -2 means "encoder prediction"
            decoder_params['input_dim'] += self.num_classes

        decoder_params['num_classes'] = num_classes
        encoder_params['num_classes'] = num_classes
        encoder_params.pop('use_instance_norm')

        self.encoder = EncoderModel(**encoder_params)
        self.decoder = DecoderModel(**decoder_params)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def q_sample(self, x_start, t, noise=None): # forward diffusion
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def model_predictions(self, backbone_feats, x, t):

        x_m = torch.clamp(x, min=-1 * self.scale, max=self.scale) # [-scale, +scale]
        x_m = denormalize(x_m, self.scale)                        # [0, 1]

        assert(x_m.max() <= 1 and x_m.min() >= 0)
        x_start = self.decoder(backbone_feats, t, x_m.float()) # torch.Size([1, C, T])
        x_start = F.softmax(x_start, 1)
        assert(x_start.max() <= 1 and x_start.min() >= 0)

        x_start = normalize(x_start, self.scale)                              # [-scale, +scale]
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return pred_noise, x_start
        

    def prepare_targets(self, event_gt):

        # event_gt: normalized [0, 1]

        assert(event_gt.max() <= 1 and event_gt.min() >= 0)

        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()

        noise = torch.randn(size=event_gt.shape, device=self.device)

        x_start = (event_gt * 2. - 1.) * self.scale  #[-scale, +scale]

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        event_diffused = ((x / self.scale) + 1) / 2.           # normalized [0, 1]

        return event_diffused, noise, t

    
    def forward(self, backbone_feats, t, event_diffused, event_gt=None, boundary_gt=None): # only for train

        if self.detach_decoder:
            backbone_feats = backbone_feats.detach()

        assert(event_diffused.max() <= 1 and event_diffused.min() >= 0)
    
        cond_type = random.choice(self.cond_types)

        if cond_type == 'full':
            event_out = self.decoder(backbone_feats, t, event_diffused.float())
        
        elif cond_type == 'zero':
            event_out = self.decoder(torch.zeros_like(backbone_feats), t, event_diffused.float())
        
        elif cond_type == 'boundary05-':
            feature_mask = (boundary_gt < 0.5).float() # maybe try 0.1
            event_out = self.decoder(feature_mask * backbone_feats, t, event_diffused.float())

        elif cond_type == 'boundary03-':
            feature_mask = (boundary_gt < 0.3).float() # maybe try 0.1
            event_out = self.decoder(feature_mask * backbone_feats, t, event_diffused.float())

        elif cond_type == 'segment=1':
            event_gt = torch.argmax(event_gt, dim=1, keepdim=True).long() # 1, 1, T
            events = torch.unique(event_gt)
            random_event = np.random.choice(events.cpu().numpy())
            feature_mask = (event_gt != random_event).float()
            event_out = self.decoder(feature_mask * backbone_feats, t, event_diffused.float())

        elif cond_type == 'segment=2':
            event_gt = torch.argmax(event_gt, dim=1, keepdim=True).long() # 1, 1, T
            events = torch.unique(event_gt)
            random_event_1 = np.random.choice(events.cpu().numpy())
            random_event_2 = np.random.choice(events.cpu().numpy())
            feature_mask = (event_gt != random_event_1).float() * (event_gt != random_event_2).float()
            event_out = self.decoder(feature_mask * backbone_feats, t, event_diffused.float())

        else:
            raise Exception('Invalid Cond Type')

        return event_out

    def get_training_loss(self, video_feats, event_gt, boundary_gt,
          encoder_ce_criterion, encoder_mse_criterion, encoder_boundary_criterion,
          decoder_ce_criterion, decoder_mse_criterion, decoder_boundary_criterion, 
          soft_label):

        if self.use_instance_norm:
            video_feats = self.ins_norm(video_feats)

        encoder_out, backbone_feats = self.encoder(video_feats, get_features=True)

        if soft_label is None:
            encoder_ce_loss = encoder_ce_criterion(
                encoder_out.transpose(2, 1).contiguous().view(-1, self.num_classes), 
                torch.argmax(event_gt, dim=1).view(-1).long()   # batch_size must = 1
            )
        else:
            soft_event_gt = torch.clone(event_gt).float().cpu().numpy()
            for i in range(soft_event_gt.shape[1]):
                soft_event_gt[0,i] = gaussian_filter1d(soft_event_gt[0,i], soft_label)
            soft_event_gt = torch.from_numpy(soft_event_gt).to(self.device)

            encoder_ce_loss = - soft_event_gt * F.log_softmax(encoder_out, 1)
            encoder_ce_loss = encoder_ce_loss.sum(0).sum(0)

        encoder_mse_loss = torch.clamp(encoder_mse_criterion(
            F.log_softmax(encoder_out[:, :, 1:], dim=1), 
            F.log_softmax(encoder_out.detach()[:, :, :-1], dim=1)), 
            min=0, max=16)

        encoder_boundary_loss = torch.tensor(0).to(self.device) # No boundary loss for encoder
        encoder_ce_loss = encoder_ce_loss.mean()
        encoder_mse_loss = encoder_mse_loss.mean()

        ########## 

        event_diffused, noise, t = self.prepare_targets(event_gt)
        event_out = self.forward(backbone_feats, t, event_diffused, event_gt, boundary_gt)

        decoder_boundary = 1 - torch.einsum('bicl,bcjl->bijl', 
            F.softmax(event_out[:,None,:,1:], 2), 
            F.softmax(event_out[:,:,None,:-1].detach(), 1)
        ).squeeze(1)
        
        if soft_label is None:    # To improve efficiency
            decoder_ce_loss = decoder_ce_criterion(
                event_out.transpose(2, 1).contiguous().view(-1, self.num_classes), 
                torch.argmax(event_gt, dim=1).view(-1).long()   # batch_size must = 1
            )
        else:
            soft_event_gt = torch.clone(event_gt).float().cpu().numpy()
            for i in range(soft_event_gt.shape[1]):
                soft_event_gt[0,i] = gaussian_filter1d(soft_event_gt[0,i], soft_label) # the soft label is not normalized
            soft_event_gt = torch.from_numpy(soft_event_gt).to(self.device)

            decoder_ce_loss = - soft_event_gt * F.log_softmax(event_out, 1)
            decoder_ce_loss = decoder_ce_loss.sum(0).sum(0)

        decoder_mse_loss = torch.clamp(decoder_mse_criterion(
            F.log_softmax(event_out[:, :, 1:], dim=1), 
            F.log_softmax(event_out.detach()[:, :, :-1], dim=1)), 
            min=0, max=16)

        decoder_boundary_loss = decoder_boundary_criterion(decoder_boundary, boundary_gt[:,:,1:])
        decoder_boundary_loss = decoder_boundary_loss.mean()

        decoder_ce_loss = decoder_ce_loss.mean()
        decoder_mse_loss = decoder_mse_loss.mean()

        loss_dict = {
            'encoder_ce_loss': encoder_ce_loss,
            'encoder_mse_loss': encoder_mse_loss,
            'encoder_boundary_loss': encoder_boundary_loss,

            'decoder_ce_loss': decoder_ce_loss,
            'decoder_mse_loss': decoder_mse_loss,
            'decoder_boundary_loss': decoder_boundary_loss,
        }

        return loss_dict


    @torch.no_grad()
    def ddim_sample(self, video_feats, seed=None):

        if self.use_instance_norm:
            video_feats = self.ins_norm(video_feats)

        encoder_out, backbone_feats = self.encoder(video_feats, get_features=True)

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # torch.Size([1, 19, 4847])
        shape = (video_feats.shape[0], self.num_classes, video_feats.shape[2])
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
        
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        # tensor([ -1., 249., 499., 749., 999.])
        times = list(reversed(times.int().tolist()))
        # [999, 749, 499, 249, -1]
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        # [(999, 749), (749, 499), (499, 249), (249, -1)]

        x_time = torch.randn(shape, device=self.device)

        x_start = None
        for time, time_next in time_pairs:

            time_cond = torch.full((1,), time, device=self.device, dtype=torch.long)

            pred_noise, x_start = self.model_predictions(backbone_feats, x_time, time_cond)

            x_return = torch.clone(x_start)
            
            if time_next < 0:
                x_time = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x_time)

            x_time = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        x_return = denormalize(x_return, self.scale)  

        if seed is not None:
            t = 1000 * Time.time() # current time in milliseconds
            t = int(t) % 2**16
            random.seed(t)
            torch.manual_seed(t)
            torch.cuda.manual_seed_all(t)

        return x_return


########################################################################################
# Encoder and Decoder are adapted from ASFormer. 
# Compared to ASFormer, the main difference is that this version applies attention in a similar manner as dilated temporal convolutions.
# This difference does not change performance evidently in preliminary experiments.


class EncoderModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, input_dim, num_classes, kernel_size, 
                 normal_dropout_rate, channel_dropout_rate, temporal_dropout_rate,
                 feature_layer_indices=None):
        super(EncoderModel, self).__init__()
        
        self.num_classes = num_classes
        self.feature_layer_indices = feature_layer_indices
        
        self.dropout_channel = nn.Dropout2d(p=channel_dropout_rate)
        self.dropout_temporal = nn.Dropout2d(p=temporal_dropout_rate)
        
        self.conv_in = nn.Conv1d(input_dim, num_f_maps, 1)
        self.encoder = MixedConvAttModule(num_layers, num_f_maps, kernel_size, normal_dropout_rate)
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)


    def forward(self, x, get_features=False):
        if get_features:
            assert(self.feature_layer_indices is not None and len(self.feature_layer_indices) > 0)
            features = []
            if -1 in self.feature_layer_indices:
                features.append(x)
            x = self.dropout_channel(x.unsqueeze(3)).squeeze(3)
            x = self.dropout_temporal(x.unsqueeze(3).transpose(1, 2)).squeeze(3).transpose(1, 2)
            x, feature = self.encoder(self.conv_in(x), feature_layer_indices=self.feature_layer_indices)
            if feature is not None:
                features.append(feature)
            out = self.conv_out(x)
            if -2 in self.feature_layer_indices:
                features.append(F.softmax(out, 1))
            return out, torch.cat(features, 1)
        else:
            x = self.dropout_channel(x.unsqueeze(3)).squeeze(3)
            x = self.dropout_temporal(x.unsqueeze(3).transpose(1, 2)).squeeze(3).transpose(1, 2)
            out = self.conv_out(self.encoder(self.conv_in(x), feature_layer_indices=None))
            return out



class DecoderModel(nn.Module):
    def __init__(self, input_dim, num_classes,
        num_layers, num_f_maps, time_emb_dim, kernel_size, dropout_rate):
        
        super(DecoderModel, self).__init__()

        self.time_emb_dim = time_emb_dim

        self.time_in = nn.ModuleList([
            torch.nn.Linear(time_emb_dim, time_emb_dim),
            torch.nn.Linear(time_emb_dim, time_emb_dim)
        ])

        self.conv_in = nn.Conv1d(num_classes, num_f_maps, 1)
        self.module = MixedConvAttModuleV2(num_layers, num_f_maps, input_dim, kernel_size, dropout_rate, time_emb_dim)
        self.conv_out =  nn.Conv1d(num_f_maps, num_classes, 1)


    def forward(self, x, t, event):

        time_emb = get_timestep_embedding(t, self.time_emb_dim)
        time_emb = self.time_in[0](time_emb)
        time_emb = swish(time_emb)
        time_emb = self.time_in[1](time_emb)

        fra = self.conv_in(event)
        fra = self.module(fra, x, time_emb)
        event_out = self.conv_out(fra)

        return event_out


class MixedConvAttModuleV2(nn.Module): # for decoder
    def __init__(self, num_layers, num_f_maps, input_dim_cross, kernel_size, dropout_rate, time_emb_dim=None):
        super(MixedConvAttModuleV2, self).__init__()

        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, num_f_maps)

        self.layers = nn.ModuleList([copy.deepcopy(
            MixedConvAttentionLayerV2(num_f_maps, input_dim_cross, kernel_size, 2 ** i, dropout_rate)
        ) for i in range(num_layers)])  #2 ** i
    
    def forward(self, x, x_cross, time_emb=None):

        if time_emb is not None:
            x = x + self.time_proj(swish(time_emb))[:,:,None]

        for layer in self.layers:
            x = layer(x, x_cross)

        return x


class MixedConvAttentionLayerV2(nn.Module):
    
    def __init__(self, d_model, d_cross, kernel_size, dilation, dropout_rate):
        super(MixedConvAttentionLayerV2, self).__init__()
        
        self.d_model = d_model
        self.d_cross = d_cross
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout_rate = dropout_rate
        self.padding = (self.kernel_size // 2) * self.dilation 
        
        assert(self.kernel_size % 2 == 1)

        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=self.padding, dilation=dilation),
        )

        self.att_linear_q = nn.Conv1d(d_model + d_cross, d_model, 1)
        self.att_linear_k = nn.Conv1d(d_model + d_cross, d_model, 1)
        self.att_linear_v = nn.Conv1d(d_model, d_model, 1)

        self.ffn_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.InstanceNorm1d(d_model, track_running_stats=False)

        self.attn_indices = None


    def get_attn_indices(self, l, device):
            
        attn_indices = []
                
        for q in range(l):
            s = q - self.padding
            e = q + self.padding + 1
            step = max(self.dilation // 1, 1)  
            # 1  2  4   8  16  32  64  128  256  512  # self.dilation
            # 1  1  1   2  4   8   16   32   64  128  # max(self.dilation // 4, 1)  
            # 3  3  3 ...                             (k=3, //1)          
            # 3  5  5  ....                           (k=3, //2)
            # 3  5  9   9 ...                         (k=3, //4)
                        
            indices = [i + self.padding for i in range(s,e,step)]

            attn_indices.append(indices)
        
        attn_indices = np.array(attn_indices)
            
        self.attn_indices = torch.from_numpy(attn_indices).long()
        self.attn_indices = self.attn_indices.to(device)
        
        
    def attention(self, x, x_cross):
        
        if self.attn_indices is None:
            self.get_attn_indices(x.shape[2], x.device)
        else:
            if self.attn_indices.shape[0] < x.shape[2]:
                self.get_attn_indices(x.shape[2], x.device)
                                
        flat_indicies = torch.reshape(self.attn_indices[:x.shape[2],:], (-1,))
        
        x_q = self.att_linear_q(torch.cat([x, x_cross], 1))
        x_k = self.att_linear_k(torch.cat([x, x_cross], 1))
        x_v = self.att_linear_v(x)

        x_k = torch.index_select(
            F.pad(x_k, (self.padding, self.padding), 'constant', 0),
            2, flat_indicies)  
        x_v = torch.index_select(
            F.pad(x_v, (self.padding, self.padding), 'constant', 0), 
            2, flat_indicies)  
                        
        x_k = torch.reshape(x_k, (x_k.shape[0], x_k.shape[1], x_q.shape[2], self.attn_indices.shape[1]))
        x_v = torch.reshape(x_v, (x_v.shape[0], x_v.shape[1], x_q.shape[2], self.attn_indices.shape[1])) 
        
        att = torch.einsum('n c l, n c l k -> n l k', x_q, x_k)
        
        padding_mask = torch.logical_and(
            self.attn_indices[:x.shape[2],:] >= self.padding,
            self.attn_indices[:x.shape[2],:] < att.shape[1] + self.padding
        ) # 1 keep, 0 mask
        
        att = att / np.sqrt(self.d_model)
        att = att + torch.log(padding_mask + 1e-6)
        att = F.softmax(att, 2)
        att = att * padding_mask

        r = torch.einsum('n l k, n c l k -> n c l', att, x_v)
        
        return r
    
                
    def forward(self, x, x_cross):
        
        x_drop = self.dropout(x)
        x_cross_drop = self.dropout(x_cross)

        out1 = self.conv_block(x_drop)
        out2 = self.attention(x_drop, x_cross_drop)
                
        out = self.ffn_block(self.norm(out1 + out2))

        return x + out


class MixedConvAttModule(nn.Module): # for encoder
    def __init__(self, num_layers, num_f_maps, kernel_size, dropout_rate, time_emb_dim=None):
        super(MixedConvAttModule, self).__init__()

        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, num_f_maps)

        self.layers = nn.ModuleList([copy.deepcopy(
            MixedConvAttentionLayer(num_f_maps, kernel_size, 2 ** i, dropout_rate)
        ) for i in range(num_layers)])  #2 ** i
    
    def forward(self, x, time_emb=None, feature_layer_indices=None):

        if time_emb is not None:
            x = x + self.time_proj(swish(time_emb))[:,:,None]

        if feature_layer_indices is None:
            for layer in self.layers:
                x = layer(x)
            return x
        else:
            out = []
            for l_id, layer in enumerate(self.layers):
                x = layer(x)
                if l_id in feature_layer_indices:
                    out.append(x)
            
            if len(out) > 0:
                out = torch.cat(out, 1)
            else:
                out = None

            return x, out
    

class MixedConvAttentionLayer(nn.Module):
    
    def __init__(self, d_model, kernel_size, dilation, dropout_rate):
        super(MixedConvAttentionLayer, self).__init__()
        
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout_rate = dropout_rate
        self.padding = (self.kernel_size // 2) * self.dilation 
        
        assert(self.kernel_size % 2 == 1)

        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=self.padding, dilation=dilation),
        )

        self.att_linear_q = nn.Conv1d(d_model, d_model, 1)
        self.att_linear_k = nn.Conv1d(d_model, d_model, 1)
        self.att_linear_v = nn.Conv1d(d_model, d_model, 1)
        
        self.ffn_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.InstanceNorm1d(d_model, track_running_stats=False)

        self.attn_indices = None


    def get_attn_indices(self, l, device):
            
        attn_indices = []
                
        for q in range(l):
            s = q - self.padding
            e = q + self.padding + 1
            step = max(self.dilation // 1, 1)  
            # 1  2  4   8  16  32  64  128  256  512  # self.dilation
            # 1  1  1   2  4   8   16   32   64  128  # max(self.dilation // 4, 1)  
            # 3  3  3 ...                             (k=3, //1)          
            # 3  5  5  ....                           (k=3, //2)
            # 3  5  9   9 ...                         (k=3, //4)
                        
            indices = [i + self.padding for i in range(s,e,step)]

            attn_indices.append(indices)
        
        attn_indices = np.array(attn_indices)
            
        self.attn_indices = torch.from_numpy(attn_indices).long()
        self.attn_indices = self.attn_indices.to(device)
        
        
    def attention(self, x):
        
        if self.attn_indices is None:
            self.get_attn_indices(x.shape[2], x.device)
        else:
            if self.attn_indices.shape[0] < x.shape[2]:
                self.get_attn_indices(x.shape[2], x.device)
                                
        flat_indicies = torch.reshape(self.attn_indices[:x.shape[2],:], (-1,))
        
        x_q = self.att_linear_q(x)
        x_k = self.att_linear_k(x)
        x_v = self.att_linear_v(x)
                
        x_k = torch.index_select(
            F.pad(x_k, (self.padding, self.padding), 'constant', 0),
            2, flat_indicies)  
        x_v = torch.index_select(
            F.pad(x_v, (self.padding, self.padding), 'constant', 0), 
            2, flat_indicies)  
                        
        x_k = torch.reshape(x_k, (x_k.shape[0], x_k.shape[1], x_q.shape[2], self.attn_indices.shape[1]))
        x_v = torch.reshape(x_v, (x_v.shape[0], x_v.shape[1], x_q.shape[2], self.attn_indices.shape[1])) 
        
        att = torch.einsum('n c l, n c l k -> n l k', x_q, x_k)
        
        padding_mask = torch.logical_and(
            self.attn_indices[:x.shape[2],:] >= self.padding,
            self.attn_indices[:x.shape[2],:] < att.shape[1] + self.padding
        ) # 1 keep, 0 mask
        
        att = att / np.sqrt(self.d_model)
        att = att + torch.log(padding_mask + 1e-6)
        att = F.softmax(att, 2) 
        att = att * padding_mask

        r = torch.einsum('n l k, n c l k -> n c l', att, x_v)
        
        return r
    
                
    def forward(self, x):
        
        x_drop = self.dropout(x)
        out1 = self.conv_block(x_drop)
        out2 = self.attention(x_drop)
        out = self.ffn_block(self.norm(out1 + out2))

        return x + out