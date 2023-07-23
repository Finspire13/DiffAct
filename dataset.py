import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from utils import get_labels_start_end_time
from scipy.ndimage import gaussian_filter1d

def get_data_dict(feature_dir, label_dir, video_list, event_list, sample_rate=4, temporal_aug=True, boundary_smooth=None):
    
    assert(sample_rate > 0)
        
    data_dict = {k:{
        'feature': None,
        'event_seq_raw': None,
        'event_seq_ext': None,
        'boundary_seq_raw': None,
        'boundary_seq_ext': None,
        } for k in video_list
    }
    
    print(f'Loading Dataset ...')
    
    for video in tqdm(video_list):
        
        feature_file = os.path.join(feature_dir, '{}.npy'.format(video))
        event_file = os.path.join(label_dir, '{}.txt'.format(video))

        event = np.loadtxt(event_file, dtype=str)
        frame_num = len(event)
                
        event_seq_raw = np.zeros((frame_num,))
        for i in range(frame_num):
            if event[i] in event_list:
                event_seq_raw[i] = event_list.index(event[i])
            else:
                event_seq_raw[i] = -100  # background

        boundary_seq_raw = get_boundary_seq(event_seq_raw, boundary_smooth)

        feature = np.load(feature_file, allow_pickle=True)
        
        if len(feature.shape) == 3:
            feature = np.swapaxes(feature, 0, 1)  
        elif len(feature.shape) == 2:
            feature = np.swapaxes(feature, 0, 1)
            feature = np.expand_dims(feature, 0)
        else:
            raise Exception('Invalid Feature.')
                    
        assert(feature.shape[1] == event_seq_raw.shape[0])
        assert(feature.shape[1] == boundary_seq_raw.shape[0])
                                
        if temporal_aug:
            
            feature = [
                feature[:,offset::sample_rate,:]
                for offset in range(sample_rate)
            ]
            
            event_seq_ext = [
                event_seq_raw[offset::sample_rate]
                for offset in range(sample_rate)
            ]

            boundary_seq_ext = [
                boundary_seq_raw[offset::sample_rate]
                for offset in range(sample_rate)
            ]
                        
        else:
            feature = [feature[:,::sample_rate,:]]  
            event_seq_ext = [event_seq_raw[::sample_rate]]
            boundary_seq_ext = [boundary_seq_raw[::sample_rate]]

        data_dict[video]['feature'] = [torch.from_numpy(i).float() for i in feature]
        data_dict[video]['event_seq_raw'] = torch.from_numpy(event_seq_raw).float()
        data_dict[video]['event_seq_ext'] = [torch.from_numpy(i).float() for i in event_seq_ext]
        data_dict[video]['boundary_seq_raw'] = torch.from_numpy(boundary_seq_raw).float()
        data_dict[video]['boundary_seq_ext'] = [torch.from_numpy(i).float() for i in boundary_seq_ext]
        
    return data_dict

def get_boundary_seq(event_seq, boundary_smooth=None):

    boundary_seq = np.zeros_like(event_seq)

    _, start_times, end_times = get_labels_start_end_time([str(int(i)) for i in event_seq])
    boundaries = start_times[1:]
    assert min(boundaries) > 0
    boundary_seq[boundaries] = 1
    boundary_seq[[i-1 for i in boundaries]] = 1

    if boundary_smooth is not None:
        boundary_seq = gaussian_filter1d(boundary_seq, boundary_smooth)
        
        # Normalize. This is ugly.
        temp_seq = np.zeros_like(boundary_seq)
        temp_seq[temp_seq.shape[0] // 2] = 1
        temp_seq[temp_seq.shape[0] // 2 - 1] = 1
        norm_z = gaussian_filter1d(temp_seq, boundary_smooth).max()
        boundary_seq[boundary_seq > norm_z] = norm_z
        boundary_seq /= boundary_seq.max()

    return boundary_seq


def restore_full_sequence(x, full_len, left_offset, right_offset, sample_rate):
        
    frame_ticks = np.arange(left_offset, full_len-right_offset, sample_rate)
    full_ticks = np.arange(frame_ticks[0], frame_ticks[-1]+1, 1)

    interp_func = interp1d(frame_ticks, x, kind='nearest')
    
    assert(len(frame_ticks) == len(x)) # Rethink this
    
    out = np.zeros((full_len))
    out[:frame_ticks[0]] = x[0]
    out[frame_ticks[0]:frame_ticks[-1]+1] = interp_func(full_ticks)
    out[frame_ticks[-1]+1:] = x[-1]

    return out




class VideoFeatureDataset(Dataset):
    def __init__(self, data_dict, class_num, mode):
        super(VideoFeatureDataset, self).__init__()
        
        assert(mode in ['train', 'test'])
        
        self.data_dict = data_dict
        self.class_num = class_num
        self.mode = mode
        self.video_list = [i for i in self.data_dict.keys()]
        
    def get_class_weights(self):
        
        full_event_seq = np.concatenate([self.data_dict[v]['event_seq_raw'] for v in self.video_list])
        class_counts = np.zeros((self.class_num,))
        for c in range(self.class_num):
            class_counts[c] = (full_event_seq == c).sum()
                    
        class_weights = class_counts.sum() / ((class_counts + 10) * self.class_num)

        return class_weights

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):

        video = self.video_list[idx]

        if self.mode == 'train':

            feature = self.data_dict[video]['feature']
            label = self.data_dict[video]['event_seq_ext']
            boundary = self.data_dict[video]['boundary_seq_ext']

            temporal_aug_num = len(feature)
            temporal_rid = random.randint(0, temporal_aug_num - 1) # a<=x<=b
            feature = feature[temporal_rid]
            label = label[temporal_rid]
            boundary = boundary[temporal_rid]

            spatial_aug_num = feature.shape[0]
            spatial_rid = random.randint(0, spatial_aug_num - 1) # a<=x<=b
            feature = feature[spatial_rid]
            
            feature = feature.T   # F x T

            boundary = boundary.unsqueeze(0)
            boundary /= boundary.max()  # normalize again
            
        if self.mode == 'test':

            feature = self.data_dict[video]['feature']
            label = self.data_dict[video]['event_seq_raw']
            boundary = self.data_dict[video]['boundary_seq_ext']  # boundary_seq_raw not used

            feature = [torch.swapaxes(i, 1, 2) for i in feature]  # [10 x F x T]
            label = label.unsqueeze(0)   # 1 X T'  
            boundary = [i.unsqueeze(0).unsqueeze(0) for i in boundary]   # [1 x 1 x T]  

        return feature, label, boundary, video

    
