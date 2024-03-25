import ast
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from natsort import natsorted
from torch.nn import functional as F
from torch.utils.data import Dataset

rand_range = lambda amin, amax: amin + (amax - amin) * np.random.rand()

val_files = ["1_6", "2_4", "4_4", "6_2", "7_4", "9_1", "10_3", "11_2", "12_3"]

def get_index(file_lens, index):
    file_lens_cumsum = np.cumsum(np.array(file_lens))
    file_id = np.searchsorted(file_lens_cumsum, index, side='right')
    sample_id = index - file_lens_cumsum[file_id - 1] if file_id > 0 else index
    return file_id, sample_id


def txt_to_npy(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(ast.literal_eval(line.strip()))
    return np.array(data)


def h5_to_npy(file_path, name):
    with h5py.File(file_path, 'r') as file:
        npy_data = file[name][:]
    return npy_data


def events_to_frames(events, size, num_frames, spatial_downsample, temporal_downsample, mode='bilinear'):
    """Perform bilinear interpolation directly on the events, 
    while converting them to frames and do spatial and temporal downsamplings 
    all at the same time.
    """
    height, width = size    
    p, x, y, t = events
    events_frames = torch.zeros([num_frames, 2, height, width]).type_as(events)
    
    def bilinear_interp(x, scale, x_max):
        if scale == 1:
            return x, x, torch.ones_like(x), torch.zeros_like(x)
        xd1 = x % scale / scale
        xd = 1 - xd1
        x = (x / scale).long().clamp(0, x_max)
        x1 = (x + 1).clamp(0, x_max)
        return x, x1, xd, xd1
    
    if mode == 'nearest':
        p = p.round().long()
        x = (x / spatial_downsample[0]).round().long().clamp(0, width - 1)
        y = (y / spatial_downsample[1]).round().long().clamp(0, height - 1)
        t = (t / temporal_downsample - 0.5).round().long().clamp(0, num_frames - 1)
        events_frames.index_put_((t, p, y, x), torch.ones_like(p, dtype=torch.float32), accumulate=True)
        return events_frames
    
    x, x1, xd, xd1 = bilinear_interp(x, spatial_downsample[0], width - 1)
    y, y1, yd, yd1 = bilinear_interp(y, spatial_downsample[1], height - 1)
    t, t1, td, td1 = bilinear_interp(t, temporal_downsample, num_frames - 1)
    
    # similar to bilinear, but temporally causal
    if mode == 'causal_linear':
        p = p.long().repeat(4)
        
        x = torch.cat([x.repeat(2), x1.repeat(2)])
        y = torch.cat([y, y1]).repeat(2)
        t = t.repeat(4)
        
        xd = torch.cat([xd.repeat(2), xd1.repeat(2)])
        yd = torch.cat([yd, yd1]).repeat(2)
        td = td1.repeat(4)  # causal
        
        events_frames.index_put_((t, p, y, x), xd * yd * td, accumulate=True)
        return events_frames

    # bilinear
    p = p.long().repeat(8)

    x = torch.cat([x.repeat(4), x1.repeat(4)])
    y = torch.cat([y.repeat(2), y1.repeat(2)]).repeat(2)
    t = torch.cat([t, t1]).repeat(4)

    xd = torch.cat([xd.repeat(4), xd1.repeat(4)])
    yd = torch.cat([yd.repeat(2), yd1.repeat(2)]).repeat(2)
    td = torch.cat([td, td1]).repeat(4)

    events_frames.index_put_((t, p, y, x), xd * yd * td, accumulate=True)
    return events_frames


class EventRandomAffine():
    """Perform random affine transformations on the events and labels
    """
    def __init__(self, size, 
                 degrees=15, translate=(0.2, 0.2), scale=(0.8, 1.2), spatial_jitter=None, 
                 augment_flag=True):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.spatial_jitter = spatial_jitter
        self.augment_flag = augment_flag

        self.height, self.width = size

    def normalize(self, coords, backward=False):
        if not backward:
            coords[0] = coords[0] / self.width - 0.5
            coords[1] = coords[1] / self.height - 0.5
        else:
            coords[0] = (coords[0] + 0.5) * self.width
            coords[1] = (coords[1] + 0.5) * self.height

        return coords

    def __call__(self, events, labels):        
        if self.augment_flag:
            degrees = rand_range(-self.degrees, self.degrees) / 180 * math.pi
            translate = [rand_range(-t, t) for t in self.translate]
            scale = [rand_range(*self.scale) for _ in range(2)]

            cos, sin = math.cos(degrees), math.sin(degrees)
            R = torch.tensor([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=torch.float).type_as(events)
            S = torch.tensor([[scale[0], 0, 0], [0, scale[1], 0], [0, 0, 1]], dtype=torch.float).type_as(events)
            T = torch.tensor([[1, 0, translate[0]], [0, 1, translate[1]], [0, 0, 1]], dtype=torch.float).type_as(events)

            trans_matrix = T @ R @ S
        
        else:
            trans_matrix = torch.eye(3).type_as(events)
        
        coords = F.pad(events[1:3], (0, 0, 0, 1), value=1)
        coords = self.normalize(trans_matrix @ self.normalize(coords), True)
        if self.spatial_jitter is not None:
            coords += torch.randn_like(coords) * self.spatial_jitter
        
        events[1:3] = coords[:2]
        val_inds = (coords[0] >= 0) & (coords[0] < self.width) & (coords[1] >= 0) & (coords[1] < self.height)
        events = events[:, val_inds]
        
        labels = labels.T
        centers = F.pad(labels[:2], (0, 0, 0, 1), value=1)
        centers = trans_matrix @ self.normalize(centers)
        centers = centers[:2] + 0.5
        
        closes = labels[-1]
        
        return events, centers, closes
    
    
class EyeTrackingDataset(Dataset):
    def __init__(self, 
                 root_path, 
                 mode='train', 
                 device='cpu', 
                 time_window=10000, 
                 frames_per_segment=50, 
                 spatial_downsample=(5, 5), 
                 events_interpolation='bilinear', 
                 spatial_affine=True, 
                 temporal_flip=True, 
                 temporal_scale=True, 
                 temporal_shift=True,
                 test_on_val=False):
        self.mode = mode
        self.time_window = time_window
        self.frames_per_segment = frames_per_segment
        self.time_window_per_segment = time_window * frames_per_segment
        self.spatial_downsample = spatial_downsample
        self.events_interpolation = events_interpolation
        assert time_window == 10000
        
        self.temporal_flip = temporal_flip
        self.temporal_scale = temporal_scale
        self.temporal_shift = temporal_shift
        
        self.test_on_val = test_on_val
        
        root_path = Path(root_path)
        if mode in ['train', 'val']:
        #if mode == 'train':
            base_path = root_path / 'train'
        elif mode == 'test':
            if test_on_val:
                base_path = root_path / 'train'
            else:
                base_path = root_path / 'test'
        else:
            raise ValueError("Invalid mode. Most be train or test.")
        #data_dirs = natsorted(base_path.glob('*'))
        
        self.events, self.labels = [], []
        self.num_frames_list, self.num_segments_list = [], []
        
        dir_paths = natsorted(base_path.glob('*'))
        if mode == 'train':
            dir_paths = [dir_path for dir_path in dir_paths if dir_path.name not in val_files]
        elif mode == 'val' or (mode == 'test' and test_on_val):
            dir_paths = [dir_path for dir_path in dir_paths if dir_path.name in val_files]


        for dir_path in dir_paths:
#        for dir_path in data_dirs:
            assert dir_path.is_dir()
            data_path = dir_path / f'{dir_path.name}.h5'
            label_path = dir_path / 'label.txt' if (mode != 'test' or test_on_val) else dir_path / 'label_zeros.txt'
            
            event, label = h5_to_npy(data_path, 'events'), txt_to_npy(label_path)
            
            num_frames = label.shape[0]
            self.num_frames_list.append(num_frames)
            self.num_segments_list.append(num_frames // frames_per_segment)
            
            # truncating off trailing events with no labels
            final_t = num_frames * time_window
            final_ind = np.searchsorted(event['t'], final_t, 'left')
            event = event[:final_ind]
            
            label = torch.tensor(label, dtype=torch.float, device=device)
            event = np.stack([event['p'].astype('float32'), event['x'].astype('float32'), event['y'].astype('float32'), event['t'].astype('float32')], axis=0)
            event = torch.tensor(event, dtype=torch.float, device=device)  # (4, N)
            
            self.events.append(event)
            self.labels.append(label)
        
        self.total_segments = sum(self.num_segments_list)
        
        # spatial affine transformation
        augment_flag = (mode == 'train') and spatial_affine
        self.augment = EventRandomAffine((480, 640), augment_flag=augment_flag)
            
    def __len__(self):
        if self.mode == 'test':
            return len(self.events)
        return self.total_segments
    
    def _process_data(self, event, label, index=None):
        event, center, close = self.augment(event, label)
        num_frames = self.frames_per_segment if self.mode != 'test' else self.num_frames_list[index]
        
        event = events_to_frames(event, 
                                 (480 // self.spatial_downsample[1], 640 // self.spatial_downsample[0]), 
                                 num_frames, self.spatial_downsample, self.time_window, 
                                 mode=self.events_interpolation)
        
        # time + polarity flip
        if self.mode == 'train' and self.temporal_flip and np.random.rand() > 0.5:
            event = event.flip(0).flip(1)  # (T, C, H, W)
            center = center.flip(-1)
            close = close.flip(-1)
        
        return event.moveaxis(0, 1), center, 1 - close
    
    def __getitem__(self, index):
        if self.mode == 'test':
            event, label = self.events[index], self.labels[index]
            return self._process_data(event, label, index)
        
        file_id, segment_id = get_index(self.num_segments_list, index)
        event, label = self.events[file_id], self.labels[file_id]
        
        start_t = segment_id * self.time_window * self.frames_per_segment
        end_t = start_t + self.time_window * self.frames_per_segment
        
        # random temporal shift
        max_offset = round(self.time_window_per_segment * 0.1)
        if self.mode == 'train' and self.temporal_shift and start_t >= max_offset:
            offset = np.random.rand() * max_offset
            start_t -= offset
            end_t -= offset
        else:
            offset = 0
        
        # random temporal scaling
        num_frames = self.num_frames_list[file_id]
        event = event.clone()
        if self.mode == 'train' and self.temporal_scale and end_t < (num_frames * self.time_window * 0.8):
            scale_factor = float(rand_range(0.8, 1.2))
            event[-1] *= scale_factor
        else:
            scale_factor = 1
        
        start_ind = torch.searchsorted(event[-1], start_t, side='left')
        end_ind = torch.searchsorted(event[-1], end_t, side='left')
        
        event_segment = event[:, start_ind.item():end_ind.item()]
        event_segment[-1] -= start_t
        
        start_label_id = segment_id * self.frames_per_segment
        end_label_id = (segment_id + 1) * self.frames_per_segment
        
        # label interpolation
        label_numpy = label.cpu().numpy()
        num_frame = label_numpy.shape[0]
        arange = np.arange(0, num_frame)
        label_offset = offset / self.time_window
        interp_range = np.linspace(
            (start_label_id - label_offset) / scale_factor, 
            (end_label_id - label_offset - 1) / scale_factor, 
            self.frames_per_segment, 
        )
        x_interp = np.interp(interp_range, arange, label_numpy[:, 0])
        y_interp = np.interp(interp_range, arange, label_numpy[:, 1])
        closeness = label_numpy[start_label_id:end_label_id, -1]
        label_segment = torch.tensor(np.stack([x_interp, y_interp, closeness], axis=1)).type_as(label)
        
        return self._process_data(event_segment, label_segment)
        