import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf as OC

from eye_dataset import EyeTrackingDataset
from tenn_model import TennSt
from losses import process_detector_prediction

# NOTE: this submission script runs the network in streaming mode, and does not use the GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''
# torch.set_num_threads(1)
torch.set_grad_enabled(False)


def streaming_inference(model, frames):
    model.eval()
    model.streaming()
    model.reset_memory()
    
    predictions = []
    with torch.inference_mode():
        for frame_id in range(frames.shape[2]):  # stream the frames to the model
            prediction = model(frames[:, :, [frame_id]])
            predictions.append(prediction)
                
    predictions = torch.cat(predictions, dim=2)
    return predictions


config_path = 'config_submission.yaml'
checkpoint_path = 'weights/submission.ckpt'

#config_path = '/home/scrouzet/AIS2024_CVPR/train_tenn/outputs/2024-03-22/06-03-29/lightning_logs/version_0/config.yaml'
#checkpoint_path = '/home/scrouzet/AIS2024_CVPR/train_tenn/outputs/2024-03-22/06-03-29/lightning_logs/version_0/checkpoints/last.ckpt'

config = OC.load(config_path)
data_path = Path(__file__).parent / 'event_data'

weights = torch.load(checkpoint_path, map_location='cpu')['state_dict']
mystr = list(weights.keys())[0].split('backbone')[0] # get the str before backbone
weights = {k.partition(mystr)[2]: v for k, v in weights.items() if k.startswith(mystr)}

model = TennSt(**OC.to_container(config.model))
model.eval()
model.load_state_dict(weights)

testset = EyeTrackingDataset(data_path, 'test', **OC.to_container(config.dataset))
event_frames_list = [event_frames for (event_frames, _, _) in testset]

predictions = []
for event_frames in event_frames_list:
    pred = streaming_inference(model, event_frames[None, :])
    pred = process_detector_prediction(pred)
    predictions.append(pred.squeeze(0)[..., ::5])
    
predictions = torch.cat(predictions, dim=-1)

predictions[0] *= 80
predictions[1] *= 60

predictions_numpy = predictions.detach().numpy().T
predictions_numpy = np.concatenate([np.arange(len(predictions_numpy))[:, None], predictions_numpy], axis=1)

df = pd.DataFrame(predictions_numpy, columns=['row_id', 'x', 'y'])
df.to_csv('submission.csv', index=False)
