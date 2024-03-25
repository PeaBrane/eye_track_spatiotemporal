import os
from pathlib import Path

from omegaconf import OmegaConf as OC

import numpy as np
import torch
import matplotlib.pyplot as plt

from eye_dataset import EyeTrackingDataset
from tenn_model import TennSt
from losses import process_detector_prediction


def check_val_score(checkpoint_path, checkpoint_config, remove_blinks=False, test_on_val=True):
    data_path = Path(__file__).parent / 'event_data'

    if test_on_val:
        # Val Data
        data_files = ["1_6", "2_4", "4_4", "6_2", "7_4", "9_1", "10_3", "11_2", "12_3"]
    else:
        # Test data
        data_files = ["1_1", "2_2", "3_1", "4_2", "5_2", "6_4", "7_5", "8_2", "8_3", "10_2", "11_3", "12_4"]

    print(checkpoint_config)
    print(checkpoint_path)

    config = OC.load(checkpoint_config)
    weights = torch.load(checkpoint_path, map_location='cpu')['state_dict']
    mystr = list(weights.keys())[0].split('backbone')[0] # get the str before backbone
    weights = {k.partition(mystr)[2]: v for k, v in weights.items() if k.startswith(mystr)}

    model = TennSt(**OC.to_container(config.model))
    model.eval()
    model.load_state_dict(weights)


    testset = EyeTrackingDataset(data_path, 'test', **OC.to_container(config.dataset), test_on_val=test_on_val)

    collected_distances = np.zeros((0,))

    results = []
    datafile = 0
    for (event, center, openness) in testset:
        pred = model(event.unsqueeze(0))
        pred = process_detector_prediction(pred).squeeze(0)
        
        pred[0] *= 80
        pred[1] *= 60
        center[0] *= 80
        center[1] *= 60
        distances = torch.norm(center - pred, dim=0)
        if remove_blinks:
            distances = distances[openness == 1]

        pred = pred.detach().numpy()
        center = center.detach().numpy()
        distances = distances.detach().numpy()

        p10 = (distances < 10).sum() / distances.size
        distances_5th = distances[::5]
        p10_5th = (distances_5th < 10).sum() / distances_5th.size
        distances.mean()
        collected_distances = np.concatenate((collected_distances, distances_5th), axis=0)
        results.append(
            {
                'datafile': data_files[datafile],
                'pred': pred,
                'center': center,
                'distances': distances,
                'p10_all': p10,
                'p10_5th': p10_5th,
                'openness': openness
            }
        )
        datafile += 1

    p10_total = (collected_distances < 10).sum() / collected_distances.size
    print('Overall p10 (100Hz): ' + str(p10_total))
    euc_total = collected_distances.mean()
    print('Overall Euc. Dist (100Hz): ' + str(euc_total))
    collected_5th_distances = collected_distances[::5]
    p10_5th_total = (collected_5th_distances < 10).sum() / collected_5th_distances.size
    print('Overall p10  (20Hz): ' + str(p10_5th_total))
    euc_5th_total = collected_5th_distances.mean()
    print('Overall Euc. Dist (20Hz): ' + str(euc_5th_total))
    return results


def plot_results(grouped_results, test_on_val=True):
    outdir = './val_results' if test_on_val else './test_results'
    os.makedirs(outdir, exist_ok = True) 
    refres = grouped_results[0]
    for eix in range(len(refres)):
        fig, axs = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True, sharex=True)
        for rix, results in enumerate(grouped_results):
            expt = results[eix]
            distances = expt['distances']
            axs[0].plot(distances)
            if np.any(distances>10):
                misses = np.where(distances>10)
                axs[0].plot(misses, np.ones_like(misses)*(11+rix), '.r')

            axs[1].plot(expt['pred'][0], alpha=0.3)
            axs[2].plot(expt['pred'][1], alpha=0.3)

        blinks = np.where(expt['openness']==0)
        axs[0].plot(blinks, np.ones_like(blinks)*(-1), '.k')
        axs[0].set_ylabel('Distance')
        axs[0].set_title('Validation File: ' + expt['datafile'])
        axs[0].set_ylim([-1.4, 20])

        axs[1].plot(expt['center'][0], 'xkcd:aqua', label='X', linewidth=3)
        axs[1].plot(blinks, np.ones_like(blinks)*(5), '.k')
        axs[1].set_ylabel('X Position')
        axs[1].set_ylim([0, 80])

        axs[2].plot(expt['center'][1], 'xkcd:salmon', label='Y', linewidth=3)
        axs[2].plot(blinks, np.ones_like(blinks)*(5), '.k')
        axs[2].set_ylabel('Y Position')
        axs[2].set_xlabel('Timestep')
        axs[2].set_ylim([0, 60])
        fig.savefig(os.path.join(outdir, 'results_'+expt['datafile']+'.png'))


if __name__=='__main__':
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_grad_enabled(False)
    
    checkpoints = [Path(__file__).parent /'weights/submission.ckpt']
    config = [Path(__file__).parent / 'submission_config.yaml']
    #checkpoints = ['/home/scrouzet/AIS2024_CVPR/train_tenn/outputs/2024-03-22/06-03-29/lightning_logs/version_0/checkpoints/last.ckpt']
    #config = ['/home/scrouzet/AIS2024_CVPR/train_tenn/outputs/2024-03-22/06-03-29/lightning_logs/version_0/config.yaml']

    test_on_val = True
    grouped_results = []
    for k, checkpoint in enumerate(checkpoints):
        grouped_results.append(check_val_score(checkpoint_path=checkpoint,
                                            checkpoint_config = config[k],
                                            remove_blinks=False,
                                            test_on_val=test_on_val
                                            ))

    plot_results(grouped_results,
                test_on_val=test_on_val)

