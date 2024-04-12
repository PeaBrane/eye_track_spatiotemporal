import os
from pathlib import Path

from omegaconf import OmegaConf as OC

import numpy as np
import torch
import matplotlib.pyplot as plt

from eye_dataset import EyeTrackingDataset
from tenn_model import TennSt
from losses import process_detector_prediction, OutputHook, MacsEstimationHook

torch.set_grad_enabled(False)


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
    model = TennSt(**OC.to_container(config.model))
    model.eval()
    
    if checkpoint_path is not None:
        weights = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        mystr = list(weights.keys())[0].split('backbone')[0] # get the str before backbone
        weights = {k.partition(mystr)[2]: v for k, v in weights.items() if k.startswith(mystr)}
        model.load_state_dict(weights)

    testset = EyeTrackingDataset(data_path, 'test', **OC.to_container(config.dataset), test_on_val=test_on_val)

    collected_distances = np.zeros((0,))

    # Setup per hooks to grab outputs from ReLU layers to measure sparsity
    output_hook = OutputHook()
    relu_counter = 0
    for mm in model.modules():
        if isinstance(mm, torch.nn.ReLU):
            mm.register_forward_hook(output_hook)
            relu_counter+=1

    evdensity_per_layer = []
    for ll in range(relu_counter):
        evdensity_per_layer.append({
            'evs_per_time': np.zeros((0, ))
        })
        
    num_conv_layers = 0
    for mm in model.modules():
        if isinstance(mm, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            num_conv_layers += 1
        
    macs_hook = MacsEstimationHook(num_conv_layers)
    for mm in model.modules():
        if isinstance(mm, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            mm.register_forward_hook(macs_hook)
    
    results = []
    datafile = 0
    for (event, center, openness) in testset:
        pred = model(event.unsqueeze(0))
        pred = process_detector_prediction(pred).squeeze(0)

        # Grab layer outputs for sparsity calculations
        for ll, outs in enumerate(output_hook):
            summation_axis = (0, 1)
            for dim in range(3, outs.ndim):
                summation_axis += (dim,)
            
            evs = torch.sum(outs > 0, axis=summation_axis).detach().numpy() / (np.prod(outs.shape) / outs.shape[2])
            evdensity_per_layer[ll]['evs_per_time'] = np.concatenate((evdensity_per_layer[ll]['evs_per_time'], evs))
        output_hook.clear()
        
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

    metrics = {}
    p10_total = (collected_distances < 10).sum() / collected_distances.size
    print('Overall p10 (100Hz): ' + str(p10_total))
    euc_total = collected_distances.mean()
    print('Overall Euc. Dist (100Hz): ' + str(euc_total))
    collected_5th_distances = collected_distances[::5]
    p10_5th_total = (collected_5th_distances < 10).sum() / collected_5th_distances.size
    print('Overall p10  (20Hz): ' + str(p10_5th_total))
    metrics['p10'] = p10_5th_total
    euc_5th_total = collected_5th_distances.mean()
    print('Overall Euc. Dist (20Hz): ' + str(euc_5th_total))
    metrics['distance'] = euc_5th_total
    
    mean_event_densities = []
    for ll in evdensity_per_layer:
        mean_event_densities.append(np.mean(ll['evs_per_time']))
        
    metrics['mean_event_density'] = mean_event_densities
    metrics['macs_per_layer'] = list(macs_hook.macs_per_layer.numpy())
    
    print(f"Parameters per conv layer: {macs_hook.params_per_layer}")
    print(f"Parameters of the network: {sum(macs_hook.params_per_layer)}")
    
    print(f"\nOutput event density per ReLU layer: {[float(f'{density:.3f}') for density in mean_event_densities]}")
    print(f"MACs per frame for each conv layer: {macs_hook.macs_per_layer}")
    print(f"Total MACs per frame of the network: {sum(macs_hook.macs_per_layer)}")
    print(f"MACs per frame for each conv layer (considering sparsity): {macs_hook.macs_per_layer_with_sparsity}")
    print(f"Total MACs per frame of the network (considering sparsity): {sum(macs_hook.macs_per_layer_with_sparsity)}")
    
    return results, metrics


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

    # centernet version
    #checkpoints = ['/home/scrouzet/AIS2024_CVPR/train_tenn/outputs/2024-03-22/11-05-58/lightning_logs/version_0/checkpoints/last.ckpt']
    #config = ['/home/scrouzet/AIS2024_CVPR/train_tenn/outputs/2024-03-22/11-05-58/lightning_logs/version_0/config.yaml']
    
    # basic version
    # checkpoints = ['/home/scrouzet/AIS2024_CVPR/train_tenn/outputs/2024-03-22/11-42-18/lightning_logs/version_0/checkpoints/last.ckpt']
    # config = ['/home/scrouzet/AIS2024_CVPR/train_tenn/outputs/2024-03-22/11-42-18/lightning_logs/version_0/config.yaml']
    
    test_on_val = True
    grouped_results = []
    for k, checkpoint in enumerate(checkpoints):
        results, _ = check_val_score(checkpoint_path=checkpoint,
                                    checkpoint_config = config[k],
                                    remove_blinks=False,
                                    test_on_val=test_on_val)
        grouped_results.append(results)        

    # plot_results(grouped_results,
    #             test_on_val=test_on_val)

