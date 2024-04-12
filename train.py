import os
import csv
from datetime import datetime
from functools import partial
from pathlib import Path

import hydra
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf as OC
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim import AdamW
from torch.utils.data import DataLoader

import losses
from eye_dataset import EyeTrackingDataset
from tenn_model import TennSt
from generate_val_results import check_val_score

filename = 'results_ablation.csv'

def flatten_2levels_dict(d, parent_key='', sep='_'):
    """
    Flatten a nested dictionary.
    """
    items = {}
    for pk, pv in d.items():
        parent_key = pk + sep
        for ck, cv in pv.items():
            items[pk+'__'+ck] = cv
    return items


class CustomModule(LightningModule):
    def __init__(self, data_path, config):
        super().__init__()
        self.data_path = data_path
        self.config = config
        
        self.batch_size = config.trainer.batch_size
        epochs = config.trainer.epochs
        detector_head = config.model.detector_head
        activity_regularization = config.trainer.activity_regularization
        
        self.model = torch.compile(TennSt(**OC.to_container(config.model)))
        
        self.trainset = EyeTrackingDataset(data_path, 'train', config.trainer.device, **OC.to_container(config.dataset))
        self.valset = EyeTrackingDataset(data_path, 'val', config.trainer.device, **OC.to_container(config.dataset)) # frames_per_segment=127, device=device)
        
        num_steps_per_epoch = len(self.trainset) // self.batch_size
        self.total_train_steps = epochs * num_steps_per_epoch
        
        self.loss_fn = losses.Losses(detector_head, activity_regularization, self.model)
        self.metric_fn = partial(losses.p10_acc, detector_head=detector_head)
            
    def forward(self, input):
        return self.model(input)
    
    def _log(self, name, metric):
        self.log(name, metric, 
                 on_step=False, on_epoch=True, prog_bar=True)
    
    def on_train_start(self):
        log_dir = Path(self.trainer.logger.log_dir)
        OC.save(self.config, log_dir / 'config.yaml')
        
    #def on_train_end(self):
        # torch.set_grad_enabled(False)
        
        # log_dir = Path(self.trainer.logger.log_dir)

        # testset = EyeTrackingDataset(self.data_path, 'test', self.config.trainer.device, 
        #                              **OC.to_container(self.config.dataset))

        # predictions = []
        # for (event, _, _) in testset:
        #     pred = self(event[None, ...])
        #     if self.config.model.detector_head:
        #         pred = losses.process_detector_prediction(pred)
        #     else:
        #         pred = torch.sigmoid(pred)
        #     predictions.append(pred.detach().squeeze(0)[..., ::5].cpu().numpy())
            
        # predictions = np.concatenate(predictions, axis=-1).T  # (frames, 2)
        # predictions[:, 0] *= 80
        # predictions[:, 1] *= 60
        # predictions = np.concatenate([np.arange(len(predictions))[:, None], predictions], axis=1)

        # df = pd.DataFrame(predictions, columns=['row_id', 'x', 'y'])
        # df.to_csv(log_dir / 'submission.csv', index=False)
    
    
    def training_step(self, batch, batch_idx):
        event, center, openness = batch
        pred = self(event)        
        loss = self.loss_fn(pred, center, openness)
        metric, metric_noblinks, distance = self.metric_fn(pred, center, openness)
        self._log('train_loss', loss)
        self._log('train_metric', metric)
        self._log('train_metric_noblinks', metric_noblinks)
        self._log('train_distance', distance)
        return loss

    def validation_step(self, batch, batch_idx):
        event, center, openness = batch
        pred = self(event)        
        loss = self.loss_fn(pred, center, openness)
        metric, metric_noblinks, distance = self.metric_fn(pred, center, openness)
        self._log('val_loss', loss)
        self._log('val_metric', metric)
        self._log('val_metric_noblinks', metric_noblinks)
        self._log('val_distance', distance)
            
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=0.002, weight_decay=0.001)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, round(self.total_train_steps * 0.025), self.total_train_steps, eta_min=1e-5)
        
        scheduler = {'scheduler': scheduler, 
                     'interval': 'step', 
                     'frequency': 1}
        return [optimizer], [scheduler]
            
    def train_dataloader(self):
        return DataLoader(self.trainset, shuffle=True, drop_last=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valset, shuffle=False, drop_last=False, batch_size=self.batch_size)
    

@hydra.main(version_base='1.1', config_path=".", config_name="config.yaml")
def main(config: OC):
    data_path = Path(__file__).parent / 'event_data'
    module = CustomModule(data_path, config)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_metric', 
        mode='max', 
        save_last=True, 
        every_n_epochs=1, 
        filename='{epoch}-{val_metric:.2f}', 
    )



    trainer = Trainer(
        max_epochs=config.trainer.epochs, 
        gradient_clip_val=1, 
        accelerator='gpu', 
        devices=[config.trainer.device], 
        benchmark=True, 
        log_every_n_steps=5, 
        callbacks=[checkpoint_callback], 
        precision='16-mixed', 
    )

    timestamp_start = datetime.now()
    trainer.fit(module)
    timestamp_end = datetime.now()

    log_dir = Path(trainer.log_dir)
    _, resmetrics = check_val_score(log_dir / "checkpoints" / "last.ckpt",
                              log_dir / "config.yaml",
                              remove_blinks=False, test_on_val=True)
    resdict = flatten_2levels_dict(config)
    resdict['time_start'] = timestamp_start.isoformat()
    resdict['time_end'] = timestamp_end.isoformat()
    resdict['training_duration'] = int((timestamp_end-timestamp_start).total_seconds()/60) # training duration in minutes
    resdict['params'] = sum(p.numel() for p in module.parameters())
    resdict['params_trainable'] = sum(p.numel() for p in module.parameters() if p.requires_grad)

    resdict['logdir'] = trainer.log_dir
    resdict['val_p10'] = resmetrics['p10']
    resdict['val_distance'] = resmetrics['distance']
    resdict['event_density'] = resmetrics['mean_event_density']
    resdict['macs'] = resmetrics['macs_per_layer']

    # Write the header and results to the CSV file
    with open(os.path.dirname(os.path.realpath(__file__))+'/'+filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=resdict.keys())
        if csvfile.tell() == 0:
            writer.writeheader()  # Write header only if file is empty
        row = {**resdict}
        writer.writerow(row)
    print("\n\n")


if __name__ == "__main__":
    main()
