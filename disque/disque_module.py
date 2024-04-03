import os
from typing import Any
import torch
from lightning import pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from torch import Tensor
import wandb
import numpy as np

from .datasets import SDRSemiGridDataset, HDRSemiGridDataset
from .models import ContentEncoder, AppearanceEncoder, Decoder
from .criteria.gen_loss import ReconLoss, ContrastLoss
from .criteria.norm_loss import NormLoss


class DisQUEModule(pl.LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        model_parser = parent_parser.add_argument_group('DisQUE')
        model_parser.add_argument('--embed_dim', help='Dimension of embedding', type=int, default=2048)
        model_parser.add_argument('--dataset', help='Dataset to train on', type=str, default='sdr', choices=['sdr', 'hdr'])
        model_parser.add_argument('--log_batches', help='Log images every n batches', type=int, default=500)
        ContrastLoss.add_module_specific_args(parent_parser)
        ReconLoss.add_module_specific_args(parent_parser)
        NormLoss.add_module_specific_args(parent_parser)

    def __init__(self, args):
        super().__init__()

        self.content_enc = ContentEncoder(args.embed_dim)
        self.appearance_enc = AppearanceEncoder(args.embed_dim)
        self.dec = Decoder(args.embed_dim, add_one=True)

        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.accum_grad_batches = args.accum_grad_batches
        self.optimize_contrast = (args.lam_contrast != 0)

        self.contrast_loss_cont = ContrastLoss(args, args.embed_dim if self.optimize_contrast else -1)  # skips projection head
        self.contrast_loss_app = ContrastLoss(args, args.embed_dim if self.optimize_contrast else -1)  # skips projection head

        self.recon_loss = ReconLoss(args)
        self.norm_loss_app = NormLoss(args)

        # Edit to set the paths to text files containing lists of SDR/HDR images
        image_list_paths = {
            'sdr': 'sdr_image_list.txt',
            'hdr': 'hdr_image_list.txt',
        }

        # Edit to set the paths of the image datasets.
        # Do not add a '/' at the end of the dataset dirs.
        image_list_base_dirs = {
            'sdr': 'sdr_dataset',
            'hdr': 'hdr_dataset',
        }

        DatasetClasses = {
            'sdr': SDRSemiGridDataset,
            'hdr': HDRSemiGridDataset,
        }

        self._validation_step_outputs = []
        self._training_step_cont_embeds = ([], [], [], [])
        self._training_step_app_embeds = ([], [], [], [])
        self._train_step_loss = 0
        self.log_batches = args.log_batches

        self.train_dataset = DatasetClasses[self.dataset](image_list_paths[self.dataset], base_dir=image_list_base_dirs[self.dataset], mode='train')
        self.val_dataset = DatasetClasses[self.dataset](image_list_paths[self.dataset], base_dir=image_list_base_dirs[self.dataset], mode='val')

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.lr_decay)
        return [optimizer], [scheduler]

    def _get_predictions(self, xs):
        conts = tuple(self.content_enc(x) for x in xs)
        apps = tuple(self.appearance_enc(x) for x in xs)
        cont_embeds = tuple(torch.cat([cont_comp.mean((2, 3)) for cont_comp in cont], dim=1) for cont in conts)
        app_embeds = tuple(torch.cat(app, dim=1) for app in apps)
        delta_apps_1 = tuple(a - a_ref for a_ref, a in zip(apps[0], apps[1]))
        delta_apps_2 = tuple(a - a_ref for a_ref, a in zip(apps[2], apps[3]))
        apps_cross = []
        apps_cross.append(tuple(a - delta_a for a, delta_a in zip(apps[1], delta_apps_2)))
        apps_cross.append(tuple(a + delta_a for a, delta_a in zip(apps[0], delta_apps_2)))
        apps_cross.append(tuple(a - delta_a for a, delta_a in zip(apps[3], delta_apps_1)))
        apps_cross.append(tuple(a + delta_a for a, delta_a in zip(apps[2], delta_apps_1)))
        conts_cross = [conts[1], conts[0], conts[3], conts[2]]
        ys = tuple(self.dec(cont, app) for cont, app in zip(conts, apps))
        y_crosss = tuple(self.dec(cont, app) for cont, app in zip(conts_cross, apps_cross))
        return ys, y_crosss, cont_embeds, app_embeds

    def _log_example_images(self):
        x_train_example = tuple(map(lambda x: x.unsqueeze(0).to(self.device), self.train_dataset[np.random.randint(len(self.train_dataset))]))
        x_val_example = tuple(map(lambda x: x.unsqueeze(0).to(self.device), self.val_dataset[np.random.randint(len(self.val_dataset))]))

        with torch.no_grad():
            y_train_example, y_cross_train_example, _, _ = self._get_predictions(x_train_example)
            y_val_example, y_cross_val_example, _, _ = self._get_predictions(x_val_example)

        image_dict = {}
        image_dict['train_x_11'], image_dict['train_x_12'], image_dict['train_x_21'], image_dict['train_x_22'] = x_train_example
        image_dict['val_x_11'], image_dict['val_x_12'], image_dict['val_x_21'], image_dict['val_x_22'] = x_val_example
        image_dict['train_y_11'], image_dict['train_y_12'], image_dict['train_y_21'], image_dict['train_y_22'] = y_train_example
        image_dict['val_y_11'], image_dict['val_y_12'], image_dict['val_y_21'], image_dict['val_y_22'] = y_val_example
        image_dict['train_y_cross_11'], image_dict['train_y_cross_12'], image_dict['train_y_cross_21'], image_dict['train_y_cross_22'] = y_cross_train_example
        image_dict['val_y_cross_11'], image_dict['val_y_cross_12'], image_dict['val_y_cross_21'], image_dict['val_y_cross_22'] = y_cross_val_example

        image_dict = {key: image_dict[key].squeeze().cpu().detach().numpy() for key in image_dict}

        if isinstance(self.logger, pl_loggers.TensorBoardLogger):
            for key, img in image_dict.items():
                self.logger.experiment.add_image(f'epoch_{self.current_epoch}/{key}', img)
        elif isinstance(self.logger, pl_loggers.WandbLogger):
            self.logger.experiment.log({key: wandb.Image(np.transpose(np.clip(image_dict[key], 0, 1), (1, 2 ,0))) for key in image_dict})

    @staticmethod
    def _transform_cont_loss_terms(loss_terms):
        return dict(zip(map(lambda x: f'{x}_cont', loss_terms.keys()), loss_terms.values()))

    @staticmethod
    def _transform_app_loss_terms(loss_terms):
        return dict(zip(map(lambda x: f'{x}_app', loss_terms.keys()), loss_terms.values()))

    def _get_contrast_loss(self, cont_embeds, app_embeds):
        loss_terms = {}
        c_11, c_12, c_21, c_22 = cont_embeds
        a_11, a_12, a_21, a_22 = app_embeds

        c_1 = torch.cat([c_11, c_21], dim=0)
        c_2 = torch.cat([c_12, c_22], dim=0)
        a_1 = torch.cat([a_11, a_12], dim=0)
        a_2 = torch.cat([a_21, a_22], dim=0)

        cont_loss, contrast_loss_terms_cont = self.contrast_loss_cont(c_1, c_2, return_terms=True)
        app_loss, contrast_loss_terms_app = self.contrast_loss_app(a_1, a_2, return_terms=True)
        loss_terms.update(self._transform_cont_loss_terms(contrast_loss_terms_cont))
        loss_terms.update(self._transform_app_loss_terms(contrast_loss_terms_app))
        return (cont_loss, app_loss), loss_terms

    def training_step(self, batch, batch_idx):
        xs = batch
        ys, y_crosss, cont_embeds, app_embeds = self._get_predictions(xs)

        is_post_accum_step = batch_idx % self.accum_grad_batches == 0
        if is_post_accum_step:
            for train_step_embed in self._training_step_cont_embeds:
                train_step_embed.clear()
            for train_step_embed in self._training_step_app_embeds:
                train_step_embed.clear()
            self._train_step_loss = 0

        for train_step_embed, cont_embed in zip(self._training_step_cont_embeds, cont_embeds):
            train_step_embed.append(cont_embed)
        for train_step_embed, app_embed in zip(self._training_step_app_embeds, app_embeds):
            train_step_embed.append(app_embed)

        loss_terms = {}

        recon_loss, recon_loss_terms = self.recon_loss(xs, ys, y_crosss, return_terms=True)
        loss_terms.update(recon_loss_terms)
        self._train_step_loss = self._train_step_loss + recon_loss

        norm_loss, norm_loss_terms = self.norm_loss_app(torch.cat(app_embeds, -1), return_terms=True)
        loss_terms.update(norm_loss_terms)
        self._train_step_loss = self._train_step_loss + norm_loss

        # Using accum_grad_batches > 1 has high memory requirements when using two encoders and a decoder.
        # Use with CAUTION!
        is_accum_step = (batch_idx % self.accum_grad_batches == self.accum_grad_batches - 1)
        if is_accum_step:
            cont_embeds = tuple(torch.cat(train_step_embed, 0) for train_step_embed in self._training_step_cont_embeds)
            app_embeds = tuple(torch.cat(train_step_embed, 0) for train_step_embed in self._training_step_app_embeds)
            (contrast_loss_cont, contrast_loss_app), contrast_loss_terms = self._get_contrast_loss(cont_embeds, app_embeds)
        else:
            contrast_loss_terms_cont = self.contrast_loss_cont.get_zero_loss_terms()
            contrast_loss_terms_app = self.contrast_loss_app.get_zero_loss_terms()
            contrast_loss_terms = {}
            contrast_loss_terms.update(self._transform_cont_loss_terms(contrast_loss_terms_cont))
            contrast_loss_terms.update(self._transform_app_loss_terms(contrast_loss_terms_app))
            contrast_loss_cont = torch.tensor(0.0, requires_grad=True)
            contrast_loss_app = torch.tensor(0.0, requires_grad=True)
        loss_terms.update(contrast_loss_terms)

        if is_accum_step:
            loss = self._train_step_loss / self.accum_grad_batches
            if self.optimize_contrast:
                loss += contrast_loss_app + contrast_loss_cont
            loss_terms['tot_loss'] = loss
        else:
            loss = torch.tensor(0.0, requires_grad=True)

        self.log_dict(loss_terms)

        if batch_idx % self.log_batches == 0:
            self._log_example_images()

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        xs = batch
        loss_terms = {}
        with torch.no_grad():
            xs = batch
            ys, y_crosss, cont_embeds, app_embeds = self._get_predictions(xs)
            _, recon_loss_terms = self.recon_loss(xs, ys, y_crosss, return_terms=True)
            loss_terms.update(recon_loss_terms)
            _, contrast_loss_terms = self._get_contrast_loss(cont_embeds, app_embeds)
            loss_terms.update(contrast_loss_terms)
            loss_terms = {key: loss_terms[key].cpu().item() for key in loss_terms}
        self._validation_step_outputs.append(loss_terms)

    def on_validation_epoch_end(self):
        mean_stats_dict = {}
        for key in self._validation_step_outputs[0]:
            mean_stats_dict[f'val_{key}'] = np.float32(np.mean([output[key] for output in self._validation_step_outputs]))
        log_data = {
            'epoch': self.current_epoch,
        }
        log_data.update(mean_stats_dict)

        print(' '.join(f'{key.upper()}: {log_data[key]}' for key in log_data))

        self.log_dict(log_data)
        self._validation_step_outputs.clear()

    def forward(self, x, return_std=True):
        cont = self.content_enc(x)
        cont_embed = torch.cat([torch.mean(cont_comp, (2, 3)) for cont_comp in cont] + [torch.std(cont_comp, (2, 3)) for cont_comp in cont], dim=1)
        app = self.appearance_enc(x, return_std=return_std)
        app_embed = torch.cat(app, dim=1)
        return cont_embed, app_embed

    def backward(self, loss: Any) -> None:
        if isinstance(loss, Tensor) and loss.requires_grad:
            loss.backward(retain_graph=True)

    def prepare_data(self) -> None:
        pass

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle=True
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=1,
            shuffle=False
        )
