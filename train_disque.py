import argparse
import os

import torch
from lightning import pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.plugins import environments as envs
import wandb

from disque import DisQUEModule

def get_parser():
    # Create parser
    parser = argparse.ArgumentParser(description='Code to run DisQUE training')

    # Add general args
    general_parser = parser.add_argument_group('General')
    general_parser.add_argument('--base_dir', help='Base directory where all outputs are saved', type=str, default='.')
    general_parser.add_argument('--save_checkpoint_path', help='Final checkpoint path (optional)', type=str, default=None)
    general_parser.add_argument('--load_checkpoint_path', help='Initial checkpoint path (optional)', type=str, default=None)
    general_parser.add_argument('--finetune_checkpoint_path', help='Initial checkpoint path for finetuning (optional)', type=str, default=None)
    general_parser.add_argument('--slurm', help='Flag to indicate SLURM is being used', action='store_true', default=False)
    general_parser.add_argument('--use_wandb', help='Flag to use WandB logging', action='store_true', default=False)

    # Add module args
    DisQUEModule.add_module_specific_args(parser)

    # Add training args
    training_parser = parser.add_argument_group('Training')
    training_parser.add_argument('--epochs', help='Maximum number of epochs', type=int, default=100)
    training_parser.add_argument('--batch_size', help='Batch size per process', type=int, default=32)
    training_parser.add_argument('--accum_grad_batches', help='Number of steps to run before gradient accumulation', type=int, default=1)
    training_parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
    training_parser.add_argument('--lr_decay', help='Learning rate decay', type=float, default=0.99)
    training_parser.add_argument('--scale_lr', help='Flag to scale learning rate with batch size', action='store_true', default=False)
    training_parser.add_argument('--find_unused_params', help='Flag to find unused params during optimization', action='store_true', default=False)
    training_parser.add_argument('--accelerator', help='Training accelerator to use', type=str, default='gpu')
    training_parser.add_argument('--devices', help='Number of devices to use (Default: Number of CUDA visible devices)', type=int, default=None)
    training_parser.add_argument('--nodes', help='Number of nodes to use (Default: 1)', type=int, default=1)
    training_parser.add_argument('--strategy', help='Training strategy to use (Default: ddp)', type=str, default='ddp')

    return parser


def main():
    torch.set_float32_matmul_precision('high')

    args = get_parser().parse_args()

    # Fix args
    if args.devices is None:
        if args.accelerator == 'gpu':
            args.devices = torch.cuda.device_count()
        else:
            args.devices = 1

    if args.scale_lr:
        args.lr *= max(1, (args.batch_size * args.devices * args.nodes * args.accum_grad_batches) / 256)  # Define base LR for batch size of 256 and do not attenuate.

    if args.save_checkpoint_path is None:
        args.save_checkpoint_path = os.path.join(args.base_dir, f'DisQUE_{args.epochs}epochs.ckpt')

    model = DisQUEModule(args)

    if args.use_wandb:
        wandb.login()

    # Set up Trainer args
    trainer_kwargs = {}
    trainer_kwargs['max_epochs'] = args.epochs
    trainer_kwargs['accumulate_grad_batches'] = args.accum_grad_batches
    trainer_kwargs['accelerator'] = 'gpu'
    if args.devices is not None:
        trainer_kwargs['devices'] = args.devices
    trainer_kwargs['num_nodes'] = args.nodes
    if args.strategy == 'ddp':
        trainer_kwargs['strategy'] = pl.strategies.DDPStrategy(find_unused_parameters=args.find_unused_params)
    else:
        trainer_kwargs['strategy'] = args.strategy
    if args.slurm:
        trainer_kwargs['plugins'] = [envs.SLURMEnvironment(auto_requeue=False)]

    if args.use_wandb:
        trainer_kwargs['logger'] = pl_loggers.WandbLogger(save_dir=args.base_dir, name=f'DisQUE', project='DisQUE', log_model='all')
    else:
        trainer_kwargs['logger'] = pl_loggers.TensorBoardLogger(save_dir=args.base_dir, name=f'DisQUE_logs')

    trainer_kwargs['callbacks'] = [pl.callbacks.ModelCheckpoint(every_n_train_steps=5000, save_top_k=-1)]

    # Create trainer and fit model
    trainer = pl.Trainer(**trainer_kwargs)

    if args.finetune_checkpoint_path is not None:
        checkpoint = torch.load(args.finetune_checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        print('Loaded checkpoint for fine tuning')

    trainer.fit(model, ckpt_path=args.load_checkpoint_path)
    print('Training done')

    # Save final checkpoint
    if trainer.global_rank == 0:
        trainer.save_checkpoint(args.save_checkpoint_path)
        print('Saved model')

if __name__ == '__main__':
    main()
