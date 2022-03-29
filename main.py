# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch

# Functions
import functions

# Other
import os
import argparse
import importlib

def main(rank, args):

    ###############################################################################
    # Init
    ###############################################################################

    # Process rank
    args.rank = rank

    # Print Mode
    if args.rank == 0:
        print("Mode: {}".format(args.mode))

    # Distributed Computing
    if args.distributed:
        torch.cuda.set_device(args.rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)

    # Load Config
    args.config = importlib.import_module(args.config_file.replace(".py", "").replace("/", "."))

    # Load Model
    model = functions.load_model(args)

    # Load Dataset
    dataset_train, dataset_eval = functions.load_datasets(args)

    ###############################################################################
    # Modes
    ###############################################################################

    # Training
    if args.mode == "training":

        model.fit(
            dataset_train=dataset_train, 
            epochs=args.config.epochs, 
            dataset_eval=dataset_eval, 
            eval_steps=args.eval_steps, 
            verbose_eval=args.verbose_eval, 
            initial_epoch=int(args.checkpoint) if args.checkpoint != None else 0, 
            callback_path=args.config.callback_path, 
            steps_per_epoch=args.steps_per_epoch,
            mixed_precision=args.config.mixed_precision,
            accumulated_steps=args.config.accumulated_steps,
            saving_period=args.saving_period,
            eval_period=args.eval_period,
            step_log_period=args.step_log_period
        )

    # Evaluation
    elif args.mode == "evaluation":

        if args.rank == 0:
            print("Evaluation:")

        eval_losses, eval_metrics, val_truths, val_preds = model.evaluate(dataset_eval, eval_steps=args.eval_steps, verbose=args.verbose_eval)

        if args.rank == 0:

            # Eval losses
            for key, value in eval_losses.items():
                print("Eval {}: {:.4f}".format(key, value))

            # Eval metrics
            for key, value in eval_metrics.items():
                print("Eval {}: {:.4f}".format(key, value))  

    # Generation
    elif args.mode == "generation":

        if args.rank == 0:
            print("Generation:")

        model.generate(dataset_eval, saving_path=args.config.generation_path)

    # Stochastic Weight Averaging
    elif args.mode == "swa":

        model.swa(dataset_train, callback_path=args.config.callback_path, start_epoch=args.swa_epochs[0] if args.swa_epochs else None, end_epoch=args.swa_epochs[1] if args.swa_epochs else None, epochs_list=args.swa_epochs_list, update_steps=args.steps_per_epoch, swa_type=args.swa_type, mixed_precision=args.config.mixed_precision)

    # Show Data
    elif args.mode == "show":

        dataset_train.dataset.show()

    # Pass
    elif args.mode == "pass":
        pass

    ###############################################################################
    # Clean
    ###############################################################################

    # Destroy Process Group
    if args.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file",          type=str,   default="configs/EfficientConformerCTCSmall.json",  help="Json configuration file containing model hyperparameters")
    parser.add_argument("-m", "--mode",                 type=str,   default="training",                                 help="Mode : training, validation-clean, test-clean, eval_time-dev-clean, ...")
    parser.add_argument("-i", "--checkpoint",           type=str,   default=None,                                       help="Load model from checkpoint name")
    parser.add_argument("-j", "--num_workers",          type=int,   default=0,                                          help="Number of data loading workers")
    parser.add_argument("--cpu",                        action="store_true",                                            help="Load model on cpu")

    # Distributed
    parser.add_argument("-d", "--distributed",          action="store_true",                                            help="Distributed data parallelization")
    parser.add_argument("--parallel",                   action="store_true",                                            help="Parallelize model using data parallelization")
    parser.add_argument("--world_size",                 type=int,   default=torch.cuda.device_count(),                  help="Number of available GPUs")

    # Training
    parser.add_argument("--steps_per_epoch",            type=int,   default=None,                                       help="Number of steps per epoch")
    parser.add_argument("--saving_period",              type=int,   default=1,                                          help="Model saving every 'n' epochs")
    parser.add_argument("--step_log_period",            type=int,   default=10,                                         help="Training step log period")

    # Eval
    parser.add_argument("--eval_period",                 type=int,   default=1,                                          help="Model evaluation every 'n' epochs")
    parser.add_argument("--batch_size_eval",            type=int,   default=None,                                          help="Evaluation batch size")
    parser.add_argument("--verbose_eval",               type=int,   default=0,                                          help="Evaluation verbose level")
    parser.add_argument("--eval_steps",                  type=int,   default=None,                                       help="Number of evaluation steps")

    # Info
    parser.add_argument("--show_dict",                  action="store_true",                                            help="Show model dict summary")
    
    # SWA
    parser.add_argument("--swa_epochs",                 nargs="+",  default=None,                                       help="Start epoch / end epoch for swa")
    parser.add_argument("--swa_epochs_list",            nargs="+",  default=None,                                       help="List of checkpoints epochs for swa")
    parser.add_argument("--swa_type",                   type=str,   default="equal",                                    help="Stochastic weight averaging type (equal/exp)")
    
    # Parse Args
    args = parser.parse_args()

    # Run main
    if args.distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '10000' # '8888'
        torch.multiprocessing.spawn(main, nprocs=args.world_size, args=(args,))  
    else:
        main(0, args)
