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

# Other
import os

def load_model(args):

    # Model Device
    device = torch.device("cuda:" + str(args.rank) if torch.cuda.is_available() and not args.cpu else "cpu")
    if "cuda" in str(device):
        print("Rank {} device: {}, {}, {}MB".format(args.rank, device, torch.cuda.get_device_properties(device).name, int(torch.cuda.get_device_properties(device).total_memory // 1e6)))
    else:
        print("Rank {} device: {}".format(args.rank, device))

    # Barrier
    if args.distributed:
        torch.distributed.barrier()

    # Set Model Device
    model = args.config.model.to(device)

    # Set EMA Model
    if hasattr(args.config, "ema_tau") and args.rank == 0:
        model.set_ema(args.config.ema_tau)

    # Load Model Checkpoint
    if args.checkpoint is not None:
        model.load(os.path.join(args.config.callback_path, "checkpoints_" + str(args.checkpoint) + ".ckpt"))

    # Barrier
    if args.distributed:
        torch.distributed.barrier()

    # Model Summary
    if args.rank == 0:
        model.summary(show_dict=args.show_dict)

    # Distribute Strategy
    if args.distributed:
        if args.rank == 0:
            print("Parallelize model on", args.world_size, "GPUs")
        model.distribute_strategy(args.rank)

    # Parallel Strategy
    if args.parallel and not args.distributed:
        print("Parallelize model on", torch.cuda.device_count(), "GPUs")
        model.parallel_strategy()

    return model

def load_datasets(args):

    # Training Dataset
    if hasattr(args.config, "training_dataset"):

        # DataLoader
        dataset_train = torch.utils.data.DataLoader(
            dataset=args.config.training_dataset,
            batch_size=args.config.batch_size,
            shuffle=(not args.distributed),
            sampler=torch.utils.data.distributed.DistributedSampler(args.config.training_dataset, num_replicas=args.world_size,rank=args.rank) if args.distributed else None,
            num_workers=args.num_workers,
            collate_fn=args.config.model.collate_fn,
            pin_memory=False,
            drop_last=True
        )
        
        # Loaded Print
        if args.rank == 0:
            print("Training Dataset: {}, {:,} samples - {:,} batches".format(dataset_train.dataset.__class__.__name__, len(dataset_train.dataset), len(dataset_train)))

    else:

        dataset_train = None

    # Evaluation Dataset
    if hasattr(args.config, "evaluation_dataset"):

        # Multiple Evaluation datasets
        if False:#isinstance(evaluation_split, list):

            dataset_eval = {}

            for split in evaluation_split:

                if args.rank == 0:
                    print("Loading evaluation dataset : {} {}".format(training_params["evaluation_dataset"], split))

                dataset = evaluation_dataset(training_params["evaluation_dataset_path"], training_params, tokenizer_params, split, args)

                if args.distributed:
                    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=args.world_size,rank=args.rank)
                else:
                    sampler = None

                dataset = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size_eval, shuffle=(not args.distributed), num_workers=args.num_workers, collate_fn=collate_fn_pad, sampler=sampler, pin_memory=False)
                
                if args.rank == 0:
                    print("Loaded :", len(dataset.dataset), "samples", "/", len(dataset), "batches")

                dataset_eval[split] = dataset

        # One Evaluation dataset
        else:

            # DataLoader
            dataset_eval = torch.utils.data.DataLoader(
                dataset=args.config.evaluation_dataset,
                batch_size=args.batch_size_eval if args.batch_size_eval != None else args.config.batch_size,
                shuffle=(not args.distributed),
                sampler=torch.utils.data.distributed.DistributedSampler(args.config.evaluation_dataset, num_replicas=args.world_size,rank=args.rank) if args.distributed else None,
                num_workers=args.num_workers,
                collate_fn=args.config.model.collate_fn,
                pin_memory=False,
                drop_last=False
            )
            
            # Loaded Print
            if args.rank == 0:
                print("Evaluation Dataset: {}, {:,} samples - {:,} batches".format(dataset_eval.dataset.__class__.__name__, len(dataset_eval.dataset), len(dataset_eval)))
    else:
        dataset_eval = None
    
    return dataset_train, dataset_eval