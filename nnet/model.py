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
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Other
from tqdm import tqdm
import os
import time
import copy

# Dictionaries
from nnet.optimizers import optim_dict
from nnet.losses import loss_dict
from nnet.decoders import decoder_dict
from nnet.metrics import metric_dict

from nnet.module import Module

# Collate Functions
from nnet.collate_fn import (
    CollateList
)
from nnet.schedulers import (
    Scheduler,
    ConstantScheduler
)

class Model(Module):

    def __init__(self, name="model"):
        super(Model, self).__init__()

        # Model Attributes
        self.is_distributed = False
        self.rank = 0
        self.is_parallel = False
        self.compiled = False
        self.built = False
        self.name = name
        self.infos = {}
        self.device = torch.device("cpu")
        self.ema_model = None
        self.ema_tau = 0.0

    def distribute_strategy(self, rank):
        object.__setattr__(self, "ddp", torch.nn.parallel.DistributedDataParallel(self, device_ids=[rank]))
        self.rank = rank
        self.is_distributed = True

    def parallel_strategy(self):
        object.__setattr__(self, "dp", torch.nn.DataParallel(self))
        self.is_parallel = True

    def to(self, device):
        self.device = device
        return super(Model, self).to(device)

    def register_buffer_module(self, name, module):
        object.__setattr__(self, name, module)
        self.set_require_grad(module, False)

    def set_ema(self, ema_tau):
        self.register_buffer_module("ema_model", copy.deepcopy(self))
        self.ema_tau = ema_tau

    def compile(self, losses, loss_weights=None, optimizer="Adam", metrics=None, decoders=None, collate_fn=CollateList()):

        # Optimizer
        if isinstance(optimizer, str):
            self.optimizer = optim_dict[optimizer](params=self.parameters())
        else:
            self.optimizer = optimizer

        # Model Step
        self.model_step = self.optimizer.scheduler.model_step

        # Losses
        if isinstance(losses, str):
            self.compiled_losses = loss_dict[losses]()
        elif losses == None:
            self.compiled_losses = []
        else:
            self.compiled_losses = losses

        # Loss Weights
        if loss_weights == None:

            self.compiled_loss_weights = ConstantScheduler(1.0)

        else:

            # Assert List or Dict
            assert isinstance(loss_weights, dict) or isinstance(loss_weights, list)

            # Convert to Scheduler
            if isinstance(loss_weights, dict):
                for key, value in loss_weights.items():
                    if not isinstance(value, Scheduler):
                        loss_weights[key] = ConstantScheduler(value)
            else:
                for i, value in enumerate(loss_weights):
                    if not isinstance(value, Scheduler):
                        loss_weights[i] = ConstantScheduler(value)

            # Assign
            self.compiled_loss_weights = loss_weights

        # Metrics
        if isinstance(metrics, str):
            self.compiled_metrics = metric_dict[metrics]()
        elif metrics == None:
            self.compiled_metrics = []
        else:
            self.compiled_metrics = metrics
            
        # Decoders
        if isinstance(decoders, str):
            self.compiled_decoders = decoder_dict[decoders]()
        elif decoders == None:
            self.compiled_decoders = []
        else:
            self.compiled_decoders = decoders

        # Collate Function
        self.collate_fn = collate_fn

        # Set Compiled to True
        self.compiled = True

    def build(self, outputs):

        # Map to Outputs
        self.losses = self.map_to_outputs(outputs, self.compiled_losses)
        self.loss_weights = self.map_to_outputs(outputs, self.compiled_loss_weights)
        self.decoders = self.map_to_outputs(outputs, self.compiled_decoders)
        self.metrics = self.map_to_outputs(outputs, self.compiled_metrics)

        # Set Built to true
        self.built = True

        #print("Built", self.name)
        #print("losses:", self.losses)
        #print("loss weights:", self.loss_weights)
        #print("metrics:", self.metrics)
        #print("decoders:", self.decoders)

    def map_to_outputs(self, outputs, struct):

        """Convenience method to conform `struct` to `outputs` structure.

        Mappings performed:
            (1) Map a struct to a dict of outputs, using the output names.
            (2) Fill missing struct elements with None.
            (3) Map a single item to all outputs.

        Args:
            outputs: Model outputs predictions dict.
            struct: Arbitrary nested structure (dict, list, item).

        Returns:
            Dict mapping `struct` to `outputs` structure.

        """

        # None
        if struct == None:

            return struct

        # Dictionary
        elif isinstance(struct, dict):

            # Assert struct key in outputs
            for key in struct:
                if not key in outputs:
                    raise Exception("Found unexpected dict key: {}. Valid output names are: {}".format(key, outputs.keys()))

            # Fill missing key with None
            for key in outputs:
                if not key in struct:
                    struct[key] = None

        # List
        elif isinstance(struct, list):

            # Map list items to outputs, Fill missing items with None, Ignore extra items
            struct = {key: struct[i] if i < len(struct) else None for i, key in enumerate(outputs)}

        # Module / Tensor
        else:

            # Map item to all outputs
            struct = {key: struct for key in outputs}

        return struct

    def transfer_to_device(self, struct, device=None):

        # Load Batch elt to model device
        if isinstance(struct, dict):
            return {key: self.transfer_to_device(value) for key, value in struct.items()}
        elif isinstance(struct, list):
            return [self.transfer_to_device(value) for value in struct]
        elif isinstance(struct, torch.Tensor) or isinstance(struct, nn.Module):
            return struct.to(device if device != None else self.device)
        else:
            raise Exception("Incorrect struct type: {}. Must be dict, list module or tensor.".format(type(struct)))

    def forward_model(self, inputs, targets, verbose=False):

        # Init Batch Dict
        batch_losses = {}
        batch_metrics = {}
        batch_truths = {}
        batch_preds = {}
        total_loss = 0.0

        # Additional Targets
        self.additional_targets = {}

        # Forward
        if self.is_distributed:
            outputs = self.ddp(inputs)
        elif self.is_parallel:
            outputs = self.dp(inputs)
        else:
            outputs = self.forward(inputs)

        # Format Outputs to dict
        if isinstance(outputs, dict):
            pass
        elif isinstance(outputs, list):
            outputs = {"output_" + str(key): value for key, value in enumerate(outputs)}
        else:
            outputs = {"output": outputs}

        # Map Targets to Outputs
        targets = self.map_to_outputs(outputs, targets)

        # Append Additional Targets
        for key in self.additional_targets:
            targets[key] = self.additional_targets[key]

        # Build Model
        if not self.built:
            self.build(outputs)

        # Outputs loop
        for key in outputs:

            # Loss Function
            if self.losses[key] != None:

                # Loss key
                key_loss = "loss_" + key

                # Loss
                batch_losses[key_loss] = self.losses[key](targets[key], outputs[key])

                # Weight Loss
                total_loss += batch_losses[key_loss] * self.loss_weights[key].step()

            # Metric Functions
            if self.metrics[key] != None:

                # To list
                if not isinstance(self.metrics[key], list):
                    metrics = [self.metrics[key]]
                else:
                    metrics = self.metrics[key]
                if not isinstance(self.decoders[key], list):
                    decoders = [self.decoders[key] for _ in metrics]
                else:
                    decoders = self.decoders[key]


                for metric, decoder in zip(metrics, decoders):

                    # Metric Key
                    key_metric = metric.name
                    if key_metric in batch_metrics:
                        key_metric += "_" + key

                    # Decoding
                    if decoder != None:
                        batch_truths[key_metric] = decoder(targets[key], from_logits=False) if targets[key] != None else None
                        batch_preds[key_metric] = decoder(outputs[key].detach())
                    else:
                        batch_truths[key_metric] = targets[key]
                        batch_preds[key_metric] = outputs[key].detach()

                    # Prediction Verbose
                    if verbose:
                        print("Groundtruths:\n", batch_truths[key_metric])
                        print("Predictions:\n", batch_preds[key_metric])

                    # Metric
                    batch_metrics[key_metric] = metric(batch_truths[key_metric], batch_preds[key_metric])

        # Module Infos / Losses
        for module in self.modules():
            if hasattr(module, "added_losses"):
                for key, value in module.added_losses.items():
                    batch_losses[key] = value
                    total_loss += value
            if hasattr(module, "added_infos"):
                self.infos.update(module.added_infos)

        # Append Total loss
        if len(batch_losses) > 1:
            batch_losses = dict({"loss": total_loss}, **batch_losses)
        else:
            batch_losses = {"loss": total_loss}

        return batch_losses, batch_metrics, batch_truths, batch_preds

    def train_step(self, inputs, targets, mixed_precision, grad_scaler, accumulated_steps, acc_step):

        # Automatic Mixed Precision Casting (model forward + loss computing)
        #with torch.autocast(device_type=str(self.device).split(":")[0], enabled=mixed_precision): 
        if "cuda" in str(self.device):
            with torch.cuda.amp.autocast(enabled=mixed_precision):
                batch_losses, batch_metrics, _, _ = self.forward_model(inputs, targets)
        else:
            batch_losses, batch_metrics, _, _ = self.forward_model(inputs, targets)

        # Accumulated Steps
        loss = batch_losses["loss"] / accumulated_steps
        acc_step += 1

        # Backward: Accumulate gradients
        grad_scaler.scale(loss).backward()

        # Continue Accumulating
        if acc_step < accumulated_steps:
            return batch_losses, batch_metrics, acc_step

        # Optimizer Step
        grad_scaler.step(self.optimizer)
        grad_scaler.update()

        # Zero Gradients
        self.optimizer.zero_grad()
        acc_step = 0

        # Update Model Infos
        self.infos["lr"] = "{:.2e}".format(self.optimizer.param_groups[0]['lr'])
        self.infos["step"] = self.model_step

        # Cuda Infos
        if "cuda" in str(self.device):
            #self.infos["{}_utilization".format(str(self.device))] = torch.cuda.utilization(self.device)

            # Memory
            memory = ("{}_memory".format(str(self.device)), round(100 * torch.cuda.memory_reserved(self.device) / torch.cuda.get_device_properties(self.device).total_memory, 2))
            if self.is_distributed:
                memory_list = [None for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather_object(memory_list, memory)
                for memory_item in memory_list:
                    self.infos[memory_item[0]] = memory_item[1]
            else:
                self.infos[memory[0]] = memory[1]

        # Update Exp Moving Avg Model
        if self.ema_model != None:
            for param_target, param_net in zip(self.ema_model.parameters(), self.parameters()):
                param_target.mul_(self.ema_tau)
                param_target.add_((1 - self.ema_tau) * param_net.detach())

        return batch_losses, batch_metrics, acc_step  

    def eval_step(self, inputs, targets, verbose=False):

        with torch.no_grad():
            batch_losses, batch_metrics, batch_truths, batch_preds = self.forward_model(inputs, targets, verbose=verbose)

        return batch_losses, batch_metrics, batch_truths, batch_preds

    def num_params(self, module=None):

        if module != None:
            if isinstance(module, list):
                return sum([self.num_params(m) for m in module])
            else:
                return sum([p.numel() for p in module.parameters()])
        else:
            return sum([p.numel() for p in self.parameters()])

    def summary(self, show_dict=False):

        # Model Name
        print("Model name: {}".format(self.name))

        # Number Params
        print("Number Parameters: {:,}".format(self.num_params()))

        # State Dict
        if show_dict:
            self.show_dict()

    def show_dict(self, module=None):

        # Default Dict
        if module != None:
            state_dict = module.state_dict(keep_vars=True)
        else:
            state_dict = self.state_dict(keep_vars=True)

        # Empty Dict
        if state_dict == {}:
            return

        # Show Dict
        max_len = max([len(key) for key in state_dict.keys()]) + 5
        for key, value in state_dict.items():
            print("{} type: {:<12} numel: {:<12} shape: {:<20} mean: {:<12.4f} std: {:<12.4f} dtype: {:<12} device: {}".format(key + " " * (max_len - len(key)), "param" if isinstance(value, nn.Parameter) else "buffer", value.numel(), str(tuple(value.size())), value.float().mean(), value.float().std(), str(value.dtype).replace("torch.", ""), str(value.device)))

    def save(self, path, save_optimizer=True):
        
        # Save Model Checkpoint
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": None if not save_optimizer else {key: value.state_dict() for key, value in self.optimizer.items()} if isinstance(self.optimizer, dict) else self.optimizer.state_dict(),
            "model_step": self.model_step,
            "is_distributed": self.is_distributed or self.is_parallel,
            "ema_model_state_dict": None if self.ema_model == None else self.ema_model.state_dict()
            }, path)

        # Print Model state
        if self.rank == 0:
            print("Model saved at step {}".format(self.model_step))

    def load(self, path):

        # Load Model Checkpoint
        checkpoint = torch.load(path, map_location=self.device)

        # Load Model State Dict
        if checkpoint["is_distributed"] and not self.is_distributed:
            self.load_state_dict({key.replace(".module.", "."):value for key, value in checkpoint["model_state_dict"].items()})
        else:
            self.load_state_dict({key:value for key, value in checkpoint["model_state_dict"].items()})

        # Load Optimizer State Dict
        if checkpoint["optimizer_state_dict"] is not None:

            if isinstance(self.optimizer, dict):
                for key, value in self.optimizer.items():
                    value.load_state_dict(checkpoint["optimizer_state_dict"][key])
            else:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Model Step
        self.model_step = checkpoint["model_step"]

        # Print Model state
        if self.rank == 0:
            print("Model loaded at step {}".format(self.model_step))

    def on_epoch_end(self, saving_period, callback_path, epoch, inputs, targets, writer):

        # Saving Checkpoint
        if saving_period != None and callback_path and self.rank == 0:
            if (epoch + 1) % saving_period == 0:
                self.save(callback_path + "checkpoints_" + str(epoch + 1) + ".ckpt")
        if self.rank == 0:
            print()

    def print_step(self, losses, metrics, infos, epoch_iterator, step):

        # Description
        description = ""

        # Losses
        for key, value in losses.items():
            description += "{}: {:.4f} - ".format(key, value / step)

        # Metrics
        for key, value in metrics.items():
            description += "{}: {:.4f} - ".format(key, value / step)

        # Infos
        for key, value in infos.items():
            description += "{}: {} - ".format(key, value)

        # Set description
        epoch_iterator.set_description(description)

    def log_step(self, losses, metrics, infos, writer, step, tag):

        # Losses
        for key, value in losses.items():
            writer.add_scalar(tag + key, value, step)

        # Metrics
        for key, value in metrics.items():
            writer.add_scalar(tag + key, value, step)

        # Infos
        for key, value in infos.items():
            if isinstance(value, float) or isinstance(value, int):
                writer.add_scalar(tag + key, float(value), step)

    def reduce_losses_metrics(self, losses, metrics):

        # Process Barrier
        torch.distributed.barrier()

        # Losses
        for key, value in losses.items():
            torch.distributed.all_reduce(value)
            losses[key] = value / torch.distributed.get_world_size()

        # Epoch Metrics
        for key, value in metrics.items():
            torch.distributed.all_reduce(value)
            metrics[key] = value / torch.distributed.get_world_size()

        return losses, metrics

    def fit(self, dataset_train, epochs, dataset_eval=None, eval_steps=None, verbose_eval=0, initial_epoch=0, callback_path=None, steps_per_epoch=None, mixed_precision=False, accumulated_steps=1, saving_period=None, eval_period=1, step_log_period=10):

        # Is Compiled
        if not self.compiled:
            if self.is_distributed:
                torch.distributed.destroy_process_group()
            raise Exception("You must compile your model before training/testing.")

        # Mixed Precision Gradient Scaler
        scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision and "cuda" in str(self.device))

        # Init Training
        acc_step = 0

        # Zero Gradients
        self.zero_grad()

        # Callbacks
        if self.rank == 0 and callback_path is not None:

            # Create Callbacks
            if not os.path.isdir(callback_path):
                os.makedirs(callback_path)

            # Create Writer
            writer = SummaryWriter(callback_path + "logs")

        else:

            writer = None

        # Try Catch
        try:

            # Training Loop
            for epoch in range(initial_epoch, epochs):

                # Sync sampler if distributed
                if self.is_distributed:
                    dataset_train.sampler.set_epoch(epoch)

                # Init Iterator
                if self.rank == 0:
                    print("Epoch {}/{}:".format(epoch + 1, epochs))
                    epoch_iterator = tqdm(dataset_train, total=steps_per_epoch * accumulated_steps if steps_per_epoch else None, dynamic_ncols=True)
                else:
                    epoch_iterator = dataset_train

                # Init Epoch Dict
                epoch_losses = {}
                epoch_metrics = {}

                # Clear Infos
                self.infos = {}

                # Training Mode
                self.train()

                # Epoch training loop
                for step, batch in enumerate(epoch_iterator):

                    # Unpack Batch
                    inputs, targets = batch["inputs"], batch["targets"]

                    # Transfer Batch elt to model device
                    inputs = self.transfer_to_device(inputs)
                    targets = self.transfer_to_device(targets)

                    # Train Step
                    batch_losses, batch_metrics, acc_step = self.train_step(inputs=inputs, targets=targets, mixed_precision=mixed_precision, grad_scaler=scaler, accumulated_steps=accumulated_steps, acc_step=acc_step)

                    # Update Batch Loss and Metric
                    for key, value in batch_losses.items():
                        epoch_losses[key] = epoch_losses[key] + value.detach() if key in epoch_losses else value.detach()
                    for key, value in batch_metrics.items():
                        epoch_metrics[key] = epoch_metrics[key] + value if key in epoch_metrics else value

                    # Continue Accumulating
                    if acc_step > 0:
                        continue

                    # Step Print (Rank 0)
                    if self.rank == 0:
                        self.print_step(epoch_losses, epoch_metrics, self.infos, epoch_iterator, step + 1)

                    # Logs Step (Rank 0)
                    if self.rank == 0 and writer is not None and (step + 1) % step_log_period == 0:
                        self.log_step(batch_losses, batch_metrics, self.infos, writer, self.model_step, "Training/batch/")

                    # Step per Epoch
                    if steps_per_epoch is not None:
                        if step + 1 >= steps_per_epoch * accumulated_steps:
                            break

                # Reduce among devices
                if self.is_distributed:
                    epoch_losses, epoch_metrics = self.reduce_losses_metrics(epoch_losses, epoch_metrics)

                # Mean loss
                for key, value in epoch_losses.items():
                    epoch_losses[key] = value / (steps_per_epoch * accumulated_steps if steps_per_epoch is not None else len(dataset_train))

                # Mean Metrics
                for key, value in epoch_metrics.items():
                    epoch_metrics[key] = value / (steps_per_epoch * accumulated_steps if steps_per_epoch is not None else len(dataset_train))

                # Logs Epoch
                if self.rank == 0 and writer is not None:
                    self.log_step(epoch_losses, epoch_metrics, {}, writer, epoch + 1, "Training/epoch/")

                # Clear Infos
                self.infos = {}

                # Evaluation
                if (epoch + 1) % eval_period == 0:

                    # Evaluation Dataset
                    if dataset_eval:

                        # To do: Support Multiple Validation Datasets
                        if isinstance(dataset_eval, dict):

                            pass

                        else:

                            # Evaluate
                            val_losses, val_metrics, val_truths, val_preds = self.evaluate(dataset_eval, eval_steps, verbose_eval)

                            # Print
                            if self.rank == 0:

                                # val losses
                                for key, value in val_losses.items():

                                    # Print
                                    print("val {}: {:.4f}".format(key, value))

                                    # Logs
                                    if writer != None:
                                        writer.add_scalar("Validation/" + key, value, epoch + 1)

                                # val metrics
                                for key, value in val_metrics.items():

                                    # Print
                                    print("val {}: {:.4f}".format(key, value))  

                                    # Logs
                                    if writer != None:
                                        writer.add_scalar("Validation/" + key, value, epoch + 1)

                            # Evaluate EMA model
                            if self.ema_model != None:

                                # Evaluate
                                val_losses, val_metrics, val_truths, val_preds = self.ema_model.evaluate(dataset_eval, eval_steps, verbose_eval)

                                # Validation
                                if self.rank == 0:

                                    # val losses
                                    for key, value in val_losses.items():

                                        # Print
                                        print("ema val {}: {:.4f}".format(key, value))

                                        # Logs
                                        if writer != None:
                                            writer.add_scalar("Validation/ema/" + key, value, epoch + 1)

                                    # val metrics
                                    for key, value in val_metrics.items():

                                        # Print
                                        print("ema val {}: {:.4f}".format(key, value))  

                                        # Logs
                                        if writer != None:
                                            writer.add_scalar("Validation/ema/" + key, value, epoch + 1)

                # On Epoch End
                self.on_epoch_end(saving_period, callback_path, epoch, inputs, targets, writer)

        # Exception Handler
        except Exception as e:

            if self.is_distributed:
                torch.distributed.destroy_process_group()

            if self.rank == 0 and writer is not None:
                writer.add_text('Exceptions:', str(e))

            raise e

    def evaluate(self, dataset_eval, eval_steps=None, verbose=False):

        # Evaluzation Mode
        self.eval()

        # Init Epoch Dict
        epoch_losses = {}
        epoch_metrics = {}

        # tqdm Iterator
        if self.rank == 0:
            eval_iterator = tqdm(dataset_eval, total=eval_steps, dynamic_ncols=True)
        else: 
            eval_iterator = dataset_eval

        # Evaluation Loop
        for step, batch in enumerate(eval_iterator):

            # Unpack Batch
            inputs, targets = batch["inputs"], batch["targets"]

            # Transfer Batch elt to model device
            inputs = self.transfer_to_device(inputs)
            targets = self.transfer_to_device(targets)

            # Eval Step
            batch_losses, batch_metrics, batch_truths, batch_preds = self.eval_step(inputs, targets, verbose)

            # Update Epoch Dict
            for key, value in batch_losses.items():
                epoch_losses[key] = epoch_losses[key] + value if key in epoch_losses else value
            for key, value in batch_metrics.items():
                epoch_metrics[key] = epoch_metrics[key] + value if key in epoch_metrics else value

            # Step print (Rank 0)
            if self.rank == 0:
                self.print_step(epoch_losses, epoch_metrics, self.infos, eval_iterator, step + 1)

            # Evaluation Steps
            if eval_steps:
                if step + 1 >= eval_steps:
                    break

        # Reduce among devices
        if self.is_distributed:
            epoch_losses, epoch_metrics = self.reduce_losses_metrics(epoch_losses, epoch_metrics)

        # Mean loss
        for key, value in epoch_losses.items():
            epoch_losses[key] = value / (eval_steps if eval_steps is not None else len(dataset_eval))

        # Mean Metrics
        for key, value in epoch_metrics.items():
            epoch_metrics[key] = value / (eval_steps if eval_steps is not None else len(dataset_eval))

        return epoch_losses, epoch_metrics, batch_truths, batch_preds

    def swa(self, dataset, callback_path, start_epoch, end_epoch, epochs_list=None, update_steps=None, swa_type="equal", swa_decay=0.9, mixed_precision=False):

        # Create SWA Model
        if swa_type == "equal":
            swa_model = torch.optim.swa_utils.AveragedModel(self)
        elif swa_type == "exp":
            swa_model = torch.optim.swa_utils.AveragedModel(self, avg_fn=lambda averaged_model_parameter, model_parameter, num_averaged: (1 - swa_decay) * averaged_model_parameter + swa_decay * model_parameter)

        if self.rank == 0:
            if epochs_list:
                print("Stochastic Weight Averaging on checkpoints : {}".format(epochs_list))
            else:
                print("Stochastic Weight Averaging on checkpoints : {}-{}".format(start_epoch, end_epoch))

        # Update SWA Model Params
        if epochs_list:

            for epoch in epochs_list:

                # Load Model Checkpoint
                self.load(callback_path + "checkpoints_" + str(epoch) + ".ckpt")

                # Update SWA Model
                swa_model.update_parameters(self)

        else:

            for epoch in range(int(start_epoch), int(end_epoch) + 1):

                # Load Model Checkpoint
                self.load(callback_path + "checkpoints_" + str(epoch) + ".ckpt")

                # Update SWA Model
                swa_model.update_parameters(self)

        # Load SWA Model Params
        self.load_state_dict({key[7:]:value for key, value in swa_model.state_dict().items() if key != "n_averaged"})

        if self.rank == 0:
            print("Updating Batch Normalization Statistics")

        # Init
        self.train()
        if self.rank == 0:
            dataset_iterator = tqdm(dataset, total=update_steps)
        else:
            dataset_iterator = dataset

        # Update Batch Normalization Statistics
        for step, batch in enumerate(dataset_iterator):

            # Unpack Batch
            inputs = batch["inputs"]

            # Transfer Batch elt to model device
            inputs = self.transfer_to_device(inputs)

            # Forward
            with torch.autocast(device_type=str(self.device).split(":")[0], enabled=mixed_precision): 
                with torch.no_grad():
                    self.forward(inputs)

            # update_steps
            if update_steps is not None:
                if step + 1 == update_steps:
                    break

        # Save Model
        if self.rank == 0:
            if epochs_list:
                self.save(callback_path + "checkpoints_swa-" + swa_type + "-" + "list" + "-" + epochs_list[0] + "-"  + epochs_list[-1] + ".ckpt", save_optimizer=False)
            else:
                self.save(callback_path + "checkpoints_swa-" + swa_type + "-" + start_epoch + "-"  + end_epoch + ".ckpt", save_optimizer=False)

        # Barrier
        if self.is_distributed:
            torch.distributed.barrier()

    def generate(self, dataset, saving_path=None):

        # Eval mode
        self.eval()

        # Create Saving Path
        if saving_path != None:

            ctr = 0

            if not os.path.isdir(saving_path):
                os.makedirs(saving_path)

        # Init
        if self.rank == 0:
            epoch_iterator = tqdm(dataset, dynamic_ncols=True)
        else:
            epoch_iterator = dataset

        # Epoch training loop
        for step, batch in enumerate(epoch_iterator):

            # Unpack Batch
            inputs, targets = batch["inputs"], batch["targets"]

            # Transfer Batch elt to model device
            inputs = self.transfer_to_device(inputs)

            # Generate Samples
            samples = self.forward_generate(inputs)

            # Save
            if saving_path != None:

                for b in range(samples.size(0)):
                    torch.save(samples[b], os.path.join(saving_path, "sample_" + str(self.rank) + "_" + str(ctr) + ".torch"))
                    ctr += 1

    def eval_time(self, dataset_eval, eval_steps=None, beam_size=1, rnnt_max_consec_dec_steps=None, profiler=False):

        def decode():

            # Start Timer
            start = time.time()

            # Evaluation Loop
            for step, batch in enumerate(eval_iterator):

                batch = [elt.to(device) for elt in batch]

                # Sequence Prediction
                with torch.no_grad():

                    if beam_size > 1:
                        outputs_pred = self.beam_search_decoding(batch[0], batch[2], beam_size)
                    else:
                        if rnnt_max_consec_dec_steps is not None:
                            outputs_pred = self.gready_search_decoding(batch[0], batch[2], rnnt_max_consec_dec_steps)
                        else:
                            outputs_pred = self.gready_search_decoding(batch[0], batch[2])

                # Evaluation Steps
                if eval_steps:
                    if step + 1 >= eval_steps:
                        break
            # Stop Timer
            return time.time() - start

        # Evaluzation Mode
        self.eval()

        # tqdm Iterator
        if self.rank == 0:
            eval_iterator = tqdm(dataset_eval, total=eval_steps)
        else: 
            eval_iterator = dataset_eval

        # Decoding
        if profiler:
            with torch.autograd.profiler.profile(profile_memory=True) as prof:
                with torch.autograd.profiler.record_function("Model Inference"):
                    timer = decode()
        else:
            timer = decode()

        # Profiler Print
        if profiler:
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        # Return Eval Time in s
        return timer

    def eval_time_encoder(self, dataset_eval, eval_steps=None, profiler=False):

        def forward():

            # Start Timer
            start = time.time()

            for step, batch in enumerate(eval_iterator):

                batch = [elt.to(device) for elt in batch]

                with torch.no_grad():
                    x, x_len, att = self.encoder.forward(batch[0], batch[2])

                # Evaluation Steps
                if eval_steps:
                    if step + 1 >= eval_steps:
                        break

            # Stop Timer
            return time.time() - start

        # Evaluzation Mode
        self.eval()

        # tqdm Iterator
        if self.rank == 0:
            eval_iterator = tqdm(dataset_eval, total=eval_steps)
        else: 
            eval_iterator = dataset_eval

        # Forward
        if profiler:
            with torch.autograd.profiler.profile(profile_memory=True) as prof:
                with torch.autograd.profiler.record_function("Model Inference"):
                    timer = forward()
        else:
            timer = forward()

        # Profiler Print
        if profiler:
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        # Return Eval Time in s
        return timer

    def eval_time_decoder(self, dataset_eval, eval_steps=None, profiler=False):

        def forward():

            # Start Timer
            start = time.time()

            for step, batch in enumerate(eval_iterator):

                batch = [elt.to(device) for elt in batch]

                hidden = None

                for i in range(batch[1].size(1)):
                    with torch.no_grad():
                        _, hidden = self.decoder.forward(batch[1][:, i:i+1], hidden)

                # Evaluation Steps
                if eval_steps:
                    if step + 1 >= eval_steps:
                        break

            # Stop Timer
            return time.time() - start

        # Evaluzation Mode
        self.eval()

        # tqdm Iterator
        if self.rank == 0:
            eval_iterator = tqdm(dataset_eval, total=eval_steps)
        else: 
            eval_iterator = dataset_eval

        # Forward
        if profiler:
            with torch.autograd.profiler.profile(profile_memory=True) as prof:
                with torch.autograd.profiler.record_function("Model Inference"):
                    timer = forward()
        else:
            timer = forward()

        # Profiler Print
        if profiler:
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        # Return Eval Time in s
        return timer

    def set_require_grad(self, networks, require_grad=True):

        if not isinstance(networks, list):
            networks = [networks]

        for network in networks:
            network.requires_grad_(require_grad)