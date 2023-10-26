# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

import torch
import torch.distributed as dist

import wandb
from slicegpt import data_utils, gpu_utils, hf_utils, layernorm_fusion, rotate
from accelerate import Accelerator, infer_auto_device_map, init_empty_weights, dispatch_model
from accelerate.utils import get_balanced_memory
import deepspeed
import gc
import time
import os
import torch.distributed as dist  
import torch.multiprocessing as mp  
from torch.nn.parallel import DistributedDataParallel as DDP 


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="OPT model to load; pass `facebook/opt-125m`.",
        choices=[
            # OPT models
            "facebook/opt-125m",
            # "facebook/opt-350m",
            "facebook/opt-1.3b",
            "facebook/opt-2.7b",
            "facebook/opt-6.7b",
            "facebook/opt-13b",
            "facebook/opt-30b",
            "facebook/opt-66b",
            # "facebook/opt-175b",
            # LLAMA 2 Models
            'meta-llama/Llama-2-7b-hf',
            'meta-llama/Llama-2-13b-hf',
            'meta-llama/Llama-2-70b-hf',
        ],
        default="facebook/opt-125m",
    )
    parser.add_argument(
        "--cal_dataset",
        type=str,
        help="Dataset to calibrate on.",
        choices=["wikitext2", "ptb", "c4"],
        default="wikitext2",
    )
    parser.add_argument(
        "--cal_nsamples",
        type=int,
        help="Number of samples of the calibration data to load.",
        default=128,
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for loading the calibration data.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    parser.add_argument("--sparsity", type=float, default=0.0, help="A measure of how much slicing is applied (in the range [0, 1])")

    parser.add_argument('--hf_token', type=str, default=None)

    parser.add_argument('--local_rank', required=False, type=int, help="Local rank for deepspeed.")

    args = parser.parse_args()
    assert args.sparsity >= 0 and args.sparsity <= 1, "Sparsity should be in the range [0, 1]!"

    return args


def evaluate(rank, world_size, model):
    os.environ['MASTER_ADDR'] = 'localhost'  
    os.environ['MASTER_PORT'] = '29500'  
  
    dist.init_process_group("nccl", rank=rank, world_size=world_size)  
  
    device = torch.device(f"cuda:{rank}")  
    model.to(device)  
    ddp_model = DDP(model, device_ids=[rank])  
  
    input_size = 10  
    num_samples = 100  
    X = torch.randn(num_samples, input_size)  
    y = torch.randn(num_samples, 1)  
    dataset = TensorDataset(X, y)  
    dataloader = DataLoader(dataset, batch_size=32)  
  
    ddp_model.eval()  
    with torch.no_grad():  
        for inputs, targets in dataloader:  
            inputs = inputs.to(device)  
            targets = targets.to(device)  
  
            outputs = ddp_model(inputs)  
  
            loss = torch.nn.MSELoss()(outputs, targets)  
            print(f"Rank {rank} Loss:", loss.item())  
  
    dist.destroy_process_group()  


def main():
    print("Running perplexity evaluation.")

    args = argparser()

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    print(f'{local_rank = }')
    world_size = int(os.getenv('WORLD_SIZE', '1'))

    try:
        wandb.init(project="slicegpt-perplexity", mode='disabled', config=args)
    except wandb.UsageError as e:
        # wandb.init will throw an error if the user is not logged in and the process is running in a non-shell
        # environment, e.g. notebook, IDE, no-shell process, etc. In this case, we want to continue without wandb.
        print(f'Failed to initialize wandb: {e}, continuing without wandb.')
        wandb.init(project="slicegpt", mode='disabled')


    # get model, data
    model, tokenizer = hf_utils.get_model(args.model, args.hf_token)
    
    print(torch.cuda.device_count())
    _, testloader = data_utils.get_loaders(
        dataset_name=args.cal_dataset,
        tokenizer=tokenizer,
        nsamples=args.cal_nsamples,
        seqlen=model.seqlen,
        batch_size=args.batch_size * torch.cuda.device_count(),
        seed=args.seed,
    )

    # ds_engine = deepspeed.init_inference(model, mp_size=world_size, dtype=torch.float)
    mp.spawn(gpu_utils.evaluate_ppl,  
             args=(num_gpus, model),  
             nprocs=num_gpus,  
             join=True) 
    dataset_ppl = gpu_utils.evaluate_ppl(ds_engine.module, testloader, device)
            
    print('Original ppl:', dataset_ppl)
    wandb.log({"original_ppl": dataset_ppl})
    

if __name__ == "__main__":
    main()
