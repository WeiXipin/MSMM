import argparse
import logging
import pathlib
import pprint
import shutil
import sys

import numpy as np
import torch
import torch.utils.data
import tqdm

import dataset

import transformers
import representation
import utils
import copy
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
import os
from torch.utils.data.distributed import DistributedSampler

@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        # default="sod-bar",
        # choices=("sod", "lmd", "lmd_full", "snd"),
        # required=True,
        help="dataset key",
    )
    parser.add_argument(
        "-t", "--train_names", type=pathlib.Path, help="training names"
    )
    parser.add_argument(
        "-v", "--valid_names", type=pathlib.Path, help="validation names"
    )
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input data directory"
    )
    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory"
    )
    # Data
    parser.add_argument(
        "-bs",
        "--batch_size",
        default=2,
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "--use_csv",
        action="store_true",
        help="whether to save outputs in CSV format (default to NPY format)",
    )
    parser.add_argument(
        "--aug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to use data augmentation",
    )
    # Model
    parser.add_argument(
        "--max_seq_len",
        default=1024,
        type=int,
        help="maximum sequence length",
    )
    parser.add_argument(
        "--max_beat",
        default=1024,
        type=int,
        help="maximum number of beats",
    )
    parser.add_argument("--dim", default=512, type=int, help="model dimension")
    parser.add_argument(
        "-l", "--layers", default=6, type=int, help="number of layers"
    )
    parser.add_argument(
        "--heads", default=8, type=int, help="number of attention heads"
    )
    parser.add_argument(
        "--dropout", default=0.2, type=float, help="dropout rate"
    )
    parser.add_argument(
        "--abs_pos_emb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to use absolute positional embedding",
    )
    parser.add_argument(
        "--rel_pos_emb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether to use relative positional embedding",
    )
    # Training
    parser.add_argument(
        "--steps",
        default=200000,
        type=int,
        help="number of steps",
    )
    parser.add_argument(
        "--valid_steps",
        default=1000,
        type=int,
        help="validation frequency",
    )
    parser.add_argument(
        "--early_stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to use early stopping",
    )
    parser.add_argument(
        "-e",
        "--early_stopping_tolerance",
        default=20,
        type=int,
        help="number of extra validation rounds before early stopping",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=0.0005,
        type=float,
        help="learning rate",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        default=5000,
        type=int,
        help="learning rate warmup steps",
    )
    parser.add_argument(
        "--lr_decay_steps",
        default=100000,
        type=int,
        help="learning rate decay end steps",
    )
    parser.add_argument(
        "--lr_decay_multiplier",
        default=0.1,
        type=float,
        help="learning rate multiplier at the end",
    )
    parser.add_argument(
        "--grad_norm_clip",
        default=1.0,
        type=float,
        help="gradient norm clipping",
    )
    # Others
    parser.add_argument("-g", "--gpu", type=int, help="gpu number")
    parser.add_argument(
        "-j",
        "--jobs",
        default=4,
        type=int,
        help="number of workers for data loading",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    parser.add_argument(
        "--local_rank",default=os.getenv('LOCAL_RANK'),  type=int
    )
    
    return parser.parse_args(args=args, namespace=namespace)


def get_lr_multiplier(
    step, warmup_steps, decay_end_steps, decay_end_multiplier
):
    """Return the learning rate multiplier with a warmup and decay schedule.

    The learning rate multiplier starts from 0 and linearly increases to 1
    after `warmup_steps`. After that, it linearly decreases to
    `decay_end_multiplier` until `decay_end_steps` is reached.

    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step > decay_end_steps:
        return decay_end_multiplier
    position = (step - warmup_steps) / (decay_end_steps - warmup_steps)
    return 1 - (1 - decay_end_multiplier) * position

def train_one_epoch_for_enc1(model,pbar,train_iterator,train_iterator1,train_iterator2,
                             train_loader,train_loader1,train_loader2,os,step,recent_losses,device):

    optimizer,scheduler  =  os
    # (pbar := tqdm.tqdm(range(args.valid_steps), ncols=80))
    for batch in pbar:
        # Get next batch
        try:
            batch = next(train_iterator)
            batch1 = next(train_iterator1)
            batch2 = next(train_iterator2)
        except StopIteration:
            # Reinitialize dataset iterator
            train_iterator = iter(train_loader)
            train_iterator1 = iter(train_loader1)
            train_iterator2 = iter(train_loader2)
            batch = next(train_iterator)
            batch1 = next(train_iterator1)
            batch2 = next(train_iterator2)



        # Get input and output pair
        seq = batch["seq"].to(device)
        seq1 = batch1["seq"].to(device)
        seq2 = batch2["seq"].to(device)
        mask = batch["mask"].to(device)
        # mask1 = batch1["mask"].to(device)
        # mask2 = batch2["mask"].to(device)

        # Update the model parameters
        optimizer.zero_grad()
        loss = model(seq,seq1,seq2, mask=mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )
        optimizer.step()
        scheduler.step()

        # Compute the moving average of the loss
        recent_losses.append(float(loss))
        if len(recent_losses) > 10:
            del recent_losses[0]
        train_loss = np.mean(recent_losses)
        pbar.set_postfix(loss=f"{train_loss:8.4f}")
        step+=1
        print(step)
    return step,seq,mask,train_loss

def eval_one_epoch_for_enc1(model,valid_loader,valid_loader1,valid_loader2,device):

    with torch.no_grad():
        total_loss = 0
        total_losses = [0] * 6
        count = 0
        valid_iterator = iter(valid_loader)
        valid_iterator1 = iter(valid_loader1)
        valid_iterator2 = iter(valid_loader2)



        for batch_index in range(len(valid_loader)):
            # Get input and output pair
            batch = next(valid_iterator)
            batch1 = next(valid_iterator1)
            batch2 = next(valid_iterator2)
            seq = batch["seq"].to(device)
            seq1 = batch1["seq"].to(device)
            seq2 = batch2["seq"].to(device)
            mask = batch["mask"].to(device)
            # Pass through the model
            loss, losses = model(seq,seq1,seq2, return_list=True, mask=mask)

            # Accumulate validation loss
            count += len(batch)
            total_loss += len(batch) * float(loss)
            for idx in range(6):
                total_losses[idx] += float(losses[idx])
    return total_loss,total_losses,count,seq,mask

    

def main():
    """Main function."""
    # Parse the command-line arguments
    args = parse_args()


    # Set default arguments

    args.train_names = pathlib.Path('data/lmd/processed/train-names.txt')
    args.train_names1 = pathlib.Path("data/lmd-bar/processed/train-names.txt")
    args.train_names2 = pathlib.Path("data/lmd-track/processed/train-names.txt")
    
    args.valid_names = pathlib.Path(
            f"data/lmd/processed/valid-names.txt"
    )
    args.valid_names1 = pathlib.Path(
            f"data/lmd-bar/processed/valid-names.txt"
    )
    args.valid_names2 = pathlib.Path(
            f"data/lmd-track/processed/valid-names.txt"
    )
    args.in_dir = pathlib.Path(f"data/lmd/processed/notes/")
    args.in_dir1 = pathlib.Path(f"data/lmd-bar/processed/notes/")
    args.in_dir2 = pathlib.Path(f"data/lmd-track/processed/notes/")
    
    args.out_dir = pathlib.Path(f"exp/lmd-bar")

    # Make sure the output directory exists
    args.out_dir.mkdir(exist_ok=True)
    (args.out_dir / "checkpoints_enc123").mkdir(exist_ok=True)

    # Set up the logger
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(args.out_dir / "train.log", "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Log command called
    logging.info(f"Running command: python {' '.join(sys.argv)}")

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Save command-line arguments
    logging.info(f"Saved arguments to {args.out_dir / 'train-args.json'}")
    utils.save_args(args.out_dir / "train-args.json", args)
    # Get the specified device
    if args.local_rank != -1 and args.local_rank !=None:
        torch.cuda.set_device(args.local_rank)
        device=torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl", init_method='env://')

        print("************",dist.get_world_size())
    else:
        device = torch.device(
        f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
        )
    logging.info(f"Using device: {device}")

    # Load the encoding

    encoding = representation.load_encoding(args.in_dir / "encoding.json")


    # Create the dataset and data loader
    logging.info(f"Creating the data loader...")


    train_dataset = dataset.MusicDataset(
            args.train_names,
            args.in_dir,
            encoding,
            max_seq_len=args.max_seq_len,
            max_beat=args.max_beat,
            use_augmentation=args.aug,
            use_csv=args.use_csv,
        )
    train_sampler = DistributedSampler(train_dataset,shuffle=False,seed=1)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=args.jobs,
        collate_fn=dataset.MusicDataset.collate,
        pin_memory=True,
    )
# data1
    train_dataset1 = dataset.MusicDataset(
            args.train_names1,
            args.in_dir1,
            encoding,
            max_seq_len=args.max_seq_len,
            max_beat=args.max_beat,
            use_augmentation=args.aug,
            use_csv=args.use_csv,
        )
    train_sampler1 = DistributedSampler(train_dataset1,shuffle=False,seed=1)
    train_loader1 = torch.utils.data.DataLoader(
        train_dataset1,
        args.batch_size,
        sampler=train_sampler1,
        shuffle=False,
        num_workers=args.jobs,
        collate_fn=dataset.MusicDataset.collate,
        pin_memory=True,
    ) 
# data2
    train_dataset2 = dataset.MusicDataset(
            args.train_names2,
            args.in_dir2,
            encoding,
            max_seq_len=args.max_seq_len,
            max_beat=args.max_beat,
            use_augmentation=args.aug,
            use_csv=args.use_csv,
        )
    train_sampler2 = DistributedSampler(train_dataset2,shuffle=False,seed=1)
    train_loader2 = torch.utils.data.DataLoader(
        train_dataset2,
        args.batch_size,
        sampler=train_sampler2,
        shuffle=False,
        num_workers=args.jobs,
        collate_fn=dataset.MusicDataset.collate,
        pin_memory=True,
    ) 
    

    valid_dataset = dataset.MusicDataset(
            args.valid_names,
            args.in_dir,
            encoding,
            max_seq_len=args.max_seq_len,
            max_beat=args.max_beat,
            use_csv=args.use_csv,
        )
    valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            args.batch_size,
            num_workers=args.jobs,
            collate_fn=dataset.MusicDataset.collate,
        )
# data1
    valid_dataset1 = dataset.MusicDataset(
            args.valid_names1,
            args.in_dir1,
            encoding,
            max_seq_len=args.max_seq_len,
            max_beat=args.max_beat,
            use_csv=args.use_csv,
        )
    valid_loader1 = torch.utils.data.DataLoader(
            valid_dataset1,
            args.batch_size,
            num_workers=args.jobs,
            collate_fn=dataset.MusicDataset.collate,
        )
    
# data2
    valid_dataset2 = dataset.MusicDataset(
            args.valid_names2,
            args.in_dir2,
            encoding,
            max_seq_len=args.max_seq_len,
            max_beat=args.max_beat,
            use_csv=args.use_csv,
        )
    valid_loader2 = torch.utils.data.DataLoader(
            valid_dataset2,
            args.batch_size,
            num_workers=args.jobs,
            collate_fn=dataset.MusicDataset.collate,
        )
    
    
    # Create the model
    logging.info(f"Creating model...")



    model = transformers.MusicXTransformer(
        dim=args.dim,
        encoding=encoding,
        depth=args.layers,
        heads=args.heads,
        max_seq_len=args.max_seq_len,
        max_beat=args.max_beat,
        rotary_pos_emb=args.rel_pos_emb,
        use_abs_pos_emb=args.abs_pos_emb,
        emb_dropout=args.dropout,
        attn_dropout=args.dropout,
        ff_dropout=args.dropout,
    ).to(device)
    '''
    d=torch.load('exp/best_model_enc1.pt')
    b=torch.load('exp/best_model_enc3.pt')

    f=open("d.txt","w")
    for key in d.keys():
        f.write(key+"\n")
        f.write(str(d[key]))
    f.close()
    f=open("b.txt","w")
    for key in b.keys():
        f.write(key+"\n")
        f.write(str(b[key]))
    f.close()
    '''
    '''
    f=open("msd.txt","w")
    for key in model.state_dict().keys():
        f.write(key+"\n")
        f.write(str(model.state_dict()[key]))
    f.close()
    '''


    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = DDP(model, device_ids=[args.local_rank],
                                                output_device=args.local_rank)

    # Summarize the model
    n_parameters = sum(p.numel() for p in model.parameters())
    n_trainables = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logging.info(f"Number of parameters: {n_parameters}")
    logging.info(f"Number of trainable parameters: {n_trainables}")

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr_multiplier(
            step,
            args.lr_warmup_steps,
            args.lr_decay_steps,
            args.lr_decay_multiplier,
        ),
    )

    # Create a file to record losses
    loss_csv = open(args.out_dir / "loss.csv", "w")
    loss_csv.write(
        "step,train_loss,valid_loss,type_loss,beat_loss,position_loss,"
        "pitch_loss,duration_loss,instrument_loss\n"
    )

    # Initialize variables
    step = 0
    min_val_loss = float("inf")
    if args.early_stopping:
        count_early_stopping = 0

    # Iterate for the specified number of steps
    train_iterator = iter(train_loader)
    train_iterator1 = iter(train_loader1)
    train_iterator2 = iter(train_loader2)
    while step < args.steps:

        # Training
        logging.info(f"Training...")
        model.train()
        recent_losses = []
        pbar = tqdm.tqdm(range(args.valid_steps), ncols=80)
        os = (optimizer,scheduler)

        step,seq,mask,train_loss = train_one_epoch_for_enc1(model,pbar,train_iterator,train_iterator1,train_iterator2,
                                                            train_loader,train_loader1,train_loader2,os,step,recent_losses,device)

        # Release GPU memory right away
        del seq, mask

        # Validation
        logging.info(f"Validating...")
        model.eval()
        total_loss,total_losses,count,seq,mask = eval_one_epoch_for_enc1(model,valid_loader,valid_loader1,valid_loader2,device)

        val_loss = total_loss / count
        individual_losses = [l / count for l in total_losses]

        logging.info(f"Validation loss: {val_loss:.4f}")
        logging.info(
            f"Individual losses: type={individual_losses[0]:.4f}, "
            f"beat: {individual_losses[1]:.4f}, "
            f"position: {individual_losses[2]:.4f}, "
            f"pitch: {individual_losses[3]:.4f}, "
            f"duration: {individual_losses[4]:.4f}, "
            f"instrument: {individual_losses[5]:.4f}"
        )

        # Release GPU memory right away
        del seq, mask

        # Write losses to file
        loss_csv.write(
            f"{step},{train_loss},{val_loss},{individual_losses[1]},"
            f"{individual_losses[1]},{individual_losses[2]},"
            f"{individual_losses[3]},{individual_losses[4]},"
            f"{individual_losses[5]}\n"
        )

        # Save the model
        checkpoint_filename = args.out_dir / "checkpoints_enc123" / f"model_enc123_{step}.pt"
        torch.save(model.state_dict(), checkpoint_filename)
        logging.info(f"Saved the model to: {checkpoint_filename}")

        # Copy the model if it is the best model so far
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            shutil.copyfile(
                checkpoint_filename,
                args.out_dir / "checkpoints_enc123" / "best_model_enc123.pt",
            )
            # Reset the early stopping counter if we found a better model
            if args.early_stopping:
                count_early_stopping = 0
        elif args.early_stopping:
            # Increment the early stopping counter if no improvement is found
            count_early_stopping += 1

        # Early stopping
        if (
            args.early_stopping
            and count_early_stopping > args.early_stopping_tolerance
        ):
            logging.info(
                "Stopped the training for no improvements in "
                f"{args.early_stopping_tolerance} rounds."
            )
            break

    # Log minimum validation loss
    logging.info(f"Minimum validation loss achieved: {min_val_loss}")

    # Save the optimizer states
    optimizer_filename = args.out_dir / "checkpoints_enc123" / f"optimizer_enc123_{step}.pt"
    torch.save(optimizer.state_dict(), optimizer_filename)
    logging.info(f"Saved the optimizer state to: {optimizer_filename}")

    # Save the scheduler states
    scheduler_filename = args.out_dir / "checkpoints_enc123" / f"scheduler_enc123_{step}.pt"
    torch.save(scheduler.state_dict(), scheduler_filename)
    logging.info(f"Saved the scheduler state to: {scheduler_filename}")

    # Close the file
    loss_csv.close()


if __name__ == "__main__":
    '''
    多gpu运行命令
    --nproc_per_node: gpu数
    python -m torch.distributed.launch --nproc_per_node=3 train123.py
    '''
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    main()

