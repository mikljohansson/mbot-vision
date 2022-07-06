import argparse
import logging
import os
import time
import math

import psutil
import torch
import torch.nn.functional as F
import torchvision
from accelerate import Accelerator
from ranger21 import Ranger21
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import ImageDataset
from src.model import create_model_cfg

parser = argparse.ArgumentParser(description='Summarize adcopy')
parser.add_argument("-o", "--output", type=str, required=True, help="File to write model pth")
parser.add_argument('-t', '--train', type=str, required=True, help='Directory of training images')
parser.add_argument('-m', '--model', type=str, help='Load pretrained weights from this model pth')
parser.add_argument('-p', '--parallel', type=int, help='Number of worker processes', default=0)
parser.add_argument('--epochs', type=int, help='Number of epochs', default=1)
parser.add_argument('--batch-size', type=int, help='Batch size', default=4)
parser.add_argument('--learning-rate', type=float, help='Learning rate', default=1e-3)
parser.add_argument('--accumulation-steps', type=int, help='Gradient accumulation steps', default=1)
parser.add_argument('--unknown-mask', action='store_true', help='Don\'t apply loss for unknown mask', default=False)
args = parser.parse_args()

output_dir = os.path.dirname(args.output)
model_name = os.path.splitext(os.path.basename(args.output))[0]

accelerator = Accelerator()
device = accelerator.device

# Setup logging, we only want one process per machine to log things on the screen.
# accelerator.is_local_main_process is only True for one process per machine.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

# Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger.info(accelerator.state)

model, cfg = create_model_cfg()

dataset = ImageDataset(args.train, target_size=cfg.model.output_size)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.parallel)
dataloader = accelerator.prepare(dataloader)

if args.model:
    logger.info(f'Loading pretrained weights from {args.model}')
    model.load_state_dict(torch.load(args.model))

optimizer = Ranger21(model.parameters(),
                     lr=args.learning_rate,
                     num_epochs=args.epochs,
                     num_batches_per_epoch=(len(dataloader) / args.accumulation_steps))
model, optimizer = accelerator.prepare(model, optimizer)

optimization_steps = int(len(dataloader) * args.epochs / args.accumulation_steps)
cuda_available = torch.cuda.is_available()

# Tensorboard output
logger.info(f'Results will be saved in {output_dir}')
writer = SummaryWriter(output_dir)

logger.info("***** Running training *****")
logger.info(f'  Number of samples {len(dataset)}')
logger.info(f'  Number of epochs {args.epochs}')
logger.info(f'  Per device batch size {args.batch_size}')
logger.info(f'  Gradient accumulation steps {args.accumulation_steps}')
logger.info(f'  Total batch size {args.batch_size * args.accumulation_steps * accelerator.num_processes}')
logger.info(f'  Total optimization steps {optimization_steps}')


def normalize_loss(v):
    # Normalize loss in a logscale between 0 and 1
    epsilon = 0.1
    v = torch.log((v / (v.max() + epsilon)) * 5 + 1) / math.log(5 + 1)
    return v.clamp(0, 1)


def downsample_like(t, like, mode='bilinear'):
    if t.shape[-2:] != like.shape[-2:]:
        return torch.nn.functional.interpolate(t, size=like.shape[-2:], mode=mode, align_corners=(False if mode != 'nearest' else None))
    return t


def upsample_like(t, like, mode='bilinear'):
    if t.shape[-2:] != like.shape[-2:]:
        return torch.nn.functional.interpolate(t, size=like.shape[-2:], mode=mode, align_corners=(False if mode != 'nearest' else None))
    return t


def calculate_loss(outputs, targets, unknown_mask, z_loss=1e-5):
    alpha_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')

    if args.unknown_mask:
        loss = (alpha_loss * (1. - unknown_mask)).mean()
    else:
        loss = alpha_loss.mean()

    # Add a separate loss to keep the logits from drifting too far from zero and encourage the
    # logits to be normalized log-probabilities. This might also help prevent NaN's
    # https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
    z = z_loss * outputs.logsumexp(dim=(2, 3)).square().sum(1).mean()
    loss += z

    return loss, alpha_loss, z


step = 0
last_logged_image = 0
pbar = tqdm(total=optimization_steps, unit=' steps', disable=not accelerator.is_local_main_process)

for epoch in range(args.epochs):
    for inputs, targets, unknown_mask in dataloader:
        outputs = model(inputs)

        loss, alpha_loss, z_loss = calculate_loss(outputs, targets, unknown_mask)
        accelerator.backward(loss)
        step += 1

        if step % args.accumulation_steps == 0 or step == len(dataloader) - 1:
            optimizer.step()
            pbar.update(1)

        writer.add_scalar(f'{model_name}/loss', loss, step)
        writer.add_scalar(f'{model_name}/z_loss', z_loss, step)
        writer.add_scalar(f'system/CPU utilization %', psutil.cpu_percent(), step)
        writer.add_scalar(f'system/Host memory usage %', psutil.virtual_memory().percent, step)

        if cuda_available:
            cuda_free_mem, cuda_total_mem = torch.cuda.mem_get_info()
            writer.add_scalar(f'system/GPU utilization %', torch.cuda.utilization(), step)
            writer.add_scalar(f'system/GPU memory usage %', round((cuda_total_mem - cuda_free_mem) / cuda_total_mem * 100.), step)
            writer.add_scalar(f'system/Torch reserved memory %', round(torch.cuda.memory_reserved() / cuda_total_mem * 100.), step)
            writer.add_scalar(f'system/Torch allocated memory %', round(torch.cuda.memory_allocated() / cuda_total_mem * 100.), step)

        if last_logged_image < time.time() - 1:
            last_logged_image = time.time()
            output_target = upsample_like(targets[[0]], inputs[[0]], mode='nearest').detach().cpu()
            output_unknown_mask = upsample_like(unknown_mask[[0]], inputs[[0]], mode='nearest').detach().cpu()
            output_loss = upsample_like(normalize_loss(alpha_loss[[0]]), inputs[[0]], mode='nearest').detach().cpu()
            output_masked_loss = upsample_like(normalize_loss(alpha_loss[[0]] * (1. - unknown_mask[[0]])), inputs[[0]], mode='nearest').detach().cpu()
            output_mask = upsample_like(torch.sigmoid(outputs[[0]]), inputs[[0]], mode='nearest').detach().cpu()

            cells = [
                inputs[0].detach().cpu(),
                output_target[0].repeat(3, 1, 1),
                output_unknown_mask[0].repeat(3, 1, 1),

                output_loss[0].repeat(3, 1, 1),
                output_mask[0].repeat(3, 1, 1),
                output_masked_loss[0].repeat(3, 1, 1),
            ]

            writer.add_image(f'{model_name}/sample', torchvision.utils.make_grid(cells, nrow=3), step)

pbar.close()

accelerator.wait_for_everyone()
unwrapped_model = accelerator.unwrap_model(model)

logger.info(f'Saving model to {args.output}')
accelerator.save(unwrapped_model.state_dict(), args.output)
