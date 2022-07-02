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
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import ImageDataset, denormalize
from src.model import create_model

parser = argparse.ArgumentParser(description='Summarize adcopy')
parser.add_argument('-t', '--train', required=True, help='Directory of training images')
parser.add_argument('-p', '--parallel', type=int, help='Number of worker processes', default=0)
parser.add_argument('--epochs', type=int, help='Number of epochs', default=50)
parser.add_argument('--batch-size', type=int, help='Batch size', default=8)
parser.add_argument('--accumulation-steps', type=int, help='Gradient accumulation steps', default=1)
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
parser.add_argument("--output_dir", type=str, default=None, help="Where to output model and logs")
args = parser.parse_args()

dataset = ImageDataset(args.train)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.parallel)
model = create_model()

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optimizer = AdamW(optimizer_grouped_parameters)

accelerator = Accelerator()
device = accelerator.device
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# Setup logging, we only want one process per machine to log things on the screen.
# accelerator.is_local_main_process is only True for one process per machine.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
logger.info(accelerator.state)

# Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

optimization_steps = int(len(dataloader) * args.epochs / args.accumulation_steps)
cuda_available = torch.cuda.is_available()

# Tensorboard output
logger.info(f'Results will be saved in {args.output_dir}')
writer = SummaryWriter(args.output_dir)

logger.info("***** Running training *****")
logger.info(f'  Number of samples {len(dataset)}')
logger.info(f'  Number of epochs {args.epochs}')
logger.info(f'  Per device batch size {args.batch_size}')
logger.info(f'  Gradient accumulation steps {args.accumulation_steps}')
logger.info(f'  Total batch size {args.batch_size * args.accumulation_steps}')
logger.info(f'  Total optimization steps {optimization_steps}')

pbar = tqdm(total=optimization_steps, unit=' steps', disable=not accelerator.is_local_main_process)
last_logged_image = 0


def normalize_loss(v):
    # Normalize loss in a logscale between 0 and 1
    epsilon = 0.01
    v = torch.log((v / (v.max() + epsilon)) * 10 + 1) / math.log(10 + 1)
    return v.clamp(0, 1)


step = 0

for epoch in range(args.epochs):
    for inputs, targets in dataloader:
        outputs = model(inputs)

        alpha_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
        loss = alpha_loss.mean()
        accelerator.backward(loss)
        step += 1

        if step % args.accumulation_steps == 0 or step == len(dataloader) - 1:
            optimizer.step()
            pbar.update(1)

        writer.add_scalar(f'model/loss', loss, step)
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
            output_mask = torch.sigmoid(outputs[0])

            cells = [
                denormalize(inputs[0]),
                targets[0].repeat(3, 1, 1),
                normalize_loss(alpha_loss[0]).repeat(3, 1, 1),
                output_mask.repeat(3, 1, 1),
            ]

            writer.add_image(f'model/sample', torchvision.utils.make_grid(cells, nrow=1), step)

pbar.close()

if args.output_dir is not None:
    accelerator.wait_for_everyone()

    model_path = os.path.join(args.output_dir, 'eye.pth')
    logger.info(f'Saving model to {model_path}')
    accelerator.save(model, model_path)
