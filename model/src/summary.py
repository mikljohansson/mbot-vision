import argparse

from torchinfo import summary

from src.model import create_model

parser = argparse.ArgumentParser(description='Summarize adcopy')
parser.add_argument('-m', '--model', type=str, help='What model architecture/config to train')
parser.add_argument('-d', '--deploy', action='store_true', help='Show model in deployment configuration')
args = parser.parse_args()

model, cfg = create_model(args.model)

if args.deploy:
    model.deploy()

summary(model, depth=6, input_size=(1, 3, cfg.model.input_size[1], cfg.model.input_size[0]),
        col_names=('input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'))
