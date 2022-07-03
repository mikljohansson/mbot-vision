from torchinfo import summary

from src.model import create_model

model = create_model()
summary(model, depth=5, input_size=(1, 3, 160, 120), col_names=('input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'))
