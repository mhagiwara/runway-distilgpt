import torch
import runway
from runway.data_types import number, text
from distilgpt2_model import DistilGPT2Model


@runway.setup(options={'max_len': number(default=1000, min=100, max=5000)})
def setup(opts):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device={device}, max_len={opts['max_len']}")
    model = DistilGPT2Model(device=device, max_len=opts['max_len'])
    return model


inputs = {'prompt': text}
outputs = {'generated': text}


@runway.command('generate', inputs=inputs, outputs=outputs, description='Generate an image.')
def generate(model, input_args):
    return model.generate(input_args['prompt'])


if __name__ == '__main__':
    runway.run(host='0.0.0.0', port=9000)
