import runway
from runway.data_types import text
from distilgpt2_model import DistilGPT2Model


@runway.setup(options={})
def setup(opts):
    model = DistilGPT2Model()
    return model


inputs = {'prompt': text}
outputs = {'generated': text}


@runway.command('generate', inputs=inputs, outputs=outputs, description='Generate an image.')
def generate(model, input_args):
    return model.generate(input_args['prompt'])


if __name__ == '__main__':
    runway.run(host='0.0.0.0', port=9000)
