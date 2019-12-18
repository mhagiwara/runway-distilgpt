import torch
import torch.nn.functional as F

from transformers import GPT2LMHeadModel, GPT2Tokenizer


class DistilGPT2Model:
    def __init__(self, device='cpu', max_len=1000):
        self.tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        self.model = GPT2LMHeadModel.from_pretrained('distilgpt2')
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.sample = True
        self.max_len = max_len

    def generate(self, prompt):
        context = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long, device=self.device)
        context = context.unsqueeze(0)
        generated = context
        prev_token = context
        past = None

        with torch.no_grad():
            for _ in range(self.max_len):
                next_token_logits, past = self.model(prev_token, past=past)
                next_token_logits = next_token_logits[:, -1, :]

                if self.sample:
                    next_token_log_probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(next_token_log_probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

                generated = torch.cat((generated, next_token), dim=1)
                prev_token = next_token

        text = self.tokenizer.decode(generated.tolist()[0], clean_up_tokenization_spaces=True)
        return text


def main():
    model = DistilGPT2Model()
    print(model.generate('Hi! My name is Masato. I\'m a '))


if __name__ == '__main__':
    main()
