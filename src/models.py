import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from colorama import Fore, Style

class SybilModelLoader:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n{Fore.RED} === Starting Sybil Engine on [{self.device.upper()}] === {Style.RESET_ALL}")

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        print (f"{Fore.CYAN} Loading the smaller model aka The Oracle {Style.RESET_ALL}")
        self.oracle = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        self.oracle.eval()

        print(f"{Fore.BLUE} Loading the medium model {Style.RESET_ALL}")
        self.sovereign = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(self.device)
        self.sovereign.eval()

        print(f"Neural Ntwroks are loaded in the memory")


