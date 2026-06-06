import time
import torch
from src.models import SybilModelLoader
from src.engine import SpeculativeEngine
from colorama import Fore, Style, init

init(autoreset = True)

def run_baseline():
    loader = SybilModelLoader()
    engine = SpeculativeEngine(loader.oracle, loader.sovereign, loader.device)
    tokenizer = loader.tokenizer
    device = loader.device

    prompt = "The fundamental nature of power is"
    max_new_tokens = 20

    #adding extra colors so it looks cool
    print(f"\n{Fore.CYAN}=========================================={Style.RESET_ALL}")
    print(f"{Fore.CYAN}    PROJECT SYBIL: INITIATED    {Style.RESET_ALL}")
    print(f"{Fore.CYAN}=========================================={Style.RESET_ALL}\n")

    #Text to tensor cuz why not
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)


#--------------------TEST ! TO CHECK-----------------------------
#----------------------------------------------------------------

    print(f"{Fore.YELLOW}[TEST 1] Standard Autoregressive Generation...{Style.RESET_ALL}") 
    current_ids_base = input_ids.clone()
    start_time = time.time() #necessary to understand the diff

    #autoregression bottleneck loop
    with torch.no_grad(): #this saves memory because it doesnt save information to calculate gradients we only want output
        for _ in range(max_new_tokens):
            outputs = loader.sovereign(current_ids_base)
            next_token_logits = outputs.logits[:, -1, :]
            next_token= torch.argmax(next_token_logits, dim =-1, keepdim=True) #model only outpus the probabilities so argmax finds the highest one by itself 
            current_ids_base = torch.cat([current_ids_base, next_token], dim=-1)

    base_time = time.time() - start_time
    base_tps = max_new_tokens/base_time

  #---------------TEST 2 WITH THE ENGINE OR SMALLER MODEL----------
  #---------------------------------------------------------------


    print(f"\n{Fore.CYAN} [TEST 2] Sybil Engine startss (k=engine.K)......{Style.RESET_ALL}")
    current_ids_spec = input_ids.clone()
    tokens_generated = 0
    start_time_spec = time.time()

    with torch.no_grad():
        while tokens_generated < max_new_tokens:
            current_ids_spec, accepted = engine.speculative_step(current_ids_spec)
            tokens_generated += accepted

    spec_time = time.time() - start_time_spec
    spec_tps = tokens_generated/spec_time


    #---------------------------THE METRICS----------------
    # ---------------------------------------------------



    speedup = spec_time/base_tps
    print(f"\n{Fore.GREEN}--- REPORT ---{Style.RESET_ALL}")
    print(f"Baseline Throughput: {Fore.RED}{base_tps:.2f} tokens/sec{Style.RESET_ALL}")
    print(f"Sybil Throughput:    {Fore.GREEN}{spec_tps:.2f} tokens/sec{Style.RESET_ALL}")
    
    if speedup > 1.0:
        print(f"\nResult: {Fore.GREEN}SUCCESS. Hardware bottleneck bypassed.{Style.RESET_ALL}")
        print(f"Absolute Speedup: {Fore.GREEN}{speedup:.2f}x{Style.RESET_ALL}\n")
    else:
        print(f"\nResult: {Fore.RED}DEGRADATION. Oracle hallucinated too heavily.{Style.RESET_ALL}")
        print(f"Absolute Speedup: {Fore.RED}{speedup:.2f}x{Style.RESET_ALL}\n")

if __name__ == "__main__":
    run_baseline()
