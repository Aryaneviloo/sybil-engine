import torch
from colorama import Fore, Style

class SpeculativeEngine:
    def __init__(self, oracle, sovereign, device):
        self.oracle = oracle
        self.sovereign = sovereign
        self.device = device

        self.K = 5 #how many tokens the smaller model drafts per cycle

    def speculative_step(self, current_ids):
        seq_len = current_ids.size(1)
        draft_ids = current_ids.clone()

        with torch.no_grad():
            for _ in range(self.K):
                oracle_outputs = self.oracle(draft_ids)
                next_oracle_token = torch.argmax(oracle_outputs.logits[:, -1, :], dim =-1, keepdim=True)
                draft_ids = torch.cat([draft_ids, next_oracle_token], dim=-1)
        drafted_tokens = draft_ids[:, seq_len:]

#parallel matrix to matri xmultiplication
        with torch.no_grad():
            sovereign_outputs = self.sovereign(draft_ids)

        sovereign_logits = sovereign_outputs.logits[:, seq_len -1 : -1, :]

        accepted_tokens = []

        for i in range(self.K):
            target_token = torch.argmax(sovereign_logits[:, i, :], dim=-1).item()
            draft_token = drafted_tokens[0, i].item()

#if small model aka oracle matches iwth the bigger one 

            if draft_token == target_token:
                accepted_tokens.append(draft_token)
            else:
                accepted_tokens.append(target_token)
                break
#if small one had 100% accuracy we still need k+1 tokens
        if len(accepted_tokens) == self.K:
            final_target_token = torch.argmax(sovereign_outputs.logits[:, -1, :], dim=-1)
            accepted_tokens.append(final_target_token)
        

        #then get it back to tensor and glue to main
        accepted_tensor = torch.tensor([accepted_tokens], dtype=torch.long).to(self.device)
        new_ids = torch.cat([current_ids, accepted_tensor], dim =-1)

#return the new seq and how many we generated
        return new_ids, len(accepted_tokens)



    

