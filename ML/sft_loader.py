import torch
from torch.utils.data import Dataset

# Assuming you have a tokenizer loaded, e.g., from transformers
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})

class SFTDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = f"Instruction: {item['instruction']}\nResponse:"
        response = item["response"]

        # Tokenize without any special tokens
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False).input_ids
        response_tokens = self.tokenizer(response, add_special_tokens=False).input_ids

        # --- Manually Build the Full Sequence with Special Tokens ---

        # 1. Add BOS token to the start
        input_ids = [self.tokenizer.bos_token_id] + prompt_tokens + response_tokens
        
        # 2. Add EOS token to the end
        input_ids.append(self.tokenizer.eos_token_id)

        # 3. Create labels, masking the prompt and the BOS token
        # The prompt length now includes the BOS token we added.
        prompt_len_with_bos = len(prompt_tokens) + 1 
        labels = [-100] * prompt_len_with_bos + response_tokens + [self.tokenizer.eos_token_id]

        # 4. Truncate if the combined length exceeds max_len
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            labels = labels[:self.max_len]

        # 5. Create attention mask
        attention_mask = [1] * len(input_ids)

        # 6. Pad to max_len
        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:
            # Use tokenizer's pad_token_id. For labels, -100 is standard.
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            labels = labels + [-100] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)