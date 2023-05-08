import pandas as pd
from datasets import Dataset, DatasetDict
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def load_dataset(config):
    data = {}
    for split in ['train', 'test', 'val']:
        max_rows_cfg = config.data[f'max_{split}']
        df = pd.read_csv(f"data/{split}.csv", nrows=max_rows_cfg)
        # ds_split = 'validation' if split == 'val' else split
        ds = Dataset.from_pandas(df, split=split)
        data[split] = ds

    return DatasetDict(data)

def preprocess_data(dataset, config, tokenizer):
    def preprocess_function(data_row):
        if data_row['article'] is None:
            return None
        prompt_ques = tokenizer.bos_token + data_row['article'] + config.answer_prompt
        prompt_ans = data_row['summary'] # + tokenizer.eos_token
        text = prompt_ques + prompt_ans
        (tokens,) = tokenizer(text, return_tensors="pt", padding=False, add_special_tokens=False),
        data = {
            'text': text,
            'prompt_ques': prompt_ques,
            'prompt_ans': prompt_ans,
            'prompt_ques_tokens': tokenizer(prompt_ques, return_tensors="pt", padding=True, add_special_tokens=False)['input_ids'][0],
            # 'prompt_ans_tokens': tokenizer(prompt_ans, return_tensors="pt", padding=True, add_special_tokens=False)['input_ids'][0],
            'input_ids': tokens['input_ids'][0],
            'attention_mask': tokens['attention_mask'][0],
            **data_row,
        }
        return data
    dataset = dataset.map(preprocess_function, batched=False)

    def filter_function(data_row):
        if data_row is None:
            return False
        # tokens = tokenizer(data_rows['text'], return_length=True)
        # return [tok_len <= config.max_tokens for tok_len in tokens.length]
        return len(data_row['input_ids']) <= config.max_tokens
    
    dataset = dataset.filter(filter_function, batched=False)
    return dataset

def get_data_loaders(dataset, config, tokenizer):
    def collate_padding(batch):
        # batch = {
        #     #**tokenizer(batch[0]['text'], return_tensors="pt", padding='max_length', truncation=True, max_length=2048),
        #     **tokenizer(batch[0]['text'], return_tensors="pt", padding=True),
        #     # **batch[0],
        # }
        batch = { key: [row[key] for row in batch] for key in batch[0].keys() }
        batch['input_ids'] = pad_sequence([torch.tensor(t) for t in batch['input_ids']], padding_value=tokenizer.pad_token_id, batch_first=True)
        batch['attention_mask'] = pad_sequence([torch.tensor(t) for t in batch['attention_mask']], padding_value=0.0, batch_first=True)
        
        batch['labels'] = batch['input_ids'].masked_fill(batch['input_ids'] == tokenizer.pad_token_id, -100)
        for i_ex, prompt_ques_tokens in enumerate(batch['prompt_ques_tokens']):
            batch['labels'][i_ex,:len(prompt_ques_tokens)] = -100 # -100 -> ignore for loss

        return batch
    train_loader = DataLoader(
        dataset["train"], shuffle=True, collate_fn=collate_padding, batch_size=config.training.batch_size, pin_memory=True
    )
    test_loader = DataLoader(dataset['test'], collate_fn=collate_padding, batch_size=config.testing.batch_size, pin_memory=True)
    val_loader = DataLoader(dataset['val'], collate_fn=collate_padding, batch_size=config.testing.batch_size, pin_memory=True)
    return train_loader, test_loader, val_loader

