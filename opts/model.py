from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
import torch
from tqdm import tqdm

from opts.utils import device


def load_tokenizer(config):
    return AutoTokenizer.from_pretrained(config.model.name)

DATA_KEYS = ['input_ids', 'attention_mask', 'labels']

class OPTSModel(torch.nn.Module):
    def __init__(self, config, prev_model=None, load_from=None):
        super().__init__()
        self.cfg = config

        if prev_model:
            self.model = prev_model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(config.model.name)

            for param in self.model.parameters():
                param.requires_grad = False  # freeze the model
            
            if load_from is None:
                print(f"Creating new PEFT adapter")
                peft_config = self.get_peft_config(config)
                self.model = get_peft_model(self.model, peft_config)
            else:
                print(f"Loading PEFT adapter from {load_from}")
                self.model = PeftModel.from_pretrained(self.model, load_from)
            self.model = self.model.to(device)

    
    def get_peft_config(self, config):
        if config.peft.method == 'lora':
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
        else:
            raise ValueError
        return peft_config
    
    def forward(self, input_ids, attention_mask, labels=None, mode='train', max_new_tokens=None):
        model_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'max_new_tokens': max_new_tokens,
            'labels': labels,
        }
        model_kwargs = {k:v for k,v in model_kwargs.items() if v is not None}

        print(input_ids.shape)

        if mode == 'train' or mode == 'evaluate':
            out_enc = self.model(**model_kwargs)
        elif mode == 'generate':
            out_enc = self.model.generate(**model_kwargs)
        else:
            raise ValueError()
        return out_enc

    def generate(self, *args, **kwargs):
        return self(*args, mode='generate', **kwargs)
    
    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )
    
    def save(self):
        self.model.save_pretrained("opts_model")
    
    def finetune_model(self, train_loader, val_loader):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.training.lr)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(train_loader) * self.cfg.training.num_epochs),
        )

        for epoch in range(self.cfg.training.num_epochs):
            self.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_loader)):

                batch = {k: batch[k].to(device).to(device) for k in DATA_KEYS}
                outputs = self(**batch, mode='train')
                loss = outputs.loss

                if loss is None:
                    print('loss', loss)
                    print('batch', batch.keys())

                total_loss += loss.detach().cpu().float()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            self.eval()
            eval_loss = 0
            eval_preds = []
            for step, batch in enumerate(tqdm(val_loader)):
                batch = {k: batch[k].to(device) for k in DATA_KEYS}
                with torch.no_grad():
                    outputs = self(**batch, mode='evaluate')
                loss = outputs.loss
                eval_loss += loss.detach().float()
                eval_preds.extend(
                    tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
                )

            eval_epoch_loss = eval_loss / len(val_loader)
            eval_ppl = torch.exp(eval_epoch_loss)
            train_epoch_loss = total_loss / len(train_loader)
            train_ppl = torch.exp(train_epoch_loss)
            print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

            self.save()
