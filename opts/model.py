from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
import torch
from tqdm import tqdm

from opts.utils import print_gpu_utilization

from accelerate import Accelerator


def load_tokenizer(config):
    return AutoTokenizer.from_pretrained(config.model.name)

DATA_KEYS = ['input_ids', 'attention_mask', 'labels']

class OPTSModel(torch.nn.Module):
    def __init__(self, config, prev_model=None, load_from=None):
        super().__init__()
        self.cfg = config

        # self.accelerator = Accelerator(gradient_accumulation_steps=8)
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        if prev_model:
            self.model = prev_model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(config.model.name)

            for param in self.model.parameters():
                param.requires_grad = False  # freeze the model

            #self.model.gradient_checkpointing_enable()

            if load_from is None:
                print(f"Creating new PEFT adapter")
                peft_config = self.get_peft_config(config)
                self.model = get_peft_model(self.model, peft_config)
            else:
                print(f"Loading PEFT adapter from {load_from}")
                self.model = PeftModel.from_pretrained(self.model, load_from)
            self.model = self.model.to(self.device)

    
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
    
    def forward(self, input_ids, attention_mask, labels=None, mode='train', max_new_tokens=None, prompt_ques_tokens=None):

        #print(input_ids.shape, attention_mask.shape)

        model_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'max_new_tokens': max_new_tokens,
            'labels': labels,
        }
        model_kwargs = {k:v for k,v in model_kwargs.items() if v is not None}

        #print(input_ids.shape)

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
    
    def save(self, iteration=None):
        if iteration is None:
            self.model.save_pretrained("results/opts_model")
        else:
            self.model.save_pretrained(f"results/opts_model_iter{iteration}")
        
    def evaluate_model(self, val_loader, tokenizer, epoch=None):
        self.eval()
        eval_loss = 0
        eval_preds = []
        # for step, batch in enumerate(tqdm(val_loader)):
        for step, batch in enumerate(val_loader):
            batch_data = {k: batch[k].to(self.device) for k in DATA_KEYS}
            with torch.no_grad():
                outputs = self(**batch_data, mode='evaluate')

            loss = outputs.loss
            eval_loss += loss.detach().float()
            batch_generated = tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            eval_preds.extend(batch_generated)
        
            if self.cfg.testing.log_step and step % self.cfg.testing.log_step == 0:
                print(f"[EVAL] [{epoch}] Step {step+1}/{len(val_loader)} loss {eval_loss/(step+1)}\"")
    
        # Evaluate generations
        for step, batch in enumerate(val_loader):
            if step >= self.cfg.testing.n_generate:
                break
            input_ids = torch.tensor(batch['prompt_ques_tokens'][0:1]).to(self.device)
            attention_mask = torch.tensor(batch['prompt_ques_attention_mask'][0:1]).to(self.device)
            max_new_tokens = min(self.cfg.max_tokens - len(input_ids[0]), self.cfg.generate_max_new_tokens)

            with torch.no_grad():
                outputs = self(input_ids=input_ids, attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens, mode='generate')

            generated = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]

            generated = generated[len(batch['prompt_ques'][0]):]
            print(f"\033[92mPrompt:\033[0m \"{batch['prompt_ques'][0]}\"")
            print(f"\033[92mWanted summary:\033[0m \"{batch['prompt_ans'][0]}\"")
            print(f"\033[92mGenerated summary:\033[0m \"{repr(generated)}\"")
            print()

        eval_epoch_loss = eval_loss / len(val_loader)
        eval_ppl = torch.exp(eval_epoch_loss)

        if self.cfg.train:
            train_epoch_loss = total_loss / len(train_loader)
            train_ppl = torch.exp(train_epoch_loss)
            print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
        else:
            print(f"{epoch=}: {eval_ppl=} {eval_epoch_loss=}")

    def finetune_model(self, train_loader, val_loader, tokenizer):
        print_gpu_utilization()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.training.lr)
        #optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.training.lr)

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(train_loader) * self.cfg.training.num_epochs),
        )

        self.model, optimizer, train_loader, val_loader, lr_scheduler = self.accelerator.prepare(
            self.model, optimizer, train_loader, val_loader, lr_scheduler
        )

        for epoch in range(self.cfg.training.num_epochs):

            print("Train:", self.cfg.train, "| Eval:", self.cfg.eval)

            if self.cfg.train:
                self.train()
                total_loss = 0
                # for step, batch in enumerate(tqdm(train_loader)):
                for step, batch in enumerate(train_loader):
                    # with self.accelerator.accumulate(self.model):

                    optimizer.zero_grad()
                    batch_data = {k: batch[k].to(self.device) for k in DATA_KEYS}

                    cumul_losses = []
                    for i_split in range(0, len(batch_data['input_ids']), self.cfg.training.batch_split_size):
                        split_data = {
                            k: batch_data[k][i_split : i_split + self.cfg.training.batch_split_size]
                            for k in DATA_KEYS
                        }
                        outputs = self(**split_data, mode='train')
                        self.accelerator.backward(outputs.loss)
                        cumul_losses.append(outputs.loss.detach().cpu().float())

                    batch_loss = sum(cumul_losses) / len(cumul_losses)
                    total_loss += batch_loss
                    optimizer.step()
                    lr_scheduler.step()

                    if self.cfg.training.log_step and step % self.cfg.training.log_step == 0:
                        print(f'[{epoch}] Step {step+1}/{len(train_loader)} loss {batch_loss:.6f} (avg {total_loss/(step+1):.6f})')

                    if step % (len(train_loader)//4) == 0 and step != 0:
                        # print(step, len(train_loader), len(train_loader)//4, epoch)
                        print("Saving model...")
                        self.save(epoch * len(train_loader) + step)

            if self.cfg.eval:
                self.evaluate_model(val_loader, tokenizer, epoch=epoch)

            self.save(f"epoch{epoch}")

