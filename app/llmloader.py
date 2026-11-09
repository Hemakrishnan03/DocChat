

import os

class LLMLoader:
    def __init__(self,model_name:str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        local_path = "./data/models/Qwen2.5-0.5B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if local_path and os.path.exists(local_path):
            print(f"Loading LLM model from local path: {local_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(local_path)
            self.model = AutoModelForCausalLM.from_pretrained(local_path)
        else:
            print(f"Downloading LLM model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.save_pretrained(local_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.save_pretrained(local_path)
    def llm(self):
        return self.model

    def generate(
     self,
     prompt: str,
     max_new_tokens: int = 200,
     temperature: float = 0.7,
     ) -> str:

     inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

     output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            **({"temperature": temperature} if temperature > 0 else {}),
        )

     prompt_length = inputs["input_ids"].shape[1]
     answer_ids = output_ids[0][prompt_length:]
     return self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

