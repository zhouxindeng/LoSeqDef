#登录Hugging Face
#huggingface-cli login

from transformers import AutoTokenizer
import transformers
import torch

#模型路径
model="/root/autodl-tmp/Llama-2-13b-chat-hf"

#tokenizer方法
tokenizer =AutoTokenizer.from_pretrained(model)

#构造pipeline
pipeline=transformers.pipeline("text-generation",model=model,torch_dtype=torch.float16,device_map="auto")

#进行预测，生成应答
sequences=pipeline('I liked "Breadkig Bad" and "Band of Brothers". Do you have recommendations of other shows I might like?\n',
                   do_sample=True,
                   top_k=10,
                   num_return_sequences=1,
                   eos_token_id=tokenizer.eos_token_id,
                   max_length=200)

for seq in sequences:
    print(f"Result: {seq['generated_text']}")