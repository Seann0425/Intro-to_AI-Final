# 导入库
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login

# 使用访问令牌进行登录
login("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")

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# 编写你的输入文本
input_text = "say a story about brave prince"
inputs = tokenizer(input_text, return_tensors="pt")

# 使用模型生成文本
outputs = model.generate(inputs["input_ids"], max_length=100)

# 解码输出文本
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)