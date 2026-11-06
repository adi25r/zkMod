import sys

model_card = sys.argv[1]
access_token = sys.argv[2]

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(model_card, token = access_token, cache_dir = "./model-storage")
model = AutoModelForCausalLM.from_pretrained(model_card, token = access_token, cache_dir = "./model-storage")