from transformers import AutoTokenizer

model_name = "meta-llama/Llama-3.1-8B-Instruct" #"google/gemma-2-9b-it"  
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Check if the tokenizer has a chat template
if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
    print("Chat Template Found:")
    print(tokenizer.chat_template)
else:
    print("No Chat Template Available for this model.")
