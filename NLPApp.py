#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DistilBertTokenizer
import torch
from safetensors.torch import load_file

app = Flask(__name__)

# Load GPT-2 tokenizer and model
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

# Define DistilBERT safety classifier model
class YourSafetyClassifier(torch.nn.Module):
    def __init__(self):
        super(YourSafetyClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        return logits

# Load safety classifier
safety_model = YourSafetyClassifier()
safety_model_weights = load_file('/Users/dhwanibhavankar/Downloads/model.safetensors')
safety_model.load_state_dict(safety_model_weights, strict=False)
safety_model.eval()

# API route for generating and classifying text
@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.json.get('prompt', '')

    # Tokenize input prompt
    input_ids = gpt2_tokenizer.encode(prompt, return_tensors='pt')

    # Generate output from GPT-2
    output = gpt2_model.generate(input_ids, max_length=100, pad_token_id=gpt2_tokenizer.eos_token_id, temperature=0.7, top_p=0.9, top_k=50)
    generated_text = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)

    # Tokenize generated text for safety classification
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    input_for_safety = tokenizer.encode(generated_text, return_tensors='pt')
    attention_mask_safety = input_for_safety.ne(tokenizer.pad_token_id).long()

    # Pass the generated text through the safety classifier
    with torch.no_grad():
        safety_pred = safety_model(input_for_safety, attention_mask_safety)

    predicted_class = "Safe" if torch.argmax(safety_pred, dim=1).item() == 1 else "Unsafe"

    return jsonify({'generated_text': generated_text, 'safety': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)

