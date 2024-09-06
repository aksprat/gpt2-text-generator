# app.py
from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load the GPT-2 model and tokenizer
model_name = 'gpt2-small' # Using a smaller model variant
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

@app.route('/generate', methods=['POST'])
def generate():
    input_data = request.json
    prompt = input_data.get('prompt', '')

    # Tokenize the input prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt')

    # Generate text
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"generated_text": generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
