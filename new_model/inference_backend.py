import json
import torch
from flask_cors import CORS
from jsonformer import Jsonformer
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

app = Flask(__name__)
CORS(app)

##

model_path = "./finetuned-model"
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    trust_remote_code=True,
    device_map="auto",
)

def model_response(user_prompt):
    prompt = f'''<|im_start|>system
You are an AI Bank assistant who generates JSON based on user requests.

<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
'''
    
    json_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "type": {"type": "string"},
        }
    }

    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
    generated_data = jsonformer() 
    
    if generated_data["action"] == "account_balance":
        return generated_data
    elif generated_data["action"] == "account_statement" and generated_data["type"] == "default":
        return generated_data
    elif generated_data["action"] == "account_statement" and generated_data["type"] == "custom":
        json_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "type": {"type": "string"},
            "start_date": {"type": "string"},
            "end_date": {"type": "string"},
        }
    }
        jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
        generated_data = jsonformer()
        return generated_data

    elif generated_data["action"] == "disable_debit_card":
        json_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string"},
            "type": {"type": "string"},
            "last_digits": {'type': 'integer'},
            "reason": {"type": "string"},
        }
    }
        jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
        generated_data = jsonformer()
        return generated_data
      
    return generated_data

##

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    prompt = data['prompt']

    model_generated_text = model_response(prompt)
    print(f"model_generated_text: {model_generated_text}")
    response_format = {"generated_text": model_generated_text}
    return jsonify(response_format)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
