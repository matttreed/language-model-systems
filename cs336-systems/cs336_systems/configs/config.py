import json
import os

class Systems_Config():
    def __init__(self, version):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, f'transformer_config_{version}.json')

        with open(json_path, 'r') as file:
            json_data = json.load(file)

        self.context_length = json_data['context_length']
        self.num_layers = json_data['num_layers']
        self.d_model = json_data['d_model']
        self.num_heads = json_data['num_heads']
        self.d_ff = json_data['d_ff']
        self.attn_pdrop = json_data['attn_pdrop']
        self.residual_pdrop = json_data['residual_pdrop']
        self.vocab_size = json_data['vocab_size']
        self.batch_size = json_data['batch_size']
        self.random_seed = json_data["random_seed"]
