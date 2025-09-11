import os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]


def save_jsonl(data, path):
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


def a_proj_b(a, b):
    """Project vector a onto vector b"""
    return (a * b).sum(dim=-1) / b.norm(dim=-1)


def main(file_path, vector_path, layer=15, model_name="meta-llama/Meta-Llama-3-8B-Instruct", overwrite=False):
    """
    Calculate per-token projection scores for responses in a CSV file.
    
    Args:
        file_path: Path to CSV file with 'prompt' and 'answer' columns
        vector_path: Path to persona vector file (.pt)
        layer: Layer number to use for projection (default: 15)
        model_name: Model name for tokenizer and model loading
        overwrite: Whether to overwrite existing token projection column
    """
    
    # Load tokenizer and model
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Load persona vector
    print(f"Loading persona vector from: {vector_path}")
    vector_dict = torch.load(vector_path, weights_only=False)
    persona_vector = vector_dict[layer].to(model.device)  # Get vector for specified layer and move to model device
    
    # Load CSV data
    print(f"Loading data from: {file_path}")
    data = pd.read_csv(file_path)
    
    # Check if required columns exist
    if 'prompt' not in data.columns or 'answer' not in data.columns:
        raise ValueError("CSV must contain 'prompt' and 'answer' columns")
    
    # Create column name for per-token projections
    vector_name = os.path.basename(vector_path).split(".")[0]
    metric_model_name = os.path.basename(model_name)
    token_proj_col = f"{metric_model_name}_{vector_name}_token_proj_layer{layer}"
    
    # Check if column already exists
    if token_proj_col in data.columns and not overwrite:
        print(f"Column {token_proj_col} already exists. Use --overwrite to replace it.")
        return
    
    print(f"Calculating per-token projections for column: {token_proj_col}")
    
    # Calculate per-token projections
    token_projections = []
    
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Processing rows"):
        prompt = row['prompt']
        answer = row['answer']
        
        # Tokenize prompt and full sequence
        full_text = prompt + answer
        inputs = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        
        # Get prompt length in tokens
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_tokens)
        
        # Get model outputs with hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Extract hidden states for response tokens only
        response_hidden_states = outputs.hidden_states[layer][:, prompt_len:, :]  # [1, response_len, hidden_dim]
        response_hidden_states = response_hidden_states.squeeze(0)  # [response_len, hidden_dim]
        
        # Calculate projection for each response token
        token_projections_row = []
        for token_idx in range(response_hidden_states.shape[0]):
            token_hidden = response_hidden_states[token_idx]  # [hidden_dim]
            # Ensure both tensors are on the same device
            token_hidden_cpu = token_hidden.cpu()
            persona_vector_cpu = persona_vector.cpu()
            projection = a_proj_b(token_hidden_cpu.unsqueeze(0), persona_vector_cpu.unsqueeze(0)).item()
            token_projections_row.append(projection)
        
        token_projections.append(token_projections_row)
    
    # Add per-token projections to dataframe
    data[token_proj_col] = token_projections.copy()
    
    # Save updated CSV
    data.to_csv(file_path, index=False)
    print(f"Per-token projection results saved to {file_path}")
    print(f"Added column: {token_proj_col}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Calculate per-token projection scores")
    parser.add_argument("--file_path", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--vector_path", type=str, required=True, help="Path to persona vector file")
    parser.add_argument("--layer", type=int, default=15, help="Layer number for projection")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing column")
    
    args = parser.parse_args()
    main(args.file_path, args.vector_path, args.layer, args.model_name, args.overwrite)
