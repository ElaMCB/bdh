#!/usr/bin/env python
# Copyright Pathway Technology, Inc.

import os
import argparse
import torch
from bdh import BDH, BDHConfig

def main():
    parser = argparse.ArgumentParser(description='Run inference with BDH model')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default='Hello, world!', help='Text prompt to start generation')
    parser.add_argument('--max_tokens', type=int, default=100, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=40, help='Top-k sampling parameter')
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model with default config
    config = BDHConfig()
    model = BDH(config)
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading model from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        print("No checkpoint provided or file not found. Using untrained model.")
    
    model.to(device)
    model.eval()
    
    # Encode prompt
    # For simplicity, we're using ASCII encoding here
    # In a real application, you would use a proper tokenizer
    encoded_prompt = torch.tensor([[ord(c) % 256 for c in args.prompt]], dtype=torch.long).to(device)
    
    print(f"Prompt: {args.prompt}")
    print("Generating...")
    
    # Generate text
    with torch.no_grad():
        output = model.generate(
            encoded_prompt, 
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
    
    # Decode output
    # Again, in a real application, you would use a proper detokenizer
    generated_text = ''.join([chr(int(i)) for i in output[0].tolist()])
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()