import torch
from config import Config
from model import Transformer
from tokenizer import BPETokenizer

def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k, device):
    """
    Generate text by sampling tokens autoregressively.
    
    model: the trained Transformer
    tokenizer: trained BPETokenizer (loaded with merges/vocab)
    prompt: starting text (string)
    max_new_tokens: how many tokens to generate (in addition to the prompt)
    temperature: randomness knob (1.0 = default, lower = more focused)
    top_k: only consider top K candidates each step (e.g., 50)
    """
    
    model.eval()    # Make sure we're in eval mode
    
    # Encode the prompt → list of token IDs
    token_ids = tokenizer.encode(prompt)
    
    # Convert to tensor of shape (1, T) — batch size 1
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
    
    context_window = model.context_window
    
    with torch.no_grad():    # We're not training, don't track gradients
        for _ in range(max_new_tokens):
            # If sequence is longer than context window, crop to last context_window tokens
            # (model can only see this many at once)
            if tokens.size(1) > context_window:
                input_tokens = tokens[:, -context_window:]
            else:
                input_tokens = tokens
            
            # Forward pass
            logits, _ = model(input_tokens)
            # logits shape: (1, T, vocab_size)
            
            # We only care about the LAST position's prediction
            # (what comes after the entire sequence so far)
            logits = logits[:, -1, :]    # shape: (1, vocab_size)
            
            # Apply temperature scaling
            logits = logits / temperature
            
            # Apply top-k filtering: zero out all but top K
            if top_k is not None:
                top_values, _ = torch.topk(logits, top_k, dim=-1)
                # The smallest of the top K becomes our threshold
                threshold = top_values[:, -1].unsqueeze(-1)
                # Set everything below threshold to -infinity (so softmax gives them 0)
                logits = torch.where(logits < threshold, float("-inf"), logits)
            
            # Softmax → probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Sample one token from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            # shape: (1, 1)
            
            # Append to our sequence
            tokens = torch.cat([tokens, next_token], dim=1)
    
    # Decode back to text
    all_token_ids = tokens[0].tolist()    # remove batch dim
    text = tokenizer.decode(all_token_ids)
    return text

def main():
    config = Config()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(config.data_dir + "/tokenizer.json")
    
    # Build model architecture (same as training)
    model = Transformer(
        vocab_size=config.vocab_size,
        context_window=config.context_window,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
    )
    
    # Load trained weights
    # weights_only=False because checkpoint contains the Config dataclass, not just weights.
    # Safe because we created the file ourselves.
    checkpoint = torch.load(config.checkpoint_dir + "/final.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    
    print(f"Loaded model from step {checkpoint['step']}")
    print("Chat with MaxGPT-1. Type 'quit' to exit.\n")
    
    # Simple chat loop
    while True:
        user_input = input("USER: ")
        if user_input.strip().lower() in ["quit", "exit"]:
            break
        
        # Format prompt in the same format as training data
        prompt = f"USER: {user_input}\nASSISTANT:"
        
        # Generate
        full_text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=200,
            temperature=0.5,
            top_k=20,
            device=device,
        )
        
        # full_text contains both the prompt and the generated text
        # Extract just the new part (after the prompt)
        generated = full_text[len(prompt):]
        
        # Optionally stop at next "USER:" since our format puts user turns there
        if "USER:" in generated:
            generated = generated.split("USER:")[0]
        
        print(f"ASSISTANT:{generated}\n")

if __name__ == "__main__":
    main()