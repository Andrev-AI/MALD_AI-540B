import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
vocab_size = 65024  # Vocabulary size
max_len = 200000  # Max context token size (200k tokens)
d_model = 44544  # Model dimension
num_heads = 192  # Number of attention heads
num_layers = 80  # Número of layers
dropout = 0.1  # dropout rate
mem_size = 200000  # Memory size (in tokens)

# Embeddings Layers
embedding = nn.Embedding(vocab_size, d_model)

# Transformers Model
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder_layers
        self.decoder = decoder_layers
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, target_ids=None, mem=None, top_k=0, top_p=1.0, freq_penalty=1.0, max_tokens=256):
        input_embeds = self.embedding(input_ids)
        if target_ids is not None:
            target_embeds = self.embedding(target_ids)

        encoder_output = input_embeds
        for layer in self.encoder:
            encoder_output = layer(encoder_output)

        if target_ids is None:
            mem = torch.zeros(1, 0, d_model, device=input_ids.device)
            output = []
            prev_output = torch.LongTensor([[0]]).to(input_ids.device)
            for _ in range(max_tokens):
                prev_embeds = self.embedding(prev_output).squeeze(1)
                decoder_output, mem, _ = self.decoder[0](prev_embeds, encoder_output, mem)
                for layer in self.decoder[1:]:
                    decoder_output, mem, _ = layer(decoder_output, encoder_output, mem)
                logits = self.output(decoder_output[:, -1])
                filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p, filter_value=-float('Inf'))
                next_token = sample_from_filtered_logits(filtered_logits, freq_penalty)
                output.append(next_token.item())
                prev_output = next_token.unsqueeze(0)
            return output
        else:
            decoder_output = target_embeds
            for layer in self.decoder:
                decoder_output, mem, _ = layer(decoder_output, encoder_output, mem)

            output = self.output(decoder_output)
            return output

def summarize(input_ids, top_k=50, top_p=0.95, freq_penalty=1.2, max_tokens=256):
    model = Transformer()
    model.eval()
    input_embeds = model.embedding(input_ids)

    encoder_output = input_embeds
    for layer in model.encoder:
        encoder_output = layer(encoder_output)

    mem = torch.zeros(1, 0, d_model, device=input_ids.device)
    output = []
    prev_output = torch.LongTensor([[0]]).to(input_ids.device)

    for _ in range(max_tokens):
        prev_embeds = model.embedding(prev_output).squeeze(1)
        decoder_output, mem, _ = model.decoder[0](prev_embeds, encoder_output, mem)
        for layer in model.decoder[1:]:
            decoder_output, mem, _ = layer(decoder_output, encoder_output, mem)
        logits = model.output(decoder_output[:, -1])
        filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p, filter_value=-float('Inf'))
        next_token = sample_from_filtered_logits(filtered_logits, freq_penalty)
        output.append(next_token.item())
        prev_output = next_token.unsqueeze(0)
        mem = torch.zeros(1, 0, d_model, device=input_ids.device)  # Limpar a memória após cada token

    return output

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        topk_logits, topk_indices = torch.topk(logits, top_k)
        weight_logits = topk_logits.clone()
        weight_logits[weight_logits < topk_logits[..., -1, None]] = filter_value
        logits = weight_logits
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits

def sample_from_filtered_logits(logits, freq_penalty=1.0):
    logits = logits / freq_penalty
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token
