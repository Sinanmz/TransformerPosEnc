import torch 
import torch.nn as nn
import torch.nn.functional as F
import math



class Head_dec(nn.Module):
    def __init__(self, n_embed, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size)
        self.query = nn.Linear(n_embed, head_size)
        self.value = nn.Linear(n_embed, head_size)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape # B: Batch size, T: Block size (Timestep), C: n_embed (channels)      
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        W = q @ k.transpose(-1, -2) * (k.shape[-1] ** -.5)
        W = W.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        W = F.softmax(W, dim= -1)
        W = self.dropout(W)
        return  W @ v
    

class Head_enc(nn.Module):
    def __init__(self, n_embed, head_size, block_size, dropout, device):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size)
        self.query = nn.Linear(n_embed, head_size)
        self.value = nn.Linear(n_embed, head_size)
        self.dropout = nn.Dropout(dropout)
        self.device = device


    def forward(self, t):
        x = t[0]
        attention_mask = t[1]
        B, T, C = x.shape # B: Batch size, T: Block size (Timestep), C: n_embed (channels)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        W = q @ k.transpose(-1, -2) * (k.shape[-1] ** -.5)
        attention_mask = attention_mask.reshape(B, T, 1).expand(B, T, T).transpose(-2, -1).to(self.device)
        W = W.masked_fill(attention_mask == 0, float('-inf')).to(self.device) # (B, T, T)
        W = F.softmax(W, dim= -1)
        W = self.dropout(W)
        return  W @ v


class MultiHeadAttention_dec(nn.Module):
    def __init__(self, n_embed, head_count, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head_dec(n_embed, head_size, block_size, dropout) for _ in range(head_count)])
        self.proj = nn.Linear(head_count*head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim = -1)
        x = self.proj(x)
        x = self.dropout(x)
        return x
    

class MultiHeadAttention_enc(nn.Module):
    def __init__(self, n_embed, head_count, head_size, block_size, dropout, device):
        super().__init__()
        self.heads = nn.ModuleList([Head_enc(n_embed, head_size, block_size, dropout, device) for _ in range(head_count)])
        self.proj = nn.Linear(head_count*head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, t):
        x = t[0]
        attention_mask = t[1]

        x = torch.cat([head((x, attention_mask)) for head in self.heads], dim = -1)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed), 
            nn.ReLU(), 
            nn.Linear(4*n_embed, n_embed), 
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    

class Block_dec(nn.Module):
    def __init__(self, n_embed, head_count, block_size, dropout):
        super().__init__()
        head_size = n_embed // head_count
        self.sa = MultiHeadAttention_dec(n_embed, head_count, head_size, block_size, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

class Block_enc(nn.Module):
    def __init__(self, n_embed, head_count, block_size, dropout, device):
        super().__init__()
        head_size = n_embed // head_count
        self.sa = MultiHeadAttention_enc(n_embed, head_count, head_size, block_size, dropout, device)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, t):
        x = t[0]
        attention_mask = t[1]

        x = x + self.sa((self.ln1(x), attention_mask))
        x = x + self.ffwd(self.ln2(x))
        return (x, attention_mask)


class PositionalEncodingWave(nn.Module):

    def __init__(self, n_embed, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_embed, 2) * (-math.log(10000.0) / n_embed))
        pe = torch.zeros(max_len, 1, n_embed)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, mask):
        pos_encode = self.pe[:x.size(0)].masked_fill(mask == 0, 0)
        x = x + pos_encode
        return self.dropout(x)








class DecoderPosEnc(nn.Module):
    def __init__(self, n_vocab, n_embed, block_size, head_count, dropout, n_layers, device):
        super().__init__()
        self.token_embeddding_table = nn.Embedding(n_vocab, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block_dec(n_embed, head_count, block_size, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.linear_head = nn.Linear(n_embed, 2)
        self.device = device
        self.n_embed = n_embed
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, attention_mask):
        # X: token indeces of shape(batch_siza, block_size)
        # attention_mask: indicates the meaningful tokens (for ignoring PAD tokens, though in the decoder
        #   it's not as important, because every token pays attention only to the tokens before it and our
        #   CLS token is the last non-PAD token, so we just use it to find the index of the CLS token)

        B, T = x.shape
        C = self.n_embed
        pos_mask = torch.cat((attention_mask[:, 1:], torch.zeros((B, 1), device= self.device)), dim= -1).reshape(B, T, 1).expand(B, T, C)
        logits_mask = pos_mask[:, :, :2] != attention_mask.reshape(B, T, 1).expand(B, T, 2)
        tok_embed = self.token_embeddding_table(x)
        pos_embed = self.position_embedding_table(torch.arange(T, device=self.device).reshape(1, T).expand(B, T))
        pos_embed = pos_embed.masked_fill(pos_mask == 0, 0).to(self.device)
        x = tok_embed + pos_embed
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.linear_head(x)
        logits = torch.masked_select(logits, logits_mask).reshape(-1, 2)
        return logits
    

class DecoderPosEncWave(nn.Module):
    def __init__(self, n_vocab, n_embed, block_size, head_count, dropout, n_layers, device):
        super().__init__()
        self.token_embeddding_table = nn.Embedding(n_vocab, n_embed)
        self.position_encoder = PositionalEncodingWave(n_embed, dropout, 1000)
        self.blocks = nn.Sequential(*[Block_dec(n_embed, head_count, block_size, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.linear_head = nn.Linear(n_embed, 2)
        self.device = device
        self.n_embed = n_embed
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, attention_mask):
        # X: token indeces of shape(batch_siza, block_size)
        # attention_mask: indicates the meaningful tokens (for ignoring PAD tokens, though in the decoder
        #   it's not as important, because every token pays attention only to the tokens before it and our
        #   CLS token is the last non-PAD token, so we just use it to find the index of the CLS token)

        B, T = x.shape
        C = self.n_embed

        pos_mask = torch.cat((attention_mask[:, 1:], torch.zeros((B, 1), device= self.device)), dim= -1).reshape(B, T, 1).expand(B, T, C)
        logits_mask = pos_mask[:, :, :2] != attention_mask.reshape(B, T, 1).expand(B, T, 2)
        
        tok_embed = self.token_embeddding_table(x) * math.sqrt(self.n_embed)

        x = self.position_encoder(tok_embed, pos_mask)

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.linear_head(x)
        logits = torch.masked_select(logits, logits_mask).reshape(-1, 2)
        return logits



class Decoder(nn.Module):
    def __init__(self, n_vocab, n_embed, block_size, head_count, dropout, n_layers, device):
        super().__init__()
        self.token_embeddding_table = nn.Embedding(n_vocab, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block_dec(n_embed, head_count, block_size, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.linear_head = nn.Linear(n_embed, 2)
        self.device = device
        self.n_embed = n_embed
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, attention_mask):
        # X: token indeces of shape(batch_siza, block_size)
        # attention_mask: indicates the meaningful tokens (for ignoring PAD tokens, though in the decoder
        #   it's not as important, because every token pays attention only to the tokens before it and our
        #   CLS token is the last non-PAD token, so we just use it to find the index of the CLS token)

        B, T = x.shape
        C = self.n_embed
        pos_mask = torch.cat((attention_mask[:, 1:], torch.zeros((B, 1), device= self.device)), dim= -1).reshape(B, T, 1).expand(B, T, C)
        logits_mask = pos_mask[:, :, :2] != attention_mask.reshape(B, T, 1).expand(B, T, 2)
        tok_embed = self.token_embeddding_table(x)

        x = tok_embed
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.linear_head(x)
        logits = torch.masked_select(logits, logits_mask).reshape(-1, 2)
        return logits
    







class EncoderPosEnc(nn.Module):
    def __init__(self, n_vocab, n_embed, block_size, head_count, dropout, n_layers, device):
        super().__init__()
        self.token_embeddding_table = nn.Embedding(n_vocab, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block_enc(n_embed, head_count, block_size, dropout, device) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.linear_head = nn.Linear(n_embed, 2)
        self.device = device
        self.n_embed = n_embed
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, attention_mask):
        # X: token indeces of shape(batch_siza, block_size)
        # attention_mask: indicates the meaningful tokens (for ignoring PAD tokens)

        B, T = x.shape
        C = self.n_embed
        pos_mask = torch.cat((attention_mask[:, 1:], torch.zeros((B, 1), device= self.device)), dim= -1).reshape(B, T, 1).expand(B, T, C)
        logits_mask = pos_mask[:, :, :2] != attention_mask.reshape(B, T, 1).expand(B, T, 2)
        tok_embed = self.token_embeddding_table(x)
        pos_embed = self.position_embedding_table(torch.arange(T, device=self.device).reshape(1, T).expand(B, T))
        pos_embed = pos_embed.masked_fill(pos_mask == 0, 0)

        x = tok_embed + pos_embed
        x = self.blocks((x, attention_mask))[0]
        x = self.ln_f(x)
        logits = self.linear_head(x)
        logits = torch.masked_select(logits, logits_mask).reshape(-1, 2)
        return logits
    


class EncoderPosEncWave(nn.Module):
    def __init__(self, n_vocab, n_embed, block_size, head_count, dropout, n_layers, device):
        super().__init__()
        self.token_embeddding_table = nn.Embedding(n_vocab, n_embed)
        self.position_encoder = PositionalEncodingWave(n_embed, dropout, 1000)
        self.blocks = nn.Sequential(*[Block_enc(n_embed, head_count, block_size, dropout, device) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.linear_head = nn.Linear(n_embed, 2)
        self.device = device
        self.n_embed = n_embed
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, attention_mask):
        # X: token indeces of shape(batch_siza, block_size)
        # attention_mask: indicates the meaningful tokens (for ignoring PAD tokens)

        B, T = x.shape
        C = self.n_embed
        pos_mask = torch.cat((attention_mask[:, 1:], torch.zeros((B, 1), device= self.device)), dim= -1).reshape(B, T, 1).expand(B, T, C)
        logits_mask = pos_mask[:, :, :2] != attention_mask.reshape(B, T, 1).expand(B, T, 2)
        
        tok_embed = self.token_embeddding_table(x) * math.sqrt(self.n_embed)

        x = self.position_encoder(tok_embed, pos_mask)

        x = self.blocks((x, attention_mask))[0]
        x = self.ln_f(x)
        logits = self.linear_head(x)
        logits = torch.masked_select(logits, logits_mask).reshape(-1, 2)
        return logits


class Encoder(nn.Module):
    def __init__(self, n_vocab, n_embed, block_size, head_count, dropout, n_layers, device):
        super().__init__()
        self.token_embeddding_table = nn.Embedding(n_vocab, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block_enc(n_embed, head_count, block_size, dropout, device) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.linear_head = nn.Linear(n_embed, 2)
        self.device = device
        self.n_embed = n_embed
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, attention_mask):
        # X: token indeces of shape(batch_siza, block_size)
        # attention_mask: indicates the meaningful tokens (for ignoring PAD tokens)

        B, T = x.shape
        C = self.n_embed
        pos_mask = torch.cat((attention_mask[:, 1:], torch.zeros((B, 1), device= self.device)), dim= -1).reshape(B, T, 1).expand(B, T, C)
        logits_mask = pos_mask[:, :, :2] != attention_mask.reshape(B, T, 1).expand(B, T, 2)
        tok_embed = self.token_embeddding_table(x)

        x = tok_embed
        x = self.blocks((x, attention_mask))[0]
        x = self.ln_f(x)
        logits = self.linear_head(x)
        logits = torch.masked_select(logits, logits_mask).reshape(-1, 2)
        return logits
    


