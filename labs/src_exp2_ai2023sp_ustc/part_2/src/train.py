#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import gc
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline

train_loss = []
val_loss = []
item=[]

class char_tokenizer:
    """
    a very simple char-based tokenizer. the tokenizer turns a string into a list of integers.
    """

    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        # TODO: calculate the vocab size and create a dictionary that maps each character to a unique integer
        self.n_vocab = len(corpus)
        self.char_to_int = {char: i for i, char in enumerate(corpus)}

    def encode(self, string: str):
        # TODO: convert a string into a list of integers and return, using the dictionary created above
        return [self.char_to_int[char] for char in string]

    def decode(self, codes: List[int]):
        # TODO: convert a list of integers into a string and return, using the dictionary created above
        return ''.join([self.corpus[code] for code in codes])


class Head(nn.Module):
    """single head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        # TODO: create three linear layers, Key, Query, and Value, each of which maps from n_embd to head_size
        #       and assign them to self.Key, self.Query, and self.Value, respectively
        self.Key = nn.Linear(head_size, head_size)
        self.Query = nn.Linear(head_size, head_size)
        self.Value = nn.Linear(head_size, head_size)
        self.head_size = head_size
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, inputs):
        # Implement the forward function of the head
        # The input is a tensor of shape (batch, time, n_embd)
        # The output should be a tensor of shape (batch, time, head_size)
        keys = self.Key(inputs)
        queries = self.Query(inputs)
        values = self.Value(inputs)
        batch, time, channel = inputs.shape

        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(inputs.shape[-1], dtype=torch.float))
        attn_probs = F.softmax(attn_scores.masked_fill(self.tril[:attn_scores.size(1), :attn_scores.size(2)] == 0, float('-inf')), dim=-1)
        out = torch.matmul(attn_probs, values)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        # Implement heads and projection
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.projection = nn.Linear(n_heads * head_size, head_size)

    def forward(self, inputs):
        # Implement the forward function of the multi-head attention
        attn_outputs = [head(inputs) for head in self.heads]
        out = torch.cat(attn_outputs, dim=-1)
        return self.projection(out)


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # Implement the feed-forward network
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, inputs):
        return self.net(inputs)


class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        # Implement the block of transformer using the MultiHeadAttention and FeedForward modules,
        # along with the layer normalization layers
        self.attention = MultiHeadAttention(n_heads, n_embd)
        self.layer_norm1 = nn.LayerNorm(n_embd)
        self.feed_forward = FeedForward(n_embd)
        self.layer_norm2 = nn.LayerNorm(n_embd)

    def forward(self, inputs):
        # Implement the forward function of the block
        attn_output = self.attention(inputs) + inputs
        out1 = self.layer_norm1(attn_output)
        ffn_output = self.feed_forward(out1) + out1
        out2 = self.layer_norm2(ffn_output)
        return out2


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: create the embedding table, the stack of blocks, the layer normalization layer,
        # and the linear layers.
        self.embedding = nn.Embedding(tokenizer.n_vocab, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_heads) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(n_embd)
        self.linear = nn.Linear(n_embd, tokenizer.n_vocab)

    def forward(self, inputs, labels=None):
        # TODO: implement the forward function of the transformer
        # inputs:(batch, context)
        batch, context = inputs.shape
        embedded = self.embedding(inputs)
        out = embedded

        for block in self.blocks:
            out = block(out)
            out = self.layer_norm(out)

        # out = self.layer_norm(out)
        logits = self.linear(out)

        # Compute the loss
        if labels is None:
            loss = None
        else:
            batch, time, channel = logits.shape
            logits = logits.view(batch * time, channel)
            labels = labels.view(batch * time)
            loss = F.cross_entropy(logits, labels)
        return logits, loss

    def generate(self, inputs, max_new_tokens):
        # TODO: generate new tokens from the transformer, using the inputs as the context,
        # and return the generated tokens with length of max_new_tokens
        for _ in range(max_new_tokens):
            logits, _ = self.forward(inputs[:, -block_size:])
            predicted_tokens = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(1)
            inputs = torch.cat((inputs, predicted_tokens), dim=1)

        return inputs


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out


def generate(model, input_text):
    context = torch.tensor(tokenizer.encode(input_text), device=device, dtype=torch.long).unsqueeze(0)
    generated_tokens = model.generate(context, max_new_tokens=500)[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)
    print(generated_text)



def train(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):

        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            train_loss.append(losses['train'])
            val_loss.append(losses['val'])
            item.append(iter)

        inputs, labels = get_batch("train")

        logits, loss = model(inputs, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


# define the hyperparameters
batch_size = 16
block_size = 256
max_iters = 2000  # set the number of training iterations as you like
eval_interval = 50
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 64
n_heads = 8
n_layers = 6


# read the dataset
with open("../data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()
chars = sorted(list(set(text)))

# initialize the vocabulary
tokenizer = char_tokenizer(chars)
encode = tokenizer.encode
decode = tokenizer.decode
n_vocab = tokenizer.n_vocab

# separate the dataset into train and validation
train_data = torch.tensor(encode(text[: -len(text) // 10]), dtype=torch.long)
val_data = torch.tensor(encode(text[-len(text) // 10 :]), dtype=torch.long)

# define the model
model = Transformer().to(device)
train(model)
input_text = "To be or not to be: that is the"
torch.cuda.empty_cache()
gc.collect()
generate(model, input_text)


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

plt.xlim(0,2000)
plt.ylim(0,5)

plt.xlabel('iter',fontsize=15)
plt.ylabel('loss',fontsize=15)

my_x_ticks=np.arange(0,2000,200)
plt.xticks(my_x_ticks,fontsize=5)
plt.yticks([1,2,3,4,5])

x=item
y1=train_loss
y2=val_loss


x_smooth = np.linspace(0, 2000, 300)  # np.linspace 等差数列,从x.min()到x.max()生成300个数，便于后续插值
y_smooth = make_interp_spline(x, y1)(x_smooth)
y2_smooth=  make_interp_spline(x, y2)(x_smooth)
plt.plot(x_smooth, y_smooth,label='train_loss')
plt.plot(x_smooth, y2_smooth,label='val_loss')

plt.legend(loc='best')
plt.show()