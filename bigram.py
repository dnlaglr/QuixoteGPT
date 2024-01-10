import torch
import torch.nn as nn
from torch.nn import functional as Fn

batchSize = 32
blockSize = 8
maxIterations = 5000
evalInterval = 300
learningRate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
evalIterations = 300
numEmbed = 32

# Get Don Quixote text from local file
with open("datasets/quixote.txt", 'r', encoding="utf-8") as f:
  text = f.read()

chars = sorted(list(set(text))) # All characters which appear in the text, sorted
vocabSize = len(chars)

# Create mapping from char to int, vice-versa
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data)) # 90% of text for training data, rest for validation
trainData = data[:n]
valData = data[n:]

def getBatch(split):
  data = trainData if split == "train" else valData
  ix = torch.randint(len(data) - blockSize, (batchSize,))
  x = torch.stack([data[i:i + blockSize] for i in ix])
  y = torch.stack([data[i + 1:i + blockSize + 1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

@torch.no_grad() # Disable gradient calculation
def estimateLoss():
  out = {}
  model.eval()
  for split in ["train", "val"]:
    losses = torch.zeros(evalIterations)
    for k in range(evalIterations):
      X, Y = getBatch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

class Head(nn.Module):
  # One head of self-attention
  def __init__(self, headSize):
    super().__init__()
    
    self.key = nn.Linear(numEmbed, headSize, bias=False)
    self.query = nn.Linear(numEmbed, headSize, bias=False)
    self.value = nn.Linear(numEmbed, headSize, bias=False)
    self.register_buffer("tril", torch.tril(torch.ones(blockSize, blockSize)))

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x) # (B, T, C)
    q = self.query(x) # (B, T, C)

    # Compute attention scores: "Affinities"
    weights = q @ k.transpose(-2, -1) * C ** -0.5 # (B, T, C) @ (B, C, T) = (B, T, T)
    weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    weights = Fn.softmax(weights, dim=-1) # (B, T, T)

    # Perform weighted aggregation of the values
    v = self.value(x)
    out = weights @ v # (B, T, T) @ (B, T, C) = (B, T, C)
    return out

class MultiHead(nn.Module):
  # Multiple heads of self-attention in parallel
  def __init__(self, numHeads, headSize):
    super().__init__()
    self.heads = nn.ModuleList([Head(headSize) for _ in range(numHeads)])
    self.proj = nn.Linear(numEmbed, numEmbed)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.proj(out)
    return out

class FeedForward(nn.Module):
  # Linear layer followed by a non-linearity
  def __init__(self, numEmbed):
    super().__init__()

    self.net = nn.Sequential(
      nn.Linear(numEmbed, 4 * numEmbed), 
      nn.ReLU(),
      nn.Linear(4 * numEmbed, numEmbed)
    )

  def forward(self, x):
    return self.net(x)
  
class Block(nn.Module):
  # Transformer block
  def __init__(self, numEmbed, numHeads):
    super().__init__()
    
    headSize = numEmbed // numHeads
    self.sa = MultiHead(numHeads, headSize)
    self.feedFwd = FeedForward(numEmbed)
    self.layerNorm1 = nn.LayerNorm(numEmbed)
    self.layerNorm2 = nn.LayerNorm(numEmbed)

  def forward(self, x):
    x = x + self.sa(self.layerNorm1(x))
    x = x + self.feedFwd(self.layerNorm2(x))
    return x

class BigramLM(nn.Module):
  def __init__(self, vocabSize):
    super().__init__()

    # Have each token read off the logits for next token from a lookup table
    self.tokenEmbeddingTable = nn.Embedding(vocabSize, numEmbed)
    self.posEmbeddingTable = nn.Embedding(blockSize, numEmbed)
    self.blocks = nn.Sequential(
      Block(numEmbed, numHeads=4),
      Block(numEmbed, numHeads=4),
      Block(numEmbed, numHeads=4),
      nn.LayerNorm(numEmbed)
    )
    self.lmHead = nn.Linear(numEmbed, vocabSize)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    # Idx and targets are both (B, T) tensor of int
    tokenEmbeds = self.tokenEmbeddingTable(idx) # (B, T, C)
    posEmbeds = self.posEmbeddingTable(torch.arange(T, device=device)) # (T, C)

    # Token embeddings + position embeddings = Token identities and where they occur
    x = tokenEmbeds + posEmbeds # (B, T, C)
    x = self.blocks(x)
    logits = self.lmHead(x) # (B, T, vocabSize)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B * T, C)
      targets = targets.view(B * T)
      loss = Fn.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, maxNewTokens):
    for _ in range(maxNewTokens):
      # Crop idx to last blockSize tokens
      idxCond = idx[:, -blockSize:]

      logits, loss = self(idxCond)
      logits = logits[:, -1, :] # Focus on last time step, becomes (B, C)
      probs = Fn.softmax(logits, dim=-1)

      nextIdx = torch.multinomial(probs, num_samples=1) # (B, 1) because one target for each batch dimension
      idx = torch.cat((idx, nextIdx), dim=1) # Create (B, T + 1) by concatinating sample idx to running sequence
    return idx

model = BigramLM(vocabSize)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learningRate)

for epoch in range(maxIterations):
  if epoch % evalInterval == 0:
    losses = estimateLoss()
    print(f"Step {epoch}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")

  xb, yb = getBatch('train')

  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

context = torch.zeros((1 , 1), dtype=torch.long, device=device)
print(decode(m.generate(context, maxNewTokens=500)[0].tolist()))