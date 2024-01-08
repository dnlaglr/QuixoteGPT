import torch
import torch.nn as nn
from torch.nn import functional as Fn

batchSize = 32
blockSize = 8
maxIterations = 3000
evalInterval = 300
learningRate = 1e-2
device = "cude" if torch.cuda.is_available() else "cpu"
evalIterations = 300

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

class BigramLM(nn.Module):
  def __init__(self, vocabSize):
    super().__init__()

    # Have each token read off the logits for next token from a lookup table
    self.tokenEmbeddingTable = nn.Embedding(vocabSize, vocabSize)

  def forward(self, idx, targets=None):
    # Idx and targets are both (B, T) tensor of int
    logits = self.tokenEmbeddingTable(idx)

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
      logits, loss = self(idx)
      logits = logits[:, -1, :] # Focus on last time step and becomes (B, C)
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