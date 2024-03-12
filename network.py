import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from utils import accuracy

class Img2Seq(nn.Module):
    def __init__(self, img_size, patch_size, n_channels, d_model):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.n_channels = n_channels

        self.nh, self.nw = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        n_tokens = self.nh * self.nw

        token_dim = patch_size[0] * patch_size[1] * n_channels

        self.linear = nn.Linear(token_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.randn(n_tokens, d_model))


    def __call__(self, batch):
        batch = self._patchify(batch)

        b, c, nh, nw, ph, pw = batch.shape

        # Flattening the patches
        batch = torch.permute(batch, [0, 2, 3, 4, 5, 1])
        batch = torch.reshape(batch, [b, nh * nw, ph * pw * c])

        batch = self.linear(batch) # b, nh*nw, d_model
        cls = self.cls_token.expand([b, -1, -1])
        emb = batch + self.pos_emb

        return torch.cat([cls, emb], axis=1)


    def _patchify(self, batch):
        """
        Patchify the batch of images

        Shape:
            batch: (b, c, h, w)
            output: (b, c, nh, nw, ph, pw)
        """
        b, _, _, _ = batch.shape

        batch_patches = torch.reshape(batch, (b, self.n_channels, self.nh, self.patch_size[0], self.nw, self.patch_size[1]))
        batch_patches = torch.permute(batch_patches, (0, 1, 2, 4, 3, 5))

        return batch_patches

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1,2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        output = self.W_o(self.combine_heads(attn_output))

        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}],  train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


class ViT(ImageClassificationBase):
    def __init__(self, img_size=(28,28), patch_size=(4,4), n_channels=1, d_model=1024, nheads=4, d_ff=2048, blocks=8, mlp_head_units=[1024, 512], nclasses=10, dropout=0.0):
        super(ViT, self).__init__()

        self.img2seq = Img2Seq(img_size, patch_size, n_channels, d_model)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, nheads, d_ff, dropout) for _ in range(blocks)])

        self.mlp = self._get_mlp(d_model, mlp_head_units, nclasses)
        self.dropout = nn.Dropout(dropout)

        self.output = nn.Sigmoid() if nclasses == 1 else nn.Softmax(dim=1)


    def _get_mlp(self, in_features, hidden_units, out_features):
        """
        Returns a MLP head
        """
        dims = [in_features] + hidden_units + [out_features]
        layers = []

        for dim1, dim2 in zip(dims[:-2], dims[1:-1]):
            layers.append(nn.Linear(dim1, dim2))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))

        return nn.Sequential(*layers)

    def __call__(self, src):

        imgseq = self.img2seq(src)

        enc_output = imgseq
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, mask=None)

        enc_output = enc_output[:, 0, :] # ?????
        output = self.output(self.dropout(self.mlp(enc_output)))

        return output