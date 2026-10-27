"""Train compact generative-family models and compare BANKING77 intent accuracy.

Run from the project root:
    uv run -m evaluation.generative_family_comparison
"""

import argparse
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

PAD = 0
BOS = 1
EOS = 2
UNK = 3
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SEED = 17
MAX_LENGTH = 32


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())[:MAX_LENGTH]


class BankingText(Dataset):
    def __init__(self, sequences: list[list[int]], labels: list[int]):
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return torch.tensor(self.sequences[index], dtype=torch.long), torch.tensor(self.labels[index], dtype=torch.long)


def pad_and_collate(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=PAD)
    return padded, lengths, torch.stack(labels)


def build_vocab(train_text):
    vocab = {"<pad>": PAD, "<bos>": BOS, "<eos>": EOS, "<unk>": UNK}
    for text in train_text:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def encode_texts(texts, vocab):
    output = []
    for text in texts:
        tokens = [BOS]
        for token in tokenize(text):
            tokens.append(vocab.get(token, UNK))
        tokens.append(EOS)
        output.append(tokens)
    return output


class CausalIntentModel(nn.Module):
    def __init__(self, vocab_size, class_count):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=PAD)
        self.gru = nn.GRU(64, 128, batch_first=True)
        self.lm_head = nn.Linear(128, vocab_size)
        self.intent_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, class_count),
        )

    def forward(self, tokens):
        hidden, _ = self.gru(self.embedding(tokens))
        mask = tokens.ne(PAD).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.lm_head(hidden), self.intent_head(pooled)


class VAEIntentModel(nn.Module):
    def __init__(self, input_dim, class_count, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, class_count),
        )

    def forward(self, x):
        params = self.encoder(x)
        mu = params[:, :32]
        logvar = params[:, 32:]
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(z)
        logits = self.classifier(mu)
        return recon, logits, mu, logvar


class GANIntentModel(nn.Module):
    def __init__(self, feature_dim, class_count):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(64 + class_count, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, feature_dim),
            nn.Sigmoid(),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
        )
        self.real_fake = nn.Linear(128, 1)
        self.intent_head = nn.Linear(128, class_count)

    def generate(self, labels):
        noise = torch.randn(labels.shape[0], 64, device=labels.device)
        onehot = F.one_hot(labels, num_classes=self.intent_head.out_features).float()
        return self.generator(torch.cat([noise, onehot], dim=1))

    def discriminate(self, features):
        hidden = self.discriminator(features)
        return self.real_fake(hidden).squeeze(1), self.intent_head(hidden)


def accuracy(preds, gold):
    return sum(int(p == g) for p, g in zip(preds, gold)) / len(gold)


def train_causal_model(train_seq, train_y, vocab_size, class_count, epochs, batch_size, device):
    dataset = BankingText(train_seq, train_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_and_collate)
    model = CausalIntentModel(vocab_size, class_count).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        running = 0.0
        for tokens, _, labels in loader:
            tokens, labels = tokens.to(device), labels.to(device)
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]
            opt.zero_grad()
            token_logits, intent_logits = model(inputs)
            language_loss = F.cross_entropy(
                token_logits.reshape(-1, vocab_size),
                targets.reshape(-1),
                ignore_index=PAD,
            )
            intent_loss = F.cross_entropy(intent_logits, labels)
            loss = language_loss + intent_loss
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"Causal LM epoch {epoch + 1}/{epochs}: avg joint loss {running / len(loader):.4f}")
    return model


def predict_causal(model, sequences, device, batch_size):
    model.eval()
    loader = DataLoader(
        BankingText(sequences, [0] * len(sequences)),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_and_collate,
    )
    preds = []
    with torch.no_grad():
        for tokens, _, _ in loader:
            tokens = tokens.to(device)
            _, intent_logits = model(tokens[:, :-1])
            preds.extend(intent_logits.argmax(dim=1).cpu().tolist())
    return preds


def train_vae_model(train_features, train_y, class_count, epochs, batch_size, device):
    dataset = TensorDataset(torch.tensor(train_features, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = VAEIntentModel(train_features.shape[1], class_count, latent_dim=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        running = 0.0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            opt.zero_grad()
            recon, logits, mu, logvar = model(features)
            recon_loss = F.mse_loss(recon, features)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / features.shape[0]
            intent_loss = F.cross_entropy(logits, labels)
            loss = recon_loss + 0.01 * kl + intent_loss
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"VAE epoch {epoch + 1}/{epochs}: avg loss {running / len(loader):.4f}")
    return model


def predict_vae(model, features, device, batch_size):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            x = torch.tensor(features[start:start + batch_size], dtype=torch.float32, device=device)
            _, logits, _, _ = model(x)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
    return preds


def train_gan_model(train_features, train_y, class_count, epochs, batch_size, device):
    dataset = TensorDataset(torch.tensor(train_features, dtype=torch.float32), torch.tensor(train_y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = GANIntentModel(train_features.shape[1], class_count).to(device)
    d_opt = torch.optim.Adam(model.discriminator.parameters(), lr=1e-4)
    d_opt.add_param_group({"params": model.real_fake.parameters()})
    d_opt.add_param_group({"params": model.intent_head.parameters()})
    g_opt = torch.optim.Adam(model.generator.parameters(), lr=1e-4)
    model.train()
    for epoch in range(epochs):
        d_run = 0.0
        g_run = 0.0
        for real, labels in loader:
            real, labels = real.to(device), labels.to(device)
            d_opt.zero_grad()
            real_logits, real_intent = model.discriminate(real)
            fake = model.generate(labels)
            fake_logits, fake_intent = model.discriminate(fake.detach())
            d_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            d_loss += F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            d_loss += F.cross_entropy(real_intent, labels)
            d_loss += 0.1 * F.cross_entropy(fake_intent, labels)
            d_loss.backward()
            d_opt.step()
            d_run += d_loss.item()

            g_opt.zero_grad()
            generated = model.generate(labels)
            fake_logits, fake_intent = model.discriminate(generated)
            g_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
            g_loss += F.cross_entropy(fake_intent, labels)
            g_loss.backward()
            g_opt.step()
            g_run += g_loss.item()
        print(f"GAN epoch {epoch + 1}/{epochs}: D {d_run / len(loader):.4f}, G {g_run / len(loader):.4f}")
    return model


def predict_gan(model, features, device, batch_size):
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.tensor(features[start:start + batch_size], dtype=torch.float32, device=device)
            _, logits = model.discriminate(batch)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
    return preds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=Path("evaluation/results/generative_family_comparison.csv"))
    args = parser.parse_args()

    set_seed(args.seed)
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    labels = sorted(train_df["category"].unique())
    label_map = {label: idx for idx, label in enumerate(labels)}
    train_y = [label_map[y] for y in train_df["category"]]
    test_y = [label_map[y] for y in test_df["category"]]

    vocab = build_vocab(train_df["text"])
    train_seq = encode_texts(train_df["text"], vocab)
    test_seq = encode_texts(test_df["text"], vocab)
    vocab_size = max(max(max(seq) for seq in train_seq + test_seq), max(vocab.values())) + 1
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2048, min_df=2)
    train_features = vectorizer.fit_transform(train_df["text"]).toarray().astype(np.float32)
    test_features = vectorizer.transform(test_df["text"]).toarray().astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset: {len(train_df)} train / {len(test_df)} test, {len(labels)} intents. Device={device}. Epochs={args.epochs}")

    causal_model = train_causal_model(train_seq, train_y, vocab_size, len(labels), args.epochs, args.batch_size, device)
    causal_train = predict_causal(causal_model, train_seq, device, args.batch_size)
    causal_test = predict_causal(causal_model, test_seq, device, args.batch_size)

    vae_model = train_vae_model(train_features, train_y, len(labels), args.epochs, args.batch_size, device)
    vae_train = predict_vae(vae_model, train_features, device, args.batch_size)
    vae_test = predict_vae(vae_model, test_features, device, args.batch_size)

    gan_model = train_gan_model(train_features, train_y, len(labels), args.epochs, args.batch_size, device)
    gan_train = predict_gan(gan_model, train_features, device, args.batch_size)
    gan_test = predict_gan(gan_model, test_features, device, args.batch_size)

    results = pd.DataFrame([
        {
            "model": "GRU causal language model with intent head",
            "generative_family": "Autoregressive / causal LM",
            "train_accuracy": accuracy(causal_train, train_y),
            "test_accuracy": accuracy(causal_test, test_y),
        },
        {
            "model": "TF-IDF VAE with intent head",
            "generative_family": "Variational autoencoder",
            "train_accuracy": accuracy(vae_train, train_y),
            "test_accuracy": accuracy(vae_test, test_y),
        },
        {
            "model": "Conditional TF-IDF auxiliary-classifier GAN",
            "generative_family": "Generative adversarial network",
            "train_accuracy": accuracy(gan_train, train_y),
            "test_accuracy": accuracy(gan_test, test_y),
        },
    ])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print("\nMeasured accuracy:")
    print(results.to_string(index=False, formatters={"train_accuracy": lambda x: f"{x:.4f}", "test_accuracy": lambda x: f"{x:.4f}"}))
    print(f"\nSaved results to {output}")


if __name__ == "__main__":
    main()
