"""Req 3 (CS1): compare three generative-model families on the same BANKING77 text.

The three families are trained from scratch, at toy scale, on the 10,003 train messages
(NumPy only, CPU, about a minute):

  1. Autoregressive LM  p(x_t | x_<t)         word-level trigram model with backoff
  2. VAE                p(x | z), z ~ N(0, I)  bag-of-words encoder/decoder, manual backprop
  3. GAN                G(z) vs discriminator  bag-of-words, straight-through for discrete words

This is NOT a replacement for Qwen. It is evidence for the report: it shows, on our own
data, what each family can and cannot do for a text-in / text-out banking assistant.
Qwen2.5-1.5B-Instruct is the full-scale, neural member of the autoregressive family.

Run from the project root:  python -m src.generative_families
Writes evaluation/results/results_generative_families.csv and generative_families_samples.txt
"""

import re
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

SEED = 0
MAX_VOCAB = 800          # shared vocabulary for VAE and GAN (most frequent words)
N_SAMPLES = 200          # generated messages per family for the metrics
PREFIXES = ["i lost my", "how do i", "my card", "why was my", "can i top"]
STOP = {"the", "and", "for", "not", "how", "why", "what", "get", "can", "was", "are", "you", "this", "that", "with"}

WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    return WORD.findall(text.lower())


# ----------------------------------------------------------------------------- data
def load_data():
    train = pd.read_csv("data/train.csv")
    docs = [tokenize(t) for t in train["text"]]
    docs = [d for d in docs if d]
    counts = Counter(w for d in docs for w in d)
    vocab = [w for w, _ in counts.most_common(MAX_VOCAB)]
    index = {w: i for i, w in enumerate(vocab)}
    domain = {w for i in train["category"].unique() for w in i.lower().split("_") if len(w) >= 4 and w not in STOP}
    real_bigrams = {(a, b) for d in docs for a, b in zip(d, d[1:])}
    return train, docs, vocab, index, domain, real_bigrams


def bow_matrix(docs, index):
    X = np.zeros((len(docs), len(index)), dtype=np.float64)
    for r, d in enumerate(docs):
        for w in d:
            if w in index:
                X[r, index[w]] += 1
    return X


# ----------------------------------------------------------------------------- family 1: autoregressive
class TrigramLM:
    """p(w_t | w_{t-2}, w_{t-1}) from counts, backing off to bigram then unigram.
    Generates one word at a time and feeds it back: the autoregressive loop of an LLM,
    without the neural network."""

    def fit(self, docs):
        self.tri, self.bi, self.uni = defaultdict(Counter), defaultdict(Counter), Counter()
        for d in docs:
            seq = ["<s>", "<s>"] + d + ["</s>"]
            for a, b, c in zip(seq, seq[1:], seq[2:]):
                self.tri[(a, b)][c] += 1
                self.bi[b][c] += 1
                self.uni[c] += 1
        return self

    def _next(self, ctx, rng):
        for table in (self.tri.get(tuple(ctx[-2:])), self.bi.get(ctx[-1]), self.uni):
            if table:
                words = list(table)
                p = np.array([table[w] for w in words], dtype=float)
                return words[rng.choice(len(words), p=p / p.sum())]
        return "</s>"

    def generate(self, rng, prefix=(), max_len=25):
        ctx = ["<s>", "<s>"] + list(prefix)
        out = list(prefix)
        while len(out) < max_len:
            w = self._next(ctx, rng)
            if w == "</s>":
                break
            out.append(w)
            ctx.append(w)
        return out


# ----------------------------------------------------------------------------- family 2: VAE
def adam_step(params, grads, state, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8):
    state["t"] += 1
    for k in params:
        state["m"][k] = b1 * state["m"][k] + (1 - b1) * grads[k]
        state["v"][k] = b2 * state["v"][k] + (1 - b2) * grads[k] ** 2
        mh = state["m"][k] / (1 - b1 ** state["t"])
        vh = state["v"][k] / (1 - b2 ** state["t"])
        params[k] -= lr * mh / (np.sqrt(vh) + eps)


def adam_init(params):
    return {"t": 0, "m": {k: np.zeros_like(v) for k, v in params.items()}, "v": {k: np.zeros_like(v) for k, v in params.items()}}


def softmax(a):
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=1, keepdims=True)


class BowVAE:
    """Multinomial bag-of-words VAE. Encoder: x -> (mu, logvar). Decoder: z -> word probabilities.
    Loss = reconstruction NLL + beta * KL(q(z|x) || N(0, I))."""

    def __init__(self, vocab_size, hidden=128, latent=16, seed=SEED):
        r = np.random.default_rng(seed)
        s = lambda a, b: r.normal(0, np.sqrt(2.0 / (a + b)), (a, b))
        self.p = {"W1": s(vocab_size, hidden), "b1": np.zeros(hidden), "Wm": s(hidden, latent), "bm": np.zeros(latent),
                  "Wv": s(hidden, latent) * 0.1, "bv": np.zeros(latent), "Wd": s(latent, vocab_size), "bd": np.zeros(vocab_size)}
        self.latent = latent

    def encode(self, X):
        Xn = X / np.maximum(X.sum(1, keepdims=True), 1)
        h = np.tanh(Xn @ self.p["W1"] + self.p["b1"])
        return Xn, h, h @ self.p["Wm"] + self.p["bm"], h @ self.p["Wv"] + self.p["bv"]

    def fit(self, X, epochs=30, batch=128, beta=1.0, seed=SEED):
        r, p, st = np.random.default_rng(seed), self.p, adam_init(self.p)
        for ep in range(epochs):
            order, losses = r.permutation(len(X)), []
            for i in range(0, len(X), batch):
                x = X[order[i:i + batch]]
                B = len(x)
                Xn, h, mu, lv = self.encode(x)
                std = np.exp(0.5 * lv)
                eps = r.standard_normal(mu.shape)
                z = mu + std * eps
                probs = softmax(z @ p["Wd"] + p["bd"])
                recon = -(x * np.log(probs + 1e-10)).sum(1)
                kl = -0.5 * (1 + lv - mu ** 2 - np.exp(lv)).sum(1)
                losses.append((recon + beta * kl).mean())
                dlog = (x.sum(1, keepdims=True) * probs - x) / B
                dz = dlog @ p["Wd"].T
                dmu = dz + beta * mu / B
                dlv = dz * eps * 0.5 * std + beta * 0.5 * (np.exp(lv) - 1) / B
                dh = dmu @ p["Wm"].T + dlv @ p["Wv"].T
                dpre = dh * (1 - h ** 2)
                g = {"Wd": z.T @ dlog, "bd": dlog.sum(0), "Wm": h.T @ dmu, "bm": dmu.sum(0), "Wv": h.T @ dlv, "bv": dlv.sum(0),
                     "W1": Xn.T @ dpre, "b1": dpre.sum(0)}
                adam_step(p, g, st)
            self.last_loss = float(np.mean(losses))
        return self

    def word_probs(self, z):
        return softmax(z @ self.p["Wd"] + self.p["bd"])

    def generate(self, rng, vocab, lengths, z=None):
        z = rng.standard_normal((1, self.latent)) if z is None else z
        probs = self.word_probs(z)[0]
        n = int(rng.choice(lengths))
        draws = rng.choice(len(vocab), size=n, p=probs / probs.sum())
        return [vocab[i] for i in draws]          # drawn in random order: the model has no word order


# ----------------------------------------------------------------------------- family 3: GAN
def sigmoid(a):
    return 1 / (1 + np.exp(-np.clip(a, -30, 30)))


class BowGAN:
    """G: z -> per-word probability, sampled to a hard 0/1 vector (straight-through gradient).
    D: 0/1 word-presence vector -> real/fake. Non-saturating generator loss."""

    def __init__(self, vocab_size, hidden=128, latent=16, seed=SEED):
        r = np.random.default_rng(seed)
        s = lambda a, b: r.normal(0, np.sqrt(2.0 / (a + b)), (a, b))
        self.g = {"W1": s(latent, hidden), "b1": np.zeros(hidden), "W2": s(hidden, vocab_size), "b2": np.full(vocab_size, -3.0)}
        self.d = {"W1": s(vocab_size, hidden), "b1": np.zeros(hidden), "W2": s(hidden, 1), "b2": np.zeros(1)}
        self.latent = latent

    def _G(self, z):
        h = np.maximum(0, z @ self.g["W1"] + self.g["b1"])
        return h, sigmoid(h @ self.g["W2"] + self.g["b2"])

    def _D(self, x):
        pre = x @ self.d["W1"] + self.d["b1"]
        h = np.where(pre > 0, pre, 0.2 * pre)
        return pre, h, sigmoid(h @ self.d["W2"] + self.d["b2"])

    def _D_backward(self, x, pre, h, out, dout):
        dpre_out = dout                                   # gradient w.r.t. the pre-sigmoid logit
        gW2, gb2 = h.T @ dpre_out, dpre_out.sum(0)
        dh = dpre_out @ self.d["W2"].T
        dpre = dh * np.where(pre > 0, 1.0, 0.2)
        return {"W1": x.T @ dpre, "b1": dpre.sum(0), "W2": gW2, "b2": gb2}, dpre @ self.d["W1"].T

    def fit(self, X, epochs=40, batch=128, seed=SEED):
        r = np.random.default_rng(seed)
        real_all = (X > 0).astype(float)
        sg, sd = adam_init(self.g), adam_init(self.d)
        for ep in range(epochs):
            order = r.permutation(len(real_all))
            for i in range(0, len(real_all), batch):
                real = real_all[order[i:i + batch]]
                B = len(real)
                # --- discriminator step: real -> 1, fake -> 0
                z = r.standard_normal((B, self.latent))
                _, p_fake = self._G(z)
                fake = (r.random(p_fake.shape) < p_fake).astype(float)
                pr, hr, outr = self._D(real)
                pf, hf, outf = self._D(fake)
                gr, _ = self._D_backward(real, pr, hr, outr, (outr - 1) / B)
                gf, _ = self._D_backward(fake, pf, hf, outf, outf / B)
                adam_step(self.d, {k: gr[k] + gf[k] for k in gr}, sd, lr=1e-3)
                # --- generator step: make D say "real" (straight-through: dL/dfake used as dL/dp)
                z = r.standard_normal((B, self.latent))
                h, p_fake = self._G(z)
                fake = (r.random(p_fake.shape) < p_fake).astype(float)
                pf, hf, outf = self._D(fake)
                _, dfake = self._D_backward(fake, pf, hf, outf, (outf - 1) / B)
                dlogit = dfake * p_fake * (1 - p_fake)
                dh = dlogit @ self.g["W2"].T * (h > 0)
                adam_step(self.g, {"W2": h.T @ dlogit, "b2": dlogit.sum(0), "W1": z.T @ dh, "b1": dh.sum(0)}, sg, lr=1e-3)
        return self

    def generate(self, rng, vocab):
        _, p = self._G(rng.standard_normal((1, self.latent)))
        p = p[0]
        present = np.where(rng.random(p.shape) < p)[0]
        return [vocab[i] for i in present[np.argsort(-p[present])]]   # ordered by probability: no word order exists


# ----------------------------------------------------------------------------- metrics
def metrics(samples, real_bigrams, domain, real_len):
    nonempty = [s for s in samples if s]
    pairs = [(a, b) for s in nonempty for a, b in zip(s, s[1:])]
    return {
        "empty_outputs_%": round(100 * (1 - len(nonempty) / len(samples)), 1),
        "valid_bigrams_%": round(100 * np.mean([p in real_bigrams for p in pairs]), 1) if pairs else 0.0,
        "banking_word_%": round(100 * np.mean([any(w in domain for w in s) for s in samples]), 1),
        "distinct_outputs_%": round(100 * len({" ".join(s) for s in samples}) / len(samples), 1),
        "mean_length": round(float(np.mean([len(s) for s in samples])), 1),
        "real_mean_length": round(real_len, 1),
    }


def run():
    t0 = time.time()
    train, docs, vocab, index, domain, real_bigrams = load_data()
    X = bow_matrix(docs, index)
    lengths = np.array([max(1, int(sum(w in index for w in d))) for d in docs])
    real_len = float(np.mean([len(d) for d in docs]))
    print(f"{len(docs)} train messages, vocabulary {len(vocab)} words, {len(domain)} banking keywords, mean length {real_len:.1f} words")

    rng = np.random.default_rng(SEED)
    results, samples_text = [], []

    # 1. autoregressive
    t = time.time()
    ar = TrigramLM().fit(docs)
    ar_t = time.time() - t
    ar_s = [ar.generate(rng) for _ in range(N_SAMPLES)]
    ar_prefix = np.mean([ar.generate(rng, prefix=p.split())[: len(p.split())] == p.split() for p in PREFIXES for _ in range(20)])

    # 2. VAE
    t = time.time()
    vae = BowVAE(len(vocab)).fit(X)
    vae_t = time.time() - t
    vae_s = [vae.generate(rng, vocab, lengths) for _ in range(N_SAMPLES)]
    vae_prefix = np.mean([vae.generate(rng, vocab, lengths)[: len(p.split())] == p.split() for p in PREFIXES for _ in range(20)])

    # 3. GAN
    t = time.time()
    gan = BowGAN(len(vocab)).fit(X)
    gan_t = time.time() - t
    gan_s = [gan.generate(rng, vocab) for _ in range(N_SAMPLES)]
    gan_prefix = np.mean([gan.generate(rng, vocab)[: len(p.split())] == p.split() for p in PREFIXES for _ in range(20)])

    for name, s, prefix, secs, extra in [
        ("Autoregressive (trigram LM)", ar_s, ar_prefix, ar_t, "likelihood: exact"),
        ("VAE (bag-of-words)", vae_s, vae_prefix, vae_t, f"ELBO loss/msg: {vae.last_loss:.1f}"),
        ("GAN (bag-of-words)", gan_s, gan_prefix, gan_t, "no likelihood"),
    ]:
        row = {"family": name, **metrics(s, real_bigrams, domain, real_len),
               "follows_given_prefix_%": round(100 * prefix, 1), "train_seconds": round(secs, 1), "note": extra}
        results.append(row)
        samples_text.append(f"== {name}\n" + "\n".join("  " + (" ".join(x) or "(empty)") for x in s[:8]))

    # AR can be steered by a prefix; show what that looks like
    samples_text.append("== Autoregressive, continuing a prefix\n" + "\n".join(f"  [{p}] -> {' '.join(ar.generate(rng, prefix=p.split()))}" for p in PREFIXES))

    df = pd.DataFrame(results)
    df.to_csv("evaluation/results/results_generative_families.csv", index=False)
    with open("evaluation/results/generative_families_samples.txt", "w") as f:
        f.write("\n\n".join(samples_text) + "\n")
    print(df.to_string(index=False))
    print()
    print("\n\n".join(samples_text))
    print(f"\ntotal time {time.time() - t0:.0f}s")
    return df


if __name__ == "__main__":
    run()
