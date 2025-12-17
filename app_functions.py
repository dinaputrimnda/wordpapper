import pymupdf as fitz
import re
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from collections import Counter
from itertools import combinations
from nltk.corpus import stopwords
import nltk

# Hanya butuh stopwords (AMAN)
nltk.download("stopwords")


# ===============================
# PDF TEXT EXTRACTION
# ===============================
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + " "
    return text


# ===============================
# TEXT PREPROCESSING (SENTENCE-BASED, TANPA punkt)
# ===============================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\.!? ]", " ", text)

    # Split kalimat MANUAL (STABIL)
    raw_sentences = re.split(r"[.!?]+", text)

    stop_words = set(stopwords.words("indonesian"))

    clean_sentences = []
    for sent in raw_sentences:
        words = sent.split()
        words = [w for w in words if w not in stop_words and len(w) > 3]
        if len(words) > 1:
            clean_sentences.append(words)

    return clean_sentences


# ===============================
# BUILD SENTENCE-BASED GRAPH
# ===============================
def build_sentence_cooccurrence_graph(sentences, top_n=80):
    all_words = [w for sent in sentences for w in sent]
    vocab = Counter(all_words).most_common(top_n)
    vocab_words = set([w for w, _ in vocab])

    G = nx.Graph()

    for sent in sentences:
        sent_words = list(set([w for w in sent if w in vocab_words]))
        for w1, w2 in combinations(sent_words, 2):
            if G.has_edge(w1, w2):
                G[w1][w2]["weight"] += 1
            else:
                G.add_edge(w1, w2, weight=1)

    return G


# ===============================
# CO-OCCURRENCE MATRIX
# ===============================
def build_cooccurrence_matrix(G):
    words = list(G.nodes)
    matrix = pd.DataFrame(0, index=words, columns=words)

    for u, v, data in G.edges(data=True):
        matrix.loc[u, v] = data["weight"]
        matrix.loc[v, u] = data["weight"]

    return matrix


# ===============================
# PAGERANK
# ===============================
def compute_pagerank(G):
    if len(G.nodes) == 0:
        return pd.DataFrame(columns=["Kata", "PageRank"])

    pr = nx.pagerank(G, weight="weight")
    df = pd.DataFrame(pr.items(), columns=["Kata", "PageRank"])
    return df.sort_values(by="PageRank", ascending=False)


# ===============================
# VISUALIZATION
# ===============================
def visualize_word_graph(G):
    fig, ax = plt.subplots(figsize=(16, 16))

    if len(G.nodes) == 0:
        ax.text(0.5, 0.5, "Graph kosong", ha="center", va="center")
        ax.axis("off")
        return fig

    pr = nx.pagerank(G, weight="weight")
    node_sizes = [pr[n] * 12000 for n in G.nodes]

    pos = nx.spring_layout(G, k=1.6, seed=42)

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color="#1f77b4",
        alpha=0.85,
        ax=ax
    )

    nx.draw_networkx_edges(
        G, pos,
        width=1,
        alpha=0.4,
        ax=ax
    )

    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_color="black",
        ax=ax
    )

    ax.axis("off")
    return fig
