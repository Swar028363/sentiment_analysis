import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D 


# ============================================================
# GET POSITIVE AND NEGATIVE WORDS FROM A SENTENCE 
# ============================================================
def get_positive_words(words: list[str], positive_words: set[str] = set()) -> list[str]:
    """Get all the Positive words form the words list. Positive words are like good, great, love, etc."""
    return [w for w in words if w in positive_words]

def get_negative_words(words: list[str], negative_words: set[str] = set()) -> list[str]:
    """Get all the Negative words from the words list. Negative words are like bad, worst, hate, etc."""
    return [w for w in words if w in negative_words]

# ============================================================
# FUNCTION TO PLOT THE WORD VECS
# ============================================================
def tsne_plot(
    model, 
    number_of_words=100, 
    figsize=(12, 10), 
    random_state=None, 
    save_figure=False, 
    save_path="tsne_plot.png", 
    plot_3d=False
):
    """
    Creates and plots a t-SNE visualization of word embeddings.
    
    Parameters:
        model: gensim Word2Vec/FastText model
        number_of_words: int — number of words to visualize
        figsize: tuple — figure size
        random_state: int or None — seed for reproducibility
        plot_3d: bool — if True, plot in 3D, else 2D
    """
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Sample words without repetition (if enough vocabulary)
    vocab = model.wv.index_to_key
    sample_size = min(number_of_words, len(vocab))
    words = random.sample(vocab, sample_size)
    
    # Extract embeddings
    tokens = np.array([model.wv[word] for word in words])
    
    # Create TSNE model
    tsne = TSNE(n_components=3 if plot_3d else 2, random_state=random_state, init="pca")
    new_values = tsne.fit_transform(tokens)
    
    # Plot
    plt.figure(figsize=figsize)
    
    if plot_3d:
        ax = plt.axes(projection='3d')
        ax.scatter(new_values[:, 0], new_values[:, 1], new_values[:, 2], c='steelblue', s=50, alpha=0.7)
        for i, word in enumerate(words):
            ax.text(new_values[i, 0], new_values[i, 1], new_values[i, 2], word, fontsize=8)
        ax.set_title("t-SNE 3D Word Embeddings")
    else:
        plt.scatter(new_values[:, 0], new_values[:, 1], c='steelblue', s=50, alpha=0.7)
        for i, word in enumerate(words):
            plt.annotate(word,
                         xy=(new_values[i, 0], new_values[i, 1]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.title("t-SNE 2D Word Embeddings")
    if save_figure:
        plt.savefig(save_path)
    plt.show()


# ============================================================
# POOLING FUNCTIONS - Users can modify this dict
# ============================================================
import numpy as np

POOLING_FUNCTIONS = {
    # === Basic Pooling ===
    'mean': lambda vecs: np.mean(vecs, axis=0) if len(vecs) > 0 else None,
    'max': lambda vecs: np.max(vecs, axis=0) if len(vecs) > 0 else None,
    'min': lambda vecs: np.min(vecs, axis=0) if len(vecs) > 0 else None,
    'sum': lambda vecs: np.sum(vecs, axis=0) if len(vecs) > 0 else None,
    'median': lambda vecs: np.median(vecs, axis=0) if len(vecs) > 0 else None,
    'std': lambda vecs: np.std(vecs, axis=0) if len(vecs) > 0 else None,
    'range': lambda vecs: (np.max(vecs, axis=0) - np.min(vecs, axis=0)) if len(vecs) > 0 else None,

    # === Normalized Pooling ===
    'l2_mean': lambda vecs: (
        lambda v: v / (np.linalg.norm(v) + 1e-12)
    )(np.mean(vecs, axis=0)) if len(vecs) > 0 else None,

    'l1_mean': lambda vecs: (
        lambda v: v / (np.sum(np.abs(v)) + 1e-12)
    )(np.mean(vecs, axis=0)) if len(vecs) > 0 else None,

    'unit_mean': lambda vecs: np.mean(
        vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12),
        axis=0
    ) if len(vecs) > 0 else None,

    # === Weighted Pooling ===
    'weighted_mean': lambda vecs: (
        lambda weights, v: np.average(v, axis=0, weights=weights)
    )(*(
        (np.linalg.norm(vecs, axis=1), vecs)
    )) if len(vecs) > 0 else None,

    'softmax_weighted_mean': lambda vecs: (
        lambda weights, v: np.sum(v * weights[:, None], axis=0)
    )(
        np.exp(np.linalg.norm(vecs, axis=1)) /
        np.sum(np.exp(np.linalg.norm(vecs, axis=1))),
        vecs
    ) if len(vecs) > 0 else None,

    # === Structural Pooling ===
    'positional_decay': lambda vecs: (
        np.average(vecs, axis=0, weights=np.linspace(1, 0.1, len(vecs)))
    ) if len(vecs) > 0 else None,

    'head_tail_concat': lambda vecs: np.concatenate([
        vecs[0], vecs[-1]
    ]) if len(vecs) >= 2 else None,

    # === Statistical Pooling ===
    'mean_std_concat': lambda vecs: np.concatenate([
        np.mean(vecs, axis=0),
        np.std(vecs, axis=0)
    ]) if len(vecs) > 0 else None,

    'mean_max_concat': lambda vecs: np.concatenate([
        np.mean(vecs, axis=0),
        np.max(vecs, axis=0)
    ]) if len(vecs) > 0 else None,

    'mean_max_std_concat': lambda vecs: np.concatenate([
        np.mean(vecs, axis=0),
        np.max(vecs, axis=0),
        np.std(vecs, axis=0)
    ]) if len(vecs) > 0 else None,

    # === Nonlinear Pooling ===
    'tanh_mean': lambda vecs: np.tanh(np.mean(vecs, axis=0)) if len(vecs) > 0 else None,
    'relu_mean': lambda vecs: np.mean(np.maximum(vecs, 0), axis=0) if len(vecs) > 0 else None,

    # === Distribution-Sensitive Pooling ===
    'covariance': lambda vecs: np.cov(vecs, rowvar=False).flatten() if len(vecs) > 1 else None,
    'skewness': lambda vecs: (
        np.mean(((vecs - np.mean(vecs, axis=0)) /
                 (np.std(vecs, axis=0) + 1e-12))**3, axis=0)
    ) if len(vecs) > 0 else None,

    # === Power / Geometric ===
    'power_mean': lambda vecs, p=3: (
        np.power(np.mean(np.power(np.abs(vecs), p), axis=0), 1/p)
    ) if len(vecs) > 0 else None,

    'geo_mean': lambda vecs: (
        np.exp(np.mean(np.log(np.abs(vecs) + 1e-12), axis=0))
    ) if len(vecs) > 0 else None,
}


# ============================================================
# VECTOR CONVERSION FUNCTIONS
# ============================================================
def apply_pooling(word_vecs, pooling_strategy, vec_dim):
    """Apply pooling strategy to word vectors."""
    if len(word_vecs) == 0:
        return np.zeros(vec_dim)
    
    pooling_func = POOLING_FUNCTIONS.get(pooling_strategy)
    if not pooling_func:
        raise ValueError(f"Unknown pooling: {pooling_strategy}")
    
    pooled = pooling_func(np.array(word_vecs))
    return pooled if pooled is not None else np.zeros(vec_dim)


def tokens_to_vectors_pooled(tokens_batch, w2v_model, pooling_strategy):
    """Convert tokens to vectors using pooling (mean, max, min, sum)."""
    vec_dim = w2v_model.vector_size
    result = []
    
    for tokens in tokens_batch:
        word_vecs = [w2v_model.wv[token] for token in tokens if token in w2v_model.wv]
        vec = apply_pooling(word_vecs, pooling_strategy, vec_dim)
        result.append(vec)
    
    return np.array(result)


def tokens_to_vectors_flatten(tokens_batch, w2v_model, max_words):
    """Convert tokens to flattened vectors with padding."""
    vec_dim = w2v_model.vector_size
    result = []
    
    for tokens in tokens_batch:
        word_vecs = []
        for token in tokens[:max_words]:
            if token in w2v_model.wv:
                word_vecs.append(w2v_model.wv[token])
            else:
                word_vecs.append(np.zeros(vec_dim))
        
        # Pad to max_words
        while len(word_vecs) < max_words:
            word_vecs.append(np.zeros(vec_dim))
        
        result.append(np.array(word_vecs).flatten())
    
    return np.array(result)


def convert_tokens_to_vectors(tokens_list, w2v_model, pooling, batch_size, max_words=None):
    """
    Convert all tokens to vectors in batches.
    
    Args:
        tokens_list: list of token lists
        w2v_model: gensim Word2Vec model
        pooling: pooling strategy
        batch_size: batch size for processing
        max_words: max words (only for flatten)
    
    Returns:
        numpy array of vectors
    """
    if pooling == 'flatten' and max_words is None:
        raise ValueError("max_words required for flatten pooling")
    
    X_batches = []
    for i in range(0, len(tokens_list), batch_size):
        batch = tokens_list[i:i+batch_size]
        
        if pooling == 'flatten':
            X_batch = tokens_to_vectors_flatten(batch, w2v_model, max_words)
        else:
            X_batch = tokens_to_vectors_pooled(batch, w2v_model, pooling)
        
        X_batches.append(X_batch)
    
    return np.vstack(X_batches)