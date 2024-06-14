import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr


data = {
    "first_sentence": [
        "I love programming.", "I love programming.",
        "The weather is nice today.", "I enjoy writing code.",
        "I hate bugs in my code.", "The weather is nice today.",
        "I love programming.", "The sky is clear.",
        "I enjoy writing code.", "I hate bugs in my code.",
        "The sky is blue.", "The sun is shining.",
        "I enjoy swimming.", "He is reading a book.",
        "She loves dancing.", "I enjoy writing code.",
        "I hate bugs in my code.", "The weather is nice today.",
        "The sky is clear.", "I love programming."
    ],
    "second_sentence": [
        "I enjoy writing code.", "I hate bugs in my code.",
        "The sun is shining.", "I hate bugs in my code.",
        "I enjoy writing code.", "I love programming.",
        "The weather is nice today.", "The sky is blue.",
        "I love programming.", "The sky is clear.",
        "I hate bugs in my code.", "The weather is nice today.",
        "He enjoys running.", "She is reading.",
        "She loves dancing.", "The sky is clear.",
        "The sky is blue.", "The sun is shining.",
        "The weather is nice today.", "I enjoy writing code."
    ],
    "similarity_score": [
        0.8, 0.3, 0.2, 0.4, 0.7, 0.5, 0.4, 0.9, 0.7, 0.6,
        0.1, 0.3, 0.2, 0.5, 1.0, 0.5, 0.4, 0.6, 0.8, 0.7
    ]
}

df = pd.DataFrame(data)


model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')


def compute_similarity(first_sentence, second_sentence):
    embeddings1 = model.encode(first_sentence)
    embeddings2 = model.encode(second_sentence)
    cosine_sim = util.pytorch_cos_sim(embeddings1, embeddings2)
    return cosine_sim.item()


df['predicted_similarity'] = df.apply(lambda row: compute_similarity(row['first_sentence'], row['second_sentence']), axis=1)


mse = mean_squared_error(df['similarity_score'], df['predicted_similarity'])


pearson_corr, _ = pearsonr(df['similarity_score'], df['predicted_similarity'])


spearman_corr, _ = spearmanr(df['similarity_score'], df['predicted_similarity'])


print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Pearson Correlation Coefficient: {pearson_corr:.4f}")
print(f"Spearman's Rank Correlation Coefficient: {spearman_corr:.4f}")


print(df)
