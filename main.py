import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

def load_data():
    df = pd.read_csv('/Users/jeonghuiyun/PycharmProjects/pythonProject2/distillBERT/converted.csv')
    selected_sentences = df['Description'].dropna()  
    return selected_sentences.tolist()

def load_embeddings():
    return np.load('/Users/jeonghuiyun/PycharmProjects/pythonProject2/distillBERT/corpus_embeddings.npy')

def search_similar_sentences(user_input, corpus, corpus_embeddings):
    embedder = SentenceTransformer('distilroberta-base-paraphrase-v1')
    user_embedding = embedder.encode([user_input])
    
    dimension = len(corpus_embeddings[0])  
    index = faiss.IndexFlatL2(dimension)
    index.add(corpus_embeddings.astype('float32'))  
    
    _, top_indices = index.search(np.array(user_embedding).astype('float32'), k=5)
    
    return [corpus[idx] for idx in top_indices[0]]

corpus = load_data()
corpus_embeddings = load_embeddings()

st.title('Company Finder')

user_input = st.text_input("Describe the characteristic of company: ")

if user_input:
    similar_sentences = search_similar_sentences(user_input, corpus, corpus_embeddings)
    st.write("## Top 5 similar sentences:")
    for i, sentence in enumerate(similar_sentences, 1):
        st.write(f"{i}. {sentence}")
