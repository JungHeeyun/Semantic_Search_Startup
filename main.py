import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

def load_data():
    df = pd.read_csv('converted.csv')
    df = df.dropna(subset=['Description'])  # Drop rows where Description is NaN
    return df  # Returning the entire DataFrame

def load_embeddings():
    return np.load('corpus_embeddings.npy')

def search_similar_sentences(user_input, df, corpus_embeddings):
    embedder = SentenceTransformer('distilroberta-base-paraphrase-v1')
    user_embedding = embedder.encode([user_input])
    
    corpus = df['Description'].tolist()
    
    dimension = len(corpus_embeddings[0])  
    index = faiss.IndexFlatL2(dimension)
    index.add(corpus_embeddings.astype('float32'))  
    
    _, top_indices = index.search(np.array(user_embedding).astype('float32'), k=5)
    
    # Getting other columns as well for the top similar sentences
    similar_rows = [df.iloc[idx] for idx in top_indices[0]]
    
    return similar_rows

df = load_data()
corpus_embeddings = load_embeddings()

st.title('Company Finder')

user_input = st.text_input("Describe the characteristic of company: ")

if user_input:
    similar_rows = search_similar_sentences(user_input, df, corpus_embeddings)
    st.write("## Top 5 similar companies:")
    for i, row in enumerate(similar_rows, 1):
        # Using an expander for each company
        with st.expander(f"{i}. {row['Organization Name']}"):
            st.markdown(f"- **Description**: {row['Description']}")
            st.markdown(f"- **Last Funding Type**: {row['Last Funding Type']}")
            st.markdown(f"- **Total Funding Amount**: USD{row['Total Funding Amount']}")
            st.markdown(f"- **Industries**: {row['Industries']}")
            st.markdown(f"- **Founded Date**: {row['Founded Date']}")
            st.markdown(f"- **Operating Status**: {row['Operating Status']}")
            st.markdown(f"- **Funding Status**: {row['Funding Status']}")
            st.markdown(f"- **Headquarters Location**: {row['Headquarters Location']}")
            st.markdown(f"- **Website**: {row['Website']}")
            st.markdown(f"- **Contact Email**: {row['Contact Email']}")
            st.markdown(f"- **Last Funding Date**: {row['Last Funding Date']}")
            st.markdown(f"- **Number of Employees**: {row['Number of Employees']}")
            st.markdown(f"- **Number of Founders**: {row['Number of Founders']}")
            st.markdown(f"- **Number of Funding Rounds**: {row['Number of Funding Rounds']}")
            st.markdown(f"- **Founders**: {row['Founders']}")
            st.markdown(f"- **Number of Investors**: {row['Number of Investors']}")
            st.markdown(f"- **Top 5 Investors**: {row['Top 5 Investors']}")
            st.markdown(f"- **IPO Status**: {row['IPO Status']}")
            st.markdown(f"- **Similar Companies**: {row['Similar Companies']}")


      

