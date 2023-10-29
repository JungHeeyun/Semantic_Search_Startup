import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import matplotlib.pyplot as plt
import requests

# For the News Aggregator
NEWS_API_ENDPOINT = 'https://newsapi.org/v2/top-headlines'


def load_data():
    df = pd.read_csv('converted.csv')
    df = df.dropna(subset=['Description'])  # Drop rows where Description is NaN
    return df  # Returning the entire DataFrame

def load_embeddings():
    return np.load('corpus_embeddings.npy')

def search_similar_sentences(user_input, df, corpus_embeddings, k=10000):
    embedder = SentenceTransformer('distilroberta-base-paraphrase-v1')
    user_embedding = embedder.encode([user_input])
    
    dimension = len(corpus_embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(corpus_embeddings.astype('float32'))
    
    _, top_indices = index.search(np.array(user_embedding).astype('float32'), k=k)
    
    similar_rows = [df.iloc[idx] for idx in top_indices[0]]
    
    return similar_rows

def plot_company_data(company_row, threshold=0.99):
    
    # Convert columns to datetime and numeric, considering '—' as NaN
    df['Last Funding Date'] = pd.to_datetime(df['Last Funding Date'], errors='coerce')
    df['Total Funding Amount'] = pd.to_numeric(df['Total Funding Amount'], errors='coerce')
    
    # Drop rows where either 'Last Funding Date' or 'Total Funding Amount' is NaN
    cleaned_df = df.dropna(subset=['Last Funding Date', 'Total Funding Amount'])
    
    # Removing outliers, keeping only companies below the certain percentile of Total Funding Amount
    threshold_value = cleaned_df['Total Funding Amount'].quantile(threshold)
    cleaned_df = cleaned_df[cleaned_df['Total Funding Amount'] < threshold_value]
    
    # Plot all companies
    plt.figure(figsize=(10, 5))
    plt.scatter(cleaned_df['Last Funding Date'], cleaned_df['Total Funding Amount'], 
                label='All Companies', alpha=0.5)
    
    # Highlighting the selected company in the basic graph
    selected_company_funding = pd.to_numeric(company_row['Total Funding Amount'], errors='coerce')
    if selected_company_funding < threshold_value:
        plt.scatter(pd.to_datetime(company_row['Last Funding Date']), selected_company_funding, 
                    color='red', label='Selected Company')
    
    plt.title('All Companies: Last Funding Date vs Total Funding Amount')
    plt.xlabel('Last Funding Date')
    plt.ylabel('Total Funding Amount')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)
    
    # Extract industries from the selected company
    industries_to_highlight = company_row['Industries'].split(', ')
    
    for industry in industries_to_highlight:
        plt.figure(figsize=(10, 5))
        
        # Highlighting the companies in the same industry as the selected company
        industry_df = cleaned_df[cleaned_df['Industries'].str.contains(industry, na=False, regex=False)]
        plt.scatter(industry_df['Last Funding Date'], industry_df['Total Funding Amount'], 
                    color='green', label=f'Companies in {industry}', alpha=0.5)
        
        # Highlighting the selected company
        selected_company_funding = pd.to_numeric(company_row['Total Funding Amount'], errors='coerce')
        if selected_company_funding < threshold_value:
            plt.scatter(pd.to_datetime(company_row['Last Funding Date']), selected_company_funding, 
                        color='red', label='Selected Company')
        
        plt.title(f'{industry}: Last Funding Date vs Total Funding Amount')
        plt.xlabel('Last Funding Date')
        plt.ylabel('Total Funding Amount')
        plt.xticks(rotation=45)
        plt.legend()
        st.pyplot(plt)

def plot_user_and_company_data(user_input, df):
    plt.figure(figsize=(10, 5))
    
    # Convert columns to datetime and numeric, considering '—' as NaN
    df['Founded Date'] = pd.to_datetime(df['Founded Date'], errors='coerce')
    df['Total Funding Amount'] = pd.to_numeric(df['Total Funding Amount'], errors='coerce')
    
    # Drop rows where either 'Last Funding Date' or 'Total Funding Amount' is NaN
    cleaned_df = df.dropna(subset=['Founded Date', 'Total Funding Amount'])
    
    plt.scatter(cleaned_df['Founded Date'], cleaned_df['Total Funding Amount'], 
                label='Other Companies')
    
    # User Input
    founded_year = pd.to_datetime(user_input['founded_year'], format='%Y')
    total_funding_amount = user_input['diff_funding']
    
    plt.scatter(founded_year, total_funding_amount, 
                color='red', label='User Input')
    
    plt.title('Founded Year vs Total Funding Amount')
    plt.xlabel('Founded Year')
    plt.ylabel('Total Funding Amount')
    plt.xticks(rotation=45)
    plt.legend()
    
    st.pyplot(plt)

def company_finder(df, corpus_embeddings, user_input, country_filter=None, industry_filter=None, 
                   last_funding_type=None, funding_status=None):
    
    if user_input:
        similar_rows = search_similar_sentences(user_input, df, corpus_embeddings)
        
        # Applying filters
        if country_filter and 'All' not in country_filter:
            similar_rows = [row for row in similar_rows if any(country.lower() in row['Headquarters Location'].lower() for country in country_filter)]
            
        if industry_filter and 'All' not in industry_filter:
            similar_rows = [row for row in similar_rows if any(industry.lower() in str(row['Industries']).lower() for industry in industry_filter)]
            
        if last_funding_type and 'All' not in last_funding_type:
            similar_rows = [row for row in similar_rows if row['Last Funding Type'] in last_funding_type]
            
        if funding_status and 'All' not in funding_status:
            similar_rows = [row for row in similar_rows if row['Funding Status'] in funding_status]
        
        # Display only top 5 companies after applying the country and industry filters
        top_5_rows = similar_rows[:5]
        
        st.write("## Top 5 similar companies:")
        for i, row in enumerate(top_5_rows, 1):
            with st.expander(f"{i}. {row['Organization Name']}"):
                st.markdown(f"- **Description**: {row['Description']}")
                st.markdown(f"- **Last Funding Type**: {row['Last Funding Type']}")
                st.markdown(f"- **Total Funding Amount**: {row['Total Funding Amount']}")
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
                if st.button(f"Show Graph for {row['Organization Name']}"):
                    plot_company_data(row)

def fetch_news(country, category=None):
    params = {
        'country': country,
        'apiKey': NEWS_API_KEY
    }
    if category:
        params['category'] = category
    response = requests.get(NEWS_API_ENDPOINT, params=params)
    return response.json()

df = load_data()

corpus_embeddings = load_embeddings()

st.sidebar.title('Navigation')
tabs = ["Company Finder", "Data Analysis", "News Aggregator"]
selected_tab = st.sidebar.selectbox("Choose a tab", tabs)

if selected_tab == 'Company Finder':
    st.title('Company Finder')
    
    user_input = st.text_input("Describe the characteristic of company: ")
    
    # Country filter
    country_options = ['All', 'China', 'Hong Kong', 'United States', 'Singapore']
    selected_country = st.multiselect('Select a country', country_options)
    
    # Industry filter
    industry_options = [
        'Software', 'Information Technology', 'Health Care', 'Financial Services', 
        'Artificial Intelligence', 'E-Commerce', 'Manufacturing', 'Internet', 
        'Biotechnology', 'SaaS', 'Medical', 'FinTech', 'Consulting', 'Finance', 
        'Machine Learning', 'Analytics', 'Blockchain', 'Education', 'Mobile', 
        'Advertising', 'Food and Beverage', 'Apps', 'Marketing', 'Electronics', 
        'Big Data', 'Enterprise Software', 'Retail', 'Others'
    ]
    selected_industry = st.multiselect('Select an industry', industry_options)
    
    # Last Funding Type filter
    last_funding_type_options = ['All', 'Series A', 'Seed', 'Series B', 'Series C', 'Pre-Seed']
    selected_last_funding_type = st.multiselect('Select a last funding type', last_funding_type_options)
    
    # Funding Status filter
    funding_status_options = ['All', 'Early Stage Venture', 'Seed', 'Late Stage Venture', 'M&A', 'IPO', 'Private Equity']
    selected_funding_status = st.multiselect('Select a funding status', funding_status_options)
    
    company_finder(df, corpus_embeddings, user_input, country_filter=selected_country, 
                   industry_filter=selected_industry, last_funding_type=selected_last_funding_type, 
                   funding_status=selected_funding_status)

elif selected_tab == 'Data Analysis':
    st.title('Data Analysis')
    st.markdown('**Enter the company information you want to analyze**')
    # Getting user inputs
    founded_year = st.number_input('Founded Year',min_value=2000)
    operating_status = st.selectbox('Operating Status', options=['Active', 'Closed'])
    num_employees = st.number_input('Number of Employees', min_value=0)
    num_founders = st.number_input('Number of Founders', min_value=1)
    num_funding_rounds = st.number_input('Number of Funding Rounds', min_value=1)
    num_investors = st.number_input('Number of Investors', min_value=1)
    ipo_status = st.selectbox('IPO Status', options=['Private', 'Public'])
    diff_funding = st.number_input('Total Funding Amount(USD)', min_value=1)
    quarter_difference = st.number_input('Company Duration time(in Quarter)', min_value=1)
    industry_options = [
        'Software', 'Information Technology', 'Health Care', 'Financial Services', 
        'Artificial Intelligence', 'E-Commerce', 'Manufacturing', 'Internet', 
        'Biotechnology', 'SaaS', 'Medical', 'FinTech', 'Consulting', 'Finance', 
        'Machine Learning', 'Analytics', 'Blockchain', 'Education', 'Mobile', 
        'Advertising', 'Food and Beverage', 'Apps', 'Marketing', 'Electronics', 
        'Big Data', 'Enterprise Software', 'Retail', 'Others'
    ]
    industry = st.selectbox('Industry', options=industry_options)    
    headquarter_location = st.text_input('Headquarter Location')
    description = st.text_area('Company Description')
    
    if st.button('Generate Graph'):
        # Where you call this function
        user_input = {
            'founded_year': founded_year,  # Assuming you have a variable named founded_year
            'diff_funding': diff_funding  # Assuming you have a variable named diff_funding
        }
        plot_user_and_company_data(user_input, df)
        st.markdown('**🔔 Note:** Additional data visualizations will be updated soon! Stay tuned for more insightful graphs and analyses.')

    if st.button('Find Similar Companies'):
        if description:  # Ensure that the description is not empty
            company_finder(df, corpus_embeddings, description)

elif selected_tab == 'News Aggregator':
    st.title('News Aggregator')
    
    # Choose the country
    countries = ['us', 'cn', 'hk', 'sg']  # Ensure country codes are in lowercase
    selected_country = st.sidebar.selectbox('Select a country', countries)
    
    # Choose the category
    categories = ['business', 'technology', 'science', 'all']
    selected_category = st.sidebar.selectbox('Select a category', categories)
    
    # Fetch the news
    if selected_category == 'all':
        news = fetch_news(selected_country)
    else:
        news = fetch_news(selected_country, category=selected_category)
    
    # Display the news articles
    for article in news['articles']:
        article_url = article['url']
        article_title = article['title']
        # Creating a markdown string with the title as a clickable link
        st.markdown(f"### [{article_title}]({article_url})")
        st.write('---')  # Adding a horizontal line to separate articles
