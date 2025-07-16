import nltk
import streamlit as st
from nltk import PorterStemmer, WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import os

nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.download('punkt', download_dir=nltk_data_path, force=True)
nltk.download('stopwords', download_dir=nltk_data_path, force=True)
nltk.data.path.append(nltk_data_path)

st.set_page_config(page_title="Insights", layout="wide")#, initial_sidebar_state="collapsed")

# st.markdown(
#     """
# <style>
#     [data-testid="stSidebarCollapsedControl"] {
#         display: none
#     }
# </style>
# """,
#     unsafe_allow_html=True,
# )

grouped = st.session_state.step_2_df
group_col = st.session_state.group_col
try:
    api_key = st.session_state.api_key
except AttributeError:
    pass

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_texts(texts, remove_stopwords, stem, lemmatize):
    processed = []
    for doc in texts:
        tokens = word_tokenize(doc.lower())
        tokens = [t for t in tokens if t.isalpha()]  # keep only words
        if remove_stopwords:
            tokens = [t for t in tokens if t not in stop_words]
        if stem:
            tokens = [stemmer.stem(t) for t in tokens]
        elif lemmatize:
            tokens = [lemmatizer.lemmatize(t) for t in tokens]
        processed.append(" ".join(tokens))
    return processed

def get_top_ngrams(texts, ngram_size=1, top_n=10, remove_stopwords=True):
    custom_stopwords = set([
            "ve", "re", "ll", "d", "s", "t", "m", "nt",  # contraction artifacts
            "amp"  # sometimes shows up from HTML entities
    ])
    stopword_list = list(stop_words.union(custom_stopwords)) if remove_stopwords else None
    vectorizer = CountVectorizer(
        stop_words = stopword_list,
        ngram_range=(ngram_size, ngram_size),
        token_pattern=r'\b[a-zA-Z]{2,}\b'  # keep only real words
    )
    X = vectorizer.fit_transform(texts)
    sum_words = X.sum(axis=0)
    freqs = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    sorted_ngrams = sorted(freqs, key=lambda x: x[1], reverse=True)[:top_n]
    return sorted_ngrams

def get_top_tfidf(texts, group_names, ngram_size=1, top_n=10, remove_stopwords=True):
    custom_stopwords = set([
        "ve", "re", "ll", "d", "s", "t", "m", "nt", "amp"
    ])
    stopword_list = list(stop_words.union(custom_stopwords)) if remove_stopwords else None

    vectorizer = TfidfVectorizer(
        stop_words=stopword_list,
        ngram_range=(ngram_size, ngram_size),
        token_pattern=r'\b[a-zA-Z]{2,}\b'
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    group_top_terms = []
    for i, group in enumerate(group_names):
        row = tfidf_matrix[i].toarray().flatten()
        top_indices = row.argsort()[::-1][:top_n]
        terms = [(feature_names[j], row[j]) for j in top_indices if row[j] > 0]
        group_top_terms.append((group, terms))

    return group_top_terms

with st.sidebar:
    st.markdown("### ⚙️ Preprocessing Options")
    tfidf_mode = st.checkbox("Use TF-IDF instead of raw frequency", value=False)

    remove_stopwords = st.checkbox("Remove stop words", value=True)
    apply_stemming = st.checkbox("Apply stemming", value=False)
    apply_lemmatization = st.checkbox("Apply lemmatization (slower)", value=False)


    if apply_stemming and apply_lemmatization:
        st.warning("Choose either stemming or lemmatization — not both.")
        st.stop()

    ngram_range = st.selectbox("Select n-gram size", options=["Unigrams (1)", "Bigrams (2)", "Trigrams (3)"])
    top_n = st.slider("How many top n-grams to show?", min_value=5, max_value=30, value=10)


st.markdown("### 📊 Top N-Grams Per Group")

n = int(ngram_range.split("(")[1][0])  # converts "Bigrams (2)" → 2
group_texts = []
group_labels = []
for i, row in grouped.iterrows():
    group_name = row[group_col]
    text = row['combined_text']
    group_texts.append(" ".join(preprocess_texts([text], remove_stopwords, apply_stemming, apply_lemmatization)))
    group_labels.append(group_name)

if tfidf_mode:
    tfidf_results = get_top_tfidf(group_texts, group_labels, ngram_size=n, top_n=top_n, remove_stopwords=remove_stopwords)
    for group_name, terms in tfidf_results:
        st.markdown(f"#### 🔹 Group: {group_name}")
        if not terms:
            st.write("No meaningful TF-IDF terms found.")
            continue
        labels, scores = zip(*terms)
        fig, ax = plt.subplots()
        sns.barplot(x=list(scores), y=list(labels), ax=ax, palette="Greens_d")
        ax.set_xlabel("TF-IDF Score")
        ax.set_ylabel(f"Top {ngram_range}")
        st.pyplot(fig)
else:
    for group_name, text in zip(group_labels, group_texts):
        st.markdown(f"#### 🔹 Group: {group_name}")
        ngrams = get_top_ngrams([text], ngram_size=n, top_n=top_n, remove_stopwords=remove_stopwords)
        if not ngrams:
            st.write("No meaningful n-grams found.")
            continue
        terms, freqs = zip(*ngrams)
        fig, ax = plt.subplots()
        sns.barplot(x=list(freqs), y=list(terms), ax=ax, palette="Blues_d")
        ax.set_xlabel("Frequency")
        ax.set_ylabel(f"Top {ngram_range}")
        st.pyplot(fig)

st.markdown("### GPT Analysis: Qualitative Differences Between Survey Groups")

query = f"Following are survey responses grouped by {group_col}. Summarise the qualitative differences between the groups and extract real world patterns and insights:\n\n"
for _,group in grouped.iterrows():
    query += f"Group {group[group_col]}: {group['combined_text']}\n"


if 'api_key' in locals():
#    try:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # or "gpt-3.5-turbo" if needed
        messages=[{"role": "user", "content": query}]
    )

    corrected = response.choices[0].message.content.strip()
    st.markdown(corrected)
#    except Exception as e:
#        st.markdown(f"ERROR: {e}")
else:
    st.warning("No OpenAI API key detected")

