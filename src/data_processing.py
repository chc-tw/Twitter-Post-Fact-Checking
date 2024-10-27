import preprocessor as p
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import tqdm

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def tokenize(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    return [token for token in tokens if token not in stop_words]

def lemma(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_text(text):
    text = p.clean(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub('<[^<]+?>', '', text)
    text = text.strip()
    text = tokenize(text)
    text = lemma(text)
    text = " ".join(text)
    return text

def fetch_claim_label(data, test=False):
    claims = []
    for t in data['metadata']:
        text = preprocess_text(t['claim'])
        claims.append(text)

    if test:
        return claims
    else:
        labels = []
        for t in data['label']:
            labels.append(t['rating'])
        return claims, labels

def evidence_extract(data, claims, articles_path):
    evidence_set = {}
    evidence = []
    preprocessing = Pipeline([('tfidf', TfidfVectorizer())])
    
    for i, r in enumerate(tqdm.tqdm(data["metadata"])):
        for t in r['premise_articles'].values():
            js = pd.read_json(articles_path + t)
            if js.empty:
                continue
            js = js[0].astype(str).values.tolist()
            js.append(claims[i])
            Tevidence = preprocessing.fit_transform(js)
            Tclaim = Tevidence[-1]
            similarities = cosine_similarity(Tclaim, Tevidence[:-1])
            ind = np.argsort(similarities[0])[-5:]
            for k in ind:
                if similarities[0][k] > 0.05:
                    if "*" not in js[k] and "_" not in js[k] and js[k] not in evidence:
                        evidence.append(js[k])
        evidence_set.update({i: evidence.copy()})
        evidence.clear()
    return evidence_set
