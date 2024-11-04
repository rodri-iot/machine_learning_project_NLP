# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score

import regex as re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Step 1:
# Import Dataset
df_raw = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv")

# Step 2 preprocessing
# Transform the nun to cat
df = df_raw.copy()
df['is_spam'] = df['is_spam'].astype(int)

# Remove duplicates
df = df.drop_duplicates().reset_index(drop=True)

print(f"Spam: {len(df.loc[df.is_spam == 1])}")
print(f"No spam: {len(df.loc[df.is_spam == 0])}")

# Function
def preprocess_url(url):
    if url is None:
        return ''
    
    url = url.lower()
    url = re.sub(r'''[^a-z\s]''', ' ', url ) # Remove any character tha is not a letter a-z or white space
    url = re.sub(r'''\s+[a-zA-Z]\s+''', ' ', url) # Remove white spaces
    url = re.sub(r'''\^[a-zA-Z]\s+''', ' ', url) # Remove white spaces
    url = re.sub(r'''\s+''', ' ', url).strip() # Remove multiples white spaces
    url = re.sub('''&lt;/?.*?&gt''', ' &lt;&gt; ', url) # Remove tags

    tokens = url.split() # Tokenize the text
    # Lemmatize and filter tokens
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token and token not in stop_words and len(token) > 3]

    return ' '.join(tokens).split()

df['processed_url'] = df['url'].apply(preprocess_url)

wordcloud = WordCloud(width=800, height=800, background_color='white', max_words=1000, min_font_size=20, random_state = 42).generate(str(df['processed_url']))

fig = plt.figure(figsize=(8, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Vectorize
X_interim = df['processed_url'].apply(lambda x: " ".join(x) if isinstance(x, list) else x)

vectorizer = TfidfVectorizer(max_features=5000, max_df=0.8, min_df=5)
X = vectorizer.fit_transform(X_interim).toarray()
y = df['is_spam']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build SVM
model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred

acc = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc}')

# Step 4: Optimized model
hyperparams = {
    'C': [0.1, 1, 10, 100, 1000],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'degree':[1, 2, 3]
}

grid_search = GridSearchCV(model, hyperparams, scoring='accuracy', cv=5)
grid_search

grid_search.fit(X_train, y_train)
print(f'Best hyperparams: {grid_search.best_params_}')

opt_model = SVC(C=1000, degree=1, gamma="auto", kernel="poly", random_state=42)
opt_model.fit(X_train, y_train)
y_pred_opt = opt_model.predict(X_test)
print(f'Accuracy Opt: {accuracy_score(y_test, y_pred_opt)}')

