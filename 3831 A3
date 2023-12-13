#Data crawling
!pip install advertools

import advertools as adv
from advertools import crawl
import pandas as pd

site ="https://simplifiedsearch.net/"
crawl(site,'simp.jl',follow_links=True)
crawl_df = pd.read_json('simp.jl',lines=True)
crawl_df

columns = list(crawl_df)

columns

#Data Wrangling
# Select the desired columns
selected_columns = ['url', 'title', 'meta_desc', 'body_text', 'jsonld_@type']

# Create a new DataFrame with the selected columns
selected_df = crawl_df[selected_columns]
selected_df

# Drop rows with NaN values in the specified columns
selected_df_cleaned = selected_df.dropna(subset=['url', 'title', 'meta_desc', 'body_text', 'jsonld_@type'])

# Display the cleaned DataFrame
selected_df_cleaned

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud  # Make sure to import WordCloud if not already imported

# Assuming selected_df_cleaned is the DataFrame after removing NaN rows

# Step 1: Text Cleaning and Normalization for 'meta_desc' and 'body_text'
selected_df_cleaned['meta_desc'] = selected_df_cleaned['meta_desc'].astype(str)
selected_df_cleaned['body_text'] = selected_df_cleaned['body_text'].astype(str)

selected_df_cleaned['cleaned_meta_desc'] = selected_df_cleaned['meta_desc'].apply(lambda x: x.lower())
selected_df_cleaned['cleaned_body_text'] = selected_df_cleaned['body_text'].apply(lambda x: x.lower())

# Step 2: Feature Extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(selected_df_cleaned['cleaned_body_text'])
feature_names = tfidf_vectorizer.get_feature_names_out()

# Create a DataFrame with TF-IDF features
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# Display the TF-IDF DataFrame
print(tfidf_df)

# Step 3: Data Summarization and Visualization

# Visualisation and interpretation of sample distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='jsonld_@type', data=selected_df_cleaned)
plt.title('Sample Distribution of JSON-LD Types')
plt.show()

# Visualisation and interpretation of corpus
# Assuming corpus_text is the concatenated text from the corpus
corpus_text = ' '.join(selected_df_cleaned['cleaned_body_text'])
wordcloud = WordCloud(width=800, height=400, max_words=200, background_color='white').generate(corpus_text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Corpus Text')
plt.show()

# Descriptive statistics of both the sample and the corpus
sample_stats = selected_df_cleaned.describe()
corpus_stats = tfidf_df.describe()

# Display the descriptive statistics
print("Sample Statistics:")
print(sample_stats)

print("\nCorpus Statistics:")
print(corpus_stats)

# Corpus limitations and Sampling biases
# Provide a qualitative discussion on limitations and biases observed in the corpus and sample

# Save the cleaned and preprocessed data for further analysis if needed
selected_df_cleaned.to_csv('cleaned_data.csv', index=False)

from sklearn.model_selection import train_test_split

# Assuming selected_df_cleaned is the DataFrame after the previous processing steps

# Select relevant columns for the model
features = ['title', 'cleaned_meta_desc', 'cleaned_body_text']
label = 'jsonld_@type'

# Create feature matrix (X) and target vector (y)
X = selected_df_cleaned[features]
y = selected_df_cleaned[label]

# Split the data into training and test sets with a larger test size (e.g., 30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

# Display the shapes of the training and test sets
print("Training set shape: X={}, y={}".format(X_train.shape, y_train.shape))
print("Test set shape: X={}, y={}".format(X_test.shape, y_test.shape))

#Machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming X_train, X_test, y_train, y_test from the previous code

# Step 1: Define the ML model
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
    ('clf', RandomForestClassifier(random_state=42))
])

# Step 2: Train the model
model.fit(X_train['cleaned_body_text'], y_train)

# Step 3: Predict on the test set
y_pred = model.predict(X_test['cleaned_body_text'])

# Step 4: Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display evaluation metrics
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:")
print(classification_rep)
print("\nConfusion Matrix:")
print(conf_matrix)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
