# Importing all necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Loading the dataset 
csv_file_path = "C:/Users/Jay/UpGrade - Data Science Course/spam.csv"
df = pd.read_csv(csv_file_path, encoding='latin-1')
print(df.head())

# Data exploration and preprocessing
# Here 'Message' tag contains the SMS messages and 'Type' tag contains the spam or ham labels
X = df['Message']
y = df['Type']
print(X.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text preprocessing and modeling pipeline
pipeline = make_pipeline(
    TfidfVectorizer(stop_words=stopwords.words('english'), tokenizer=nltk.word_tokenize, max_features=5000),
    SelectFromModel(LinearSVC(penalty="l1", dual=False, random_state=42)),
    StandardScaler(with_mean=False),
    SVC(kernel='linear')
)

# Training of the model
pipeline.fit(X_train, y_train)

# Prediction of the model
y_pred = pipeline.predict(X_test)

# Evaluation of the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('\nConfusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(classification_rep)

# Testing the model with a new SMS message #

new_sms = ["Congratulations! You've won a free vacation. Claim your prize now!"]
predicted_label = pipeline.predict(new_sms)
print(f'Predicted Label for the new SMS: {predicted_label[0]}')
