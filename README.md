# SMS-Spam-Detector
# Overview
This project focuses on building a Python script for SMS spam detection using Natural Language Processing (NLP). The script utilizes machine learning techniques to classify SMS messages as either spam or ham (non-spam). The primary goal is to enhance user experience by filtering out unwanted spam messages.

# Key Features
Data Preprocessing: The script preprocesses SMS data by cleaning and tokenizing the text, making it suitable for machine learning analysis.
NLP Techniques: Natural Language Processing techniques, including tokenization and text processing, are employed to extract meaningful features from the SMS messages.
Machine Learning Model: A machine learning model, trained on labeled SMS data, is applied to classify messages as spam or ham.

# Usage
# Installation:
Clone the repository: git clone (https://github.com/jayhadiyal03/SMS-Spam-Detector.git)
Install dependencies: pip install pandas nltk
Download NLTK resources: python -m nltk.downloader punkt

# Run the Script:
Execute the script: python "SMS_spam.py"

# Dataset Format
Ensure your SMS dataset follows the CSV format:

--> Example : 
Type,Message
ham,Go until jurong point, crazy.. Available only ...
spam,Free entry in 2 a wkly comp to win FA Cup fina...
ham,Ok lar... Joking wif u oni...
...

# Results:
The script outputs predictions, allowing users to identify and filter spam messages effectively.





