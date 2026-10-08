import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# FAKE NEWS DETECTION USING TEXT ANALYSIS
# ============================================================

print("=" * 60)
print("       FAKE NEWS DETECTION USING TEXT ANALYSIS")
print("=" * 60)

# ------------------------------------------------------------
# 1. Sample Dataset
# ------------------------------------------------------------

data = {
    "text": [
        "Government announces new scholarship program for college students",
        "Scientists confirm that drinking water improves human health",
        "Breaking news celebrity claims aliens have landed in India",
        "Social media post says the moon will disappear tomorrow",
        "Government launches new digital education program",
        "Researchers publish new study on climate change",
        "Famous actor says eating only chocolate can make people live forever",
        "Internet users claim that smartphones can read human thoughts",
        "New technology introduced to improve online education",
        "University announces free skill development courses",
        "Scientists discover a new method for treating diseases",
        "Experts warn about increasing global temperatures",
        "Viral message claims that humans will stop aging next year",
        "Fake report says all banks will close for one month",
        "Government introduces new employment opportunities",
        "Education department announces new examination guidelines",
        "Scientists develop an innovative renewable energy system",
        "News report claims that drinking coffee gives supernatural powers",
        "Researchers develop a new water purification technology",
        "Government announces financial support for small businesses"
    ],
    
    "label": [
        "REAL",
        "REAL",
        "FAKE",
        "FAKE",
        "REAL",
        "REAL",
        "FAKE",
        "FAKE",
        "REAL",
        "REAL",
        "REAL",
        "REAL",
        "FAKE",
        "FAKE",
        "REAL",
        "REAL",
        "REAL",
        "FAKE",
        "REAL",
        "REAL"
    ]
}

df = pd.DataFrame(data)

print("\nDataset loaded successfully!")
print("Total News Articles:", len(df))

# ------------------------------------------------------------
# 2. Display Dataset
# ------------------------------------------------------------

print("\nSample Dataset:")
print(df.head())

# ------------------------------------------------------------
# 3. Split Dataset
# ------------------------------------------------------------

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ------------------------------------------------------------
# 4. Convert Text into Numerical Features
# ------------------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ------------------------------------------------------------
# 5. Train Machine Learning Model
# ------------------------------------------------------------

model = LogisticRegression()

model.fit(X_train_tfidf, y_train)

print("\nMachine Learning model trained successfully!")

# ------------------------------------------------------------
# 6. Test Model
# ------------------------------------------------------------

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ------------------------------------------------------------
# 7. Predict New News
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("              NEWS PREDICTION")
print("=" * 60)

while True:

    news = input("\nEnter a news article (or type 'exit' to stop): ")

    if news.lower() == "exit":
        print("\nThank you for using Fake News Detection System!")
        break

    if news.strip() == "":
        print("Please enter some news text.")
        continue

    # Convert input text to TF-IDF
    news_tfidf = vectorizer.transform([news])

    # Prediction
    prediction = model.predict(news_tfidf)[0]

    # Probability
    probability = model.predict_proba(news_tfidf)[0]
    confidence = max(probability) * 100

    print("\n----------------------------------------")

    if prediction == "FAKE":
        print("Prediction : FAKE NEWS")
    else:
        print("Prediction : REAL NEWS")

    print("Confidence :", round(confidence, 2), "%")
    print("----------------------------------------")