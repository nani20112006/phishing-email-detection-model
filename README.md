# phishing-email-detection-model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample Dataset
emails = [
    "Your bank account is locked. Click here to verify.",
    "Congratulations! You won a free iPhone. Claim now.",
    "Meeting scheduled for tomorrow at 10 AM.",
    "Project report has been submitted successfully.",
    "Update your password immediately using this link.",
    "Lunch meeting at 1 PM today.",
    "Your package delivery is pending. Click here.",
    "Team meeting agenda attached."
]

labels = [
    "Phishing",
    "Phishing",
    "Safe",
    "Safe",
    "Phishing",
    "Safe",
    "Phishing",
    "Safe"
]

# Convert text into numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42
)

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred) * 100)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Test Email
new_email = ["Verify your account immediately by clicking this link"]
new_email_vector = vectorizer.transform(new_email)

prediction = model.predict(new_email_vector)

print("\nEmail:", new_email[0])
print("Prediction:", prediction[0])
