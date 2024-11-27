import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Sample data:
documents = [
    "The soccer team won their game yesterday with a 3-1 victory.",
    "Basketball playoffs are heating up, with the best teams making the finals.",
    "The presidential candidate held a rally in the city today.",
    "The debate last night was intense and drew millions of viewers.",
    "The Olympics will be held in Tokyo next year.",
    "The senator gave an impassioned speech about healthcare reforms.",
    "The football game was canceled due to bad weather.",
    "The president addressed the nation in a press conference today.",
    "The coach discussed the team's strategy for the upcoming match.",
    "The government is planning new policies to boost the economy."
]

labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]

vectorizer = CountVectorizer(stop_words='english')

X = vectorizer.fit_transform(documents)

X_train,X_test,y_train,y_test =train_test_split(X,labels,test_size=0.3,random_state=42)

classifier = MultinomialNB()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

sample_text = ["The president is meeting with the international leaders today."]
sample_vector = vectorizer.transform(sample_text)
sample_pred = classifier.predict(sample_vector)

# Output the predicted class
if sample_pred == 1:
    print("The text is classified as: Sports")
else:
    print("The text is classified as: Politics")
