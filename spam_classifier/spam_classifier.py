import pandas as pd
data= {
    "testo": [
        "vinci un iphone gratis ora",
        "ciao come stai?",
        "offerta limitata affrettati a comprare",
        "ci vediamo domami",
        "vuoi guadagni facili? clicca qui",
        "ti mando il file che volevi"
    ],
    "label":[
        "spam",
        "ham",
        "spam",
        "ham",
        "spam",
        "ham"
    ]
} 
df=pd.DataFrame(data)
print(df)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

vectorizer= TfidfVectorizer()
X= vectorizer.fit_transform(df["testo"])
y=df["label"]

model=MultinomialNB()
model.fit(X,y)

testi = [
    "guadagna soldi subito gratis",
    "ciao ci vediamo dopo"
]

X_test = vectorizer.transform(testi)

predizioni = model.predict(X_test)

print(predizioni)