import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib 
data= {
     "testo": [
        "vinci un iphone gratis ora",
        "ciao come stai?",
        "offerta limitata affrettati a comprare",
        "ci vediamo domami",
        "vuoi guadagni facili? clicca qui"
        "ti mando il file che volevi"
    ],
    "label":[
        "spam","ham","spam","ham", "spam",  "ham"
    ]
}
df=pd.DataFrame(data)

#vectorizer
vectorizer= TfidfVectorizer()
X= vectorizer.fit_transform(df["testo"])
y=df["label"]

#model
model= LogisticRegression()
model.fit(X,y)

#salva modello
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl") 

print("Modello salvato")