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
#from sklearn.naive_bayes import MultinomialNB

vectorizer= TfidfVectorizer( 
    ngram_range= (1,2),         #parole singole e coppie 
    stop_words=None             #quali parole ignorare durante la creazione di un modello di testo
)
X= vectorizer.fit_transform(df["testo"])
y=df["label"]

from sklearn.linear_model import LogisticRegression
model= LogisticRegression()
model.fit(X,y)

testi = [
    "gratis ciao riunione progetto",
    "ciao ci vediamo dopo",
    "ciao GRATIS gratis",
    "gratis gratis progetto",
    "offerta limitata clicca"
]

X_test = vectorizer.transform(testi)
predizioni = model.predict(X_test)

for t,p in zip(testi, predizioni):
    print(f"{t}->{p}")

messaggio= input("scrivi un messaggio: ")
C_test= vectorizer.transform([messaggio])
print(model.predict(C_test))

#print(vectorizer.get_feature_names_out())