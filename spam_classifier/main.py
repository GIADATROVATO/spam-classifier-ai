from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib 
app= FastAPI()

#già addestrati 
model=joblib.load("model.pkl")                  #carico modello AI
vectorizer=joblib.load("vectorizer.pkl")        #carico trasformatore di testo

@app.get("/" , response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>"Spam classifier"</title>
            <style>
                body{
                    font-family: Arial;
                    text-align: center;
                    margin-top: 50px;
                }
                input{
                    padding: 10px;
                    width: 300px;
                }
                button{
                    padding: 10px;
                    background-color: blue;
                    color: white;
                    border-radius: 8px;
                    border:none;
                }
            </style>
        </head>
        <body>
             <h2>"Spam detector"</h2>
            <form action="/predict" method="post">
                <input type="text" name="text" placeholder="scrivi qui" />
                <button type="submit" >Check</button>
            </form>
        </body>
    </html>
    """

#ricevi dati dall'esterno
@app.post("/predict", response_class=HTMLResponse)
def predict( text: str= Form(...)):
    if text.strip()== "":
        return """
            <h3> Scrivi qualcosa</h3>
            <a href="/">Torna indietro</a> 
        """
    X= vectorizer.transform([text])
    prediction= model.predict(X)[0]
    if prediction== "spam":
        color= "red"
        sms="Questo messaggio è spam"
    else:
        color= "green"
        sms="Questo messaggio è sicuro"
    return f"""
        <html>
            <body>
                <h2>Risultato</h2>
                <p><b>Testo: </b>{text}</p>
                <p style="color:{color};">
                    <b>{sms}</b>
                </p>
                <a href="/">Torna indietro</a>
            </body>
        </html>
        """