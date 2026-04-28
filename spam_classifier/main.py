from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib 
app= FastAPI()
history= []
#già addestrati 
model=joblib.load("model.pkl")                  #carico modello AI
vectorizer=joblib.load("vectorizer.pkl")        #carico trasformatore di testo

@app.get("/" , response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Spam classifier</title>
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
             <h2>Spam Detector</h2>
            <form action="/predict" method="post">
                <input type="text" name="text" placeholder="scrivi qui..." />
                <button type="submit">Check</button>
           <a href="/history">
                <button type="button">Storico</button>
            </a>
            <a href="/table">
                <button type="button">Tabella</button>
            </a>
           
            
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
    prob= model.predict_proba(X)[0]
    spam_prob=prob[1]               #probabilità spam


    if prediction== "spam":
        color= "red"
        sms="Questo messaggio è spam"
    else:
        color= "green"
        sms="Questo messaggio è sicuro"
    history.append({
        "text": text,
        "prediction": prediction
    })
   

    return f"""
        <html>
            <body>
                <h2>Risultato</h2>
                <p><b>Testo: </b>{text}</p>
                <p style="color:{color};">
                    <b>{sms}</b>
                </p>
                <p><b>Probabilità: </b>{spam_prob:.2f}</p>
                
                <a href="/">Torna indietro</a>
            </body>
        </html>
        """

@app.get("/history", response_class=HTMLResponse)
def show_history():
   
    history_html= "<br>".join(
       [ f"{h['text']} -> {h['prediction']}" for h in history])
    return f"""
        <html>
            <body style="font-family:Arial; text-align:center">
                <hr>
                <h3>Storico ultimi messaggi </h3>
                <p>{history_html} </p>
                <br>
                <a href="/">Home</a>
            </body> 
        </html>
    """

@app.get("/table", response_class=HTMLResponse)
def table():
    rows= ""
    for h in history: 
        rows +=f"""
        <tr>
            <td>{h['text']}</td>
            <td>{h['prediction']}</td>
        </tr>   
    """
    return f"""
        <html>
            <head>
                <style>
                    body{{
                        font-family:Arial;
                        text-align:center;
                    }}
                    table {{
                    margin: auto;
                    border-collapse: collapse;
                    width: 60%;
                    }}

                    th, td {{
                        border: 1px solid black;
                        padding: 10px;
                    }}

                    th {{
                        background-color: #ccb5b5;
                    }}
                </style>
            </head>

            <body>
                <table>
                    <tr>
                        <th>Testo</th>
                        <th>Predizione</th>
                    </tr> 
                    {rows}
                </table>
                <br>
                <a href="/">Home</a>
            </body>
        </html>

    """