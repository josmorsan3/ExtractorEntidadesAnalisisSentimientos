
from flask import Flask, request, render_template
import pandas as pd
import spacy
from langdetect import detect
import en_core_web_sm
import es_core_news_md
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentiment_analysis_spanish import sentiment_analysis


nltk.download('vader_lexicon')

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/process', methods=["POST"])
def process():
    if request.method == 'POST':
        eleccion = request.form['taskoption']
        texto = request.form['rawtext']

        if detect(texto) == "en":
            nlp = spacy.load("en_core_web_sm")
        elif detect(texto) == "es":
            nlp = spacy.load("es_core_news_md")
        else:
            return render_template("index.html", resultados=["Por favor introduzca texto en inglés o en español"], numeroResultados="0")

        doc = nlp(texto)
        d = []
        for ent in doc.ents:
            d.append((ent.label_, ent.text))
            df = pd.DataFrame(d, columns=('named entity', 'output'))
            ORG_named_entity = df.loc[df['named entity'] == 'ORG']['output']
            PERSON_named_entity = df.loc[df['named entity']
                                         == 'PERSON']['output']
            LOC_named_entity = df.loc[df['named entity'] == 'LOC']['output']
            MISC_named_entity = df.loc[df['named entity'] == 'MISC']['output']
            MONEY_named_entity = df.loc[df['named entity']
                                        == 'MONEY']['output']

        if eleccion == 'organization':
            resultados = ORG_named_entity
            numeroResultados = len(resultados)
        elif eleccion == 'person':
            resultados = PERSON_named_entity
            numeroResultados = len(resultados)
        elif eleccion == "location":
            resultados = LOC_named_entity
            numeroResultados = len(resultados)
        elif eleccion == 'miscelanea':
            resultados = MISC_named_entity
            numeroResultados = len(resultados)
        elif eleccion == 'money':
            resultados = MONEY_named_entity
            numeroResultados = len(resultados)
        else:
            resultados = ["Debe seleccionar una opción"]
            numeroResultados = "0"
        print(detect(texto))
        if detect(texto) == "en":
            sentimiento = SentimentIntensityAnalyzer()
            analisisSentimiento = sentimiento.polarity_scores(texto)[
                "compound"]
        elif detect(texto) == "es":
            sentimiento = sentiment_analysis.SentimentAnalysisSpanish()
            analisisSentimiento = sentimiento.sentiment(texto)
    return render_template("index.html", resultados=resultados, numeroResultados=numeroResultados, analisisSentimiento=analisisSentimiento)


if __name__ == '__main__':
    app.run(debug=True)
