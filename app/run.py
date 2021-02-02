import json
import plotly
import pandas as pd
import string


from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

from nltk.corpus import stopwords

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine("sqlite:///data/DisasterResponse.db")
df = pd.read_sql_table('disaster_messages_tbl', engine)

# load model
model = joblib.load("./models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Custom Viz 1
    viz_1 = pd.DataFrame(df.iloc[:, 4:].mean(), columns=['Val']).sort_values('Val', ascending=False).iloc[:5, :]
    top_5_vals = viz_1['Val']
    top_5_cols = list(viz_1.index)

    # Custom Viz 2 Setup
    popular_words = {}

    stop_words = stopwords.words('english')
    punct = [p for p in string.punctuation]

    for m in df['message']:
        for word in m.split():
            new_word = word.lower()
            if new_word not in stop_words and new_word not in punct:
                if new_word in popular_words:
                    popular_words[new_word] += 1
                else:
                    popular_words[new_word] = 1

    viz_2 = pd.DataFrame.from_dict(popular_words, orient='index')
    viz_2.columns = ['Val']
    top_5_words_vals = viz_2.sort_values('Val', ascending=False)[:5]['Val']
    top_5_words = list(viz_2.sort_values('Val', ascending=False)[:5].index)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
        , {
            'data': [
                Bar(
                    x=top_5_cols,
                    y=top_5_vals
                )
            ],

            'layout': {
                'title': 'Top 5 Message Types',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Message Type"
                }
            }
        }
        , {
            'data': [
                Bar(
                    x=top_5_words,
                    y=top_5_words_vals
                )
            ],

            'layout': {
                'title': 'Top 5 Most Used Words',
                'yaxis': {
                    'title': "Total Count from all Messages"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
