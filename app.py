from flask import Flask, escape, render_template, request
from GoogleNews import GoogleNews
import pickle

googleNews = GoogleNews()
googleNews.enableException(True)
googleNews = GoogleNews(lang='en')
googleNews = GoogleNews(encode='utf-8')

vector = pickle.load(open("tfidfvect2.pkl", 'rb'))
model = pickle.load(open("PassiveAggressiveClassifier.pkl", 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        news = str(request.form['news'].replace(" ", "%"))
        print(news)
        search_url = "https://news.google.com/search?q=" + news
        # encoded_news = news.encode()
        # googleNews.get_news(news)

        predict = model.predict(vector.transform([news]))[0]
        print(predict)

        return render_template("prediction.html", prediction_text="News headline is -> {}".format(predict),
                               search_url="The link for Google News source is -> {}".format(search_url), news=news)

    else:
        return render_template("prediction.html")


if __name__ == '__main__':
    app.run(debug=True)
