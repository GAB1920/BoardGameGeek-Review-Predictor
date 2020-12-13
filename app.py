# importing necessary libraries
import pickle
from flask import Flask, render_template, request, url_for

# importing pickle files
cv = pickle.load(open('tfidf.pkl', 'rb'))
clf = pickle.load(open('best_rf.pkl', 'rb'))

# initiate flask
app = Flask(__name__)

# create app route for home page
@app.route('/')
def home():
    return render_template('BoardGameGeek Reviews Prediction.html')

# create app route for prediction
@app.route('/Predict', methods = ['POST'])
def Predict():

    if request.method == 'POST':
        review_entered = request.form['message']
        review = [review_entered]
        vect = cv.transform(review).toarray()
        pred = clf.predict(vect)

    return render_template('BoardGameGeek Reviews Prediction.html', prediction = pred)

if __name__ == '__main__':
    app.run(debug = False, port = 80)
