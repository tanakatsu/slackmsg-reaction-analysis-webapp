from flask import Flask, request
import MeCab
import json

from infer import Predictor

app = Flask(__name__)


def parse_text(text):
    tagger = MeCab.Tagger("-Owakati")
    result = tagger.parse(text)
    print('result=' + result)
    return result.split(' ')


@app.route('/')
def hello_world():
    return "Hello World!"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        text = request.args.get('text')
    if request.method == 'POST':
        text = request.form['text']

    print('text=' + text.encode('utf-8'))
    if text:
        words = parse_text(text.encode('utf-8'))
    else:
        return json.dumps({'error': 'no text'})
    if words[-1] == '\n':
        words = words[:-1]
    # print(words)

    unit = 300
    seq_len = 30
    predictor = Predictor(unit, seq_len)
    result = predictor.eval(words)
    print(result)

    return json.dumps(result)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
