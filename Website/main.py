import trained
from flask import Flask, render_template, request, redirect
app = Flask(__name__, template_folder='template', static_folder='assets')


@app.route("/train", methods=["GET", "POST"])
def train():
    trained.train()
    return redirect('/')


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        predicted = trained.predict(message)
        return render_template('result.html', prediction=predicted, message=message)

    # jika GET
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
