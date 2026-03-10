from flask import Flask, render_template
import pandas as pd

app = Flask(__name__)

def get_results():
    data = pd.read_csv("resources.csv")
    data['Recommendation'] = data['Usage'].apply(
        lambda x: "Underutilized - reallocate" if x < 40
        else "Overutilized - reduce load" if x > 80
        else "Normal usage"
    )
    return data.to_dict(orient='records')

@app.route("/")
def home():
    results = get_results()
    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
