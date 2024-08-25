from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)

# Load and preprocess the dataset once
dataset_url = "https://raw.githubusercontent.com/apogiatzis/breast-cancer-azure-ml-notebook/master/breast-cancer-data.csv"
df = pd.read_csv(dataset_url)
df = df.drop(columns=['id', 'Unnamed: 32'], axis=1)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean']
X = df[features]
y = df.diagnosis

# Train the model once
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=500, n_jobs=-1)
model.fit(X_train, y_train)
joblib.dump(model, 'random_forest_model.pkl')

@app.route("/", methods=['GET', 'POST'])
def cancerPrediction():
    if request.method == 'POST':
        inputQuery1 = request.form['query1']
        inputQuery2 = request.form['query2']
        inputQuery3 = request.form['query3']
        inputQuery4 = request.form['query4']
        inputQuery5 = request.form['query5']

        data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5]]
        new_df = pd.DataFrame(data, columns=features)
        
        # Load the trained model
        model = joblib.load('random_forest_model.pkl')
        single = model.predict(new_df)[0]
        proba = model.predict_proba(new_df)[:, 1][0]

        if single == 1:
            output1 = "The patient is diagnosed with Breast cancer"
            output2 = "Confidence: {:.2f}%".format(proba * 100)
        else:
            output1 = "The patient is not diagnosed with Breast cancer"
            output2 = ""

        return render_template('home.html', output1=output1, output2=output2,
                               query1=inputQuery1, query2=inputQuery2, query3=inputQuery3,
                               query4=inputQuery4, query5=inputQuery5)

    return render_template('home.html', query1="", query2="", query3="", query4="", query5="")

if __name__ == '__main__':
    app.run(debug=True)
