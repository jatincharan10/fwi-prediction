import pickle
from flask import Flask, request, render_template  # type: ignore
import numpy as np  # pyright: ignore[reportMissingImports]

# Initialize Flask app
application = Flask(__name__)
app = application

# Load Ridge Regressor model and StandardScaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/", methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        try:
            # Get form data
            Temperature = float(request.form.get("Temperature"))
            RH = float(request.form.get("RH"))
            Ws = float(request.form.get("Ws"))
            Rain = float(request.form.get("Rain"))
            FFMC = float(request.form.get("FFMC"))
            DMC = float(request.form.get("DMC"))
            ISI = float(request.form.get("ISI"))
            Classes = float(request.form.get("Classes"))
            Region = float(request.form.get("Region"))

            # Scale input
            new_data_scaled = standard_scaler.transform(
                [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
            )

            # Predict
            prediction = ridge_model.predict(new_data_scaled)
            results = round(prediction[0], 2)

        except Exception as e:
            results = f"Error: {str(e)}"

    return render_template('index.html', results=results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
