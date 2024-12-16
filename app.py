from flask import Flask, request, render_template_string, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load('c:/Lookas/STIKOM/angga-ai/dev/energy_consumption_model.pkl')

# HTML template for input form
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predict Energy Consumption</title>
</head>
<body>
    <h1>Energy Consumption Prediction</h1>
    <form method="POST" action="/predict">
        <label for="appliance">Appliance:</label>
        <input type="text" id="appliance" name="appliance" required><br><br>
        <label for="room">Room:</label>
        <input type="text" id="room" name="room" required><br><br>
        <label for="status">Status:</label>
        <input type="text" id="status" name="status" required><br><br>
        <label for="hour">Hour:</label>
        <input type="number" id="hour" name="hour" required><br><br>
        <label for="day">Day:</label>
        <input type="number" id="day" name="day" required><br><br>
        <button type="submit">Predict</button>
    </form>
</body>
</html>
'''

@app.route('/', methods=['GET'])
def home():
    # Render form
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Create a DataFrame from the form data
        input_data = pd.DataFrame({
            'Appliance': [request.form['appliance']],
            'Room': [request.form['room']],
            'Status': [request.form['status']],
            'Hour': [int(request.form['hour'])],
            'Day': [int(request.form['day'])]
        })

        prediction = model.predict(input_data)

        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
