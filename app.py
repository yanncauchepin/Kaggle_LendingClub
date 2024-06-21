import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

with open('lending_club_mlp_binary_classifier.pkl', 'rb') as f:
    model = pickle.load(f)
    
map_output = {
    0: 'Fully Paid',
    1: 'Charged Off'
    }
    
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction, _ = model.predict([data['features']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)