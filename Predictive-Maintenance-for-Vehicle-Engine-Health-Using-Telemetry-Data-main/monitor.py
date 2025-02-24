from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/latest-prediction', methods=['GET'])
def latest_prediction():
    try:
        df = pd.read_csv("predictions_log.csv")
        latest = df.iloc[-1].to_dict()
        return jsonify(latest)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
