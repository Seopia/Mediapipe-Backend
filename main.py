# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/ai/keypoints', methods=['POST'])
def receive_keypoints():
    data = request.get_json()
    print("받은 좌표:", data)
    return jsonify({'status': 'received'})


if __name__ == '__main__':
    app.run(debug=True)
