from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np

import joblib
from tensorflow.keras.layers import LSTM, Bidirectional



app = Flask(__name__)
CORS(app)

# 모델과 인코더 불러오기
model = tf.keras.models.load_model("model/sign_lstm_model.h5")
label_encoder = joblib.load("label_encoder.pkl")


@app.route('/ai/sequence_predict', methods=['POST'])
def sequence_predict():
    """
    # 입력 데이터 예시 (JSON)
    {
      "sequence": [
        [0.11, 0.22, ..., 0.84],  // 1번 프레임 (예: 84개 좌표)
        [0.13, 0.23, ..., 0.92],  // 2번 프레임
        ...
        [0.10, 0.24, ..., 0.93]   // N번 프레임
      ]
    }
    """
    try:
        data = request.get_json()
        if not data or "sequence" not in data:
            print("입력 데이터 없음")
            return jsonify({"error": "입력 데이터 없음"}), 400

        sequence = data["sequence"]

        # 1. **받은 데이터 출력**
        for i, frame in enumerate(sequence[:3]):  # 처음 3프레임만
            print(f"  [{i}]: {frame}")
        if len(sequence) > 3:
            print(f"  ...({len(sequence)} frames in total)")

        # numpy 변환 (형태: (N, feature_dim))
        sequence_np = np.array(sequence, dtype=np.float32)
        if sequence_np.ndim != 2:
            print(f"입력 shape 오류: {sequence_np.shape}")
            return jsonify({"error": f"입력 shape 오류: {sequence_np.shape}"}), 400

        # 모델에 맞는 차원 맞추기 (예: (1, N, feature_dim))
        input_np = sequence_np[np.newaxis, :, :]

        # 2. **예측**
        preds = model.predict(input_np)
        pred_index = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        pred_word = label_encoder.inverse_transform([pred_index])[0]

        # 3. **예측값 출력**
        print(f"예측 결과: {pred_word} (index: {pred_index}), 신뢰도: {confidence:.3f}")

        return jsonify({
            'prediction': pred_word,
            'confidence': confidence
        })

    except Exception as e:
        print(f"에러: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
