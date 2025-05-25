from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

# 모델과 인코더 불러오기
model = tf.keras.models.load_model("model/sign_dnn_model.h5")
label_encoder = joblib.load("label_encoder.pkl")

@app.route('/ai/sequence_predict', methods=['POST'])
def sequence_predict():
    try:
        data = request.get_json()
        if not data or "sequence" not in data:
            print("시퀀스 데이터가 없어요.")
            return jsonify({"error": "시퀀스 데이터가 없어요."}), 400

        sequence = data["sequence"]
        if len(sequence) == 0:
            print("시퀀스가 비어있어요.")
            return jsonify({"error": "시퀀스가 비어있어요."}), 400

        # 각 프레임 처리
        processed_frames = []
        for frame in sequence:
            pose = frame.get("pose_keypoints_2d", [])
            left = frame.get("hand_left_keypoints_2d", [])
            right = frame.get("hand_right_keypoints_2d", [])
            face = frame.get("face_keypoints_2d", [])

            if len(pose) != 50 or len(left) != 42 or len(right) != 42 or len(face) != 140:
                print("프레임 차원이 맞지 않아요.")
                return jsonify({"error": "프레임 차원이 맞지 않아요."}), 400

            flat = pose + left + right + face  # 274차원
            processed_frames.append(flat)

        sequence_np = np.array(processed_frames)  # (N, 274)
        sequence_np = sequence_np.reshape(1, *sequence_np.shape)  # (1, N, 274)

        preds = model.predict(sequence_np)
        pred_index = np.argmax(preds, axis=1)[0]
        confidence = np.max(preds)

        if confidence >= 0.8:
            word = label_encoder.inverse_transform([pred_index])[0]
            return jsonify({'prediction': word, 'confidence': float(confidence)})
        else:
            return jsonify({'prediction': None, 'confidence': float(confidence)})

    except Exception as e:
        print(e)
        print("알 수 없는 에러")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
