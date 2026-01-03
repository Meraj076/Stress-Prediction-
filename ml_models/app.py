from flask import Flask, request, jsonify
import joblib, os, json
import numpy as np
import pandas as pd

app = Flask(__name__)

# --- Model Loading ---
MODEL_UNDER20 = None
MODEL_20PLUS = None
LABEL_ENCODER = None

# Models are expected to be in the same directory
under_candidates = ['model_under20.pkl', 'stress_prediction_model_under20.pkl', 'stress_model_under20.pkl']
plus_candidates = ['model_20plus.pkl', 'stress_model_20plus.pkl']

for fn in under_candidates:
    if os.path.exists(fn):
        try:
            MODEL_UNDER20 = joblib.load(fn)
            print('Loaded under-20 model from', fn)
            break
        except Exception as e:
            print('Failed loading', fn, e)

for fn in plus_candidates:
    if os.path.exists(fn):
        try:
            MODEL_20PLUS = joblib.load(fn)
            print('Loaded 20+ model from', fn)
            break
        except Exception as e:
            print('Failed loading', fn, e)

if os.path.exists('label_encoder_y.pkl'):
    try:
        LABEL_ENCODER = joblib.load('label_encoder_y.pkl')
        print('Loaded label encoder')
    except Exception as e:
        print('Failed label encoder', e)

# --- Helper Functions ---
def prepare_X(answers):
    arr = []
    for a in answers:
        if isinstance(a, str):
            s = a.strip()
            try:
                v = float(s)
                arr.append(v)
            except:
                # try to handle '0'/'1' strings or formatted numbers
                try:
                    v = float(s.replace(',', ''))
                    arr.append(v)
                except:
                    arr.append(s)
        elif isinstance(a, (int, float)):
            arr.append(float(a))
        else:
            arr.append(a)
    
    # helper check
    all_num = all(isinstance(x, (int, float)) for x in arr)
    if all_num:
        return np.array(arr).reshape(1, -1)
    else:
        return pd.DataFrame([arr])

# --- Routes ---

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint for stress prediction.
    Expects JSON input:
    {
        "age": 25,
        "answers": [value1, value2, ...],        <-- List of feature values
        "features": { "col1": val1, ... }        <-- OR Dictionary of features (optional alternative)
    }
    """
    try:
        data = request.get_json(force=True)
    except:
        return jsonify({'error':'invalid json'}), 400

    if not data:
        return jsonify({'error':'no json payload'}), 400

    age = data.get('age', None)
    answers = data.get('answers', None)
    features = data.get('features', None)

    if age is None:
        return jsonify({'error':'age is required'}), 400
    
    try:
        age_int = int(age)
    except:
        return jsonify({'error':'age must be a number'}), 400

    X = None
    
    # 1. Try features dict
    if isinstance(features, dict) and features:
        try:
            X = pd.DataFrame([features])
        except Exception as e:
            X = None

    # 2. Try answers list
    if X is None:
        if not answers:
            return jsonify({'error':'answers or features required'}), 400
        try:
            X = prepare_X(answers)
        except Exception as e:
            return jsonify({'error':'could not prepare features: '+str(e)}), 400

    # Select Model
    model = MODEL_UNDER20 if age_int < 20 else MODEL_20PLUS
    group = 'under20' if age_int < 20 else '20plus'

    if model is None:
        return jsonify({'error': f'No model loaded for age group {group} on server'}), 500

    # Predict
    try:
        pred = model.predict(X)
    except Exception as e:
        return jsonify({'error':'model predict error: '+str(e)}), 500

    # Decode Label/Score
    label = None
    score = None
    try:
        val = pred[0] if hasattr(pred, '__len__') else pred
        
        if LABEL_ENCODER is not None:
            try:
                label = LABEL_ENCODER.inverse_transform([val])[0]
            except:
                # Fallback if inverse transform fails
                try: 
                    score = float(val)
                except: 
                    label = str(val)
        else:
            try:
                score = float(val)
            except:
                label = str(val)
    except Exception as e:
        label = str(pred)

    return jsonify({
        'age': age_int,
        'group': group,
        'score': score,
        'label': label
    })

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'Stress Prediction API is running'})

if __name__ == '__main__':
    print('Models loaded: under20', MODEL_UNDER20 is not None, '20plus', MODEL_20PLUS is not None)
    app.run(host='0.0.0.0', port=5000, debug=True)
