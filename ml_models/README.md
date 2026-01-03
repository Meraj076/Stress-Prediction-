# Stress Prediction Model API

This directory contains the standalone microservice for the Stress Prediction Machine Learning model. It is designed to be hosted independently (e.g., on Render) and accessed via a REST API.

## 🧠 Project Overview

This API serves a Machine Learning system that predicts stress levels based on user responses. It intelligently selects between two specialized models based on the user's age:

- **Under 20 Model**: Optimized for students and adolescents.
- **20 Plus Model**: Optimized for adults and working professionals.

## 📂 File Structure

```plaintext
ml_models/
├── app.py                  # Main Flask API application (Entry Point)
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── model_under20.pkl       # Trained model for age < 20
├── model_20plus.pkl        # Trained model for age >= 20
├── label_encoder_y.pkl     # Encoder for interpreting model output
└── [other .pkl files]      # Supporting model pipelines
```

## 🚀 API Endpoints

### 1. Health Check

- **URL**: `/`
- **Method**: `GET`
- **Response**: Returns status of the API.

### 2. Predict Stress

- **URL**: `/api/predict`
- **Method**: `POST`
- **Content-Type**: `application/json`
- **Body Parameters**:

| Parameter  | Type   | Description                                                                                                 |
| :--------- | :----- | :---------------------------------------------------------------------------------------------------------- |
| `age`      | `int`  | **Required**. The age of the user. Determines which model to use.                                           |
| `answers`  | `list` | **Required**. A list of feature values (answers to the questionnaire) in the exact order the model expects. |
| `features` | `dict` | (Optional) Key-value pairs of features if not using `answers`.                                              |

#### Example Request

```json
{
  "age": 25,
  "answers": [
    "Male",
    25,
    "Yes",
    "No",
    3.5,
    ...
  ]
}
```

#### Example Response

```json
{
  "age": 25,
  "group": "20plus",
  "score": 1,
  "label": "High Stress"
}
```

## 🛠️ Local Setup & Running

1.  **Navigate to the directory**:

    ```bash
    cd ml_models
    ```

2.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Server**:
    ```bash
    python app.py
    ```
    The server will start at `http://localhost:5000`.

## ☁️ Deployment (Render)

1.  **Root Directory**: Set to `ml_models`.
2.  **Build Command**: `pip install -r requirements.txt`
3.  **Start Command**: `gunicorn app:app`
