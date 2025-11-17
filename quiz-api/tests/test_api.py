import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_generate_quiz():
    response = client.post("/generate-quiz", json={
        "user_requirement": "aptitude test",
        "complexity": ["Easy", "Medium"],
        "question": {
            "ShortText": 2,
            "MultipleChoiceSingleAnswer": 1,
            "MultipleChoiceMultipleAnswer": 3,
            "TrueFalse": 4,
            "Coding": 0
        }
    })
    assert response.status_code == 200
    assert "type" in response.json()[0]
    assert "questions" in response.json()[0]

def test_generate_quiz_missing_fields():
    response = client.post("/generate-quiz", json={
        "user_requirement": "aptitude test",
        "complexity": ["Easy", "Medium"]
    })
    assert response.status_code == 422  # Unprocessable Entity