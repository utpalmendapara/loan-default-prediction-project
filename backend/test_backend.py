import sys
from fastapi.testclient import TestClient
from backend.main import app

def test_api():
    client = TestClient(app)

    # 1. Health check
    res_health = client.get("/api/health")
    assert res_health.status_code == 200, f"Health check failed: {res_health.text}"
    print("[PASS] GET /api/health ->", res_health.json())

    # 2. Models list
    res_models = client.get("/api/models")
    assert res_models.status_code == 200, f"Models check failed: {res_models.text}"
    models_data = res_models.json()
    expected_model_ids = {
        "decision_tree", "gaussian_nb", "knn", "logistic_regression", "svc"
    }
    registered_model_ids = {model["id"] for model in models_data["models"]}
    assert expected_model_ids.issubset(registered_model_ids)
    print("[PASS] GET /api/models ->", models_data)

    # 3. Predict Low Risk
    payload_low_risk = {
        "Age": 56,
        "Income": 85994,
        "LoanAmount": 50587,
        "CreditScore": 750,
        "MonthsEmployed": 80,
        "NumCreditLines": 4,
        "InterestRate": 5.23,
        "LoanTerm": 36,
        "DTIRatio": 0.24,
        "Education": "Bachelor's",
        "EmploymentType": "Full-time",
        "MaritalStatus": "Married",
        "HasMortgage": "Yes",
        "HasDependents": "Yes",
        "LoanPurpose": "Home",
        "HasCoSigner": "Yes",
        "model_id": "decision_tree"
    }
    res_pred = client.post("/api/predict", json=payload_low_risk)
    assert res_pred.status_code == 200, f"Prediction failed: {res_pred.text}"
    pred_data = res_pred.json()
    assert pred_data["prediction"] == 0, f"Expected 0, got {pred_data['prediction']}"
    assert pred_data["is_default"] is False
    print("[PASS] POST /api/predict (Low Risk) ->", pred_data)

    # 4. Predict High Risk
    payload_high_risk = {
        "Age": 22,
        "Income": 15000,
        "LoanAmount": 140000,
        "CreditScore": 350,
        "MonthsEmployed": 2,
        "NumCreditLines": 9,
        "InterestRate": 24.5,
        "LoanTerm": 60,
        "DTIRatio": 0.88,
        "Education": "High School",
        "EmploymentType": "Unemployed",
        "MaritalStatus": "Single",
        "HasMortgage": "No",
        "HasDependents": "Yes",
        "LoanPurpose": "Other",
        "HasCoSigner": "No",
        "model_id": "decision_tree"
    }
    res_pred_high = client.post("/api/predict", json=payload_high_risk)
    assert res_pred_high.status_code == 200, f"High risk prediction failed: {res_pred_high.text}"
    pred_high_data = res_pred_high.json()
    print("[PASS] POST /api/predict (High Risk) ->", pred_high_data)

    # 5. Verify each additional dropdown model can make a prediction.
    for model_id in expected_model_ids - {"decision_tree"}:
        payload = {**payload_low_risk, "model_id": model_id}
        response = client.post("/api/predict", json=payload)
        assert response.status_code == 200, f"{model_id} failed: {response.text}"
        assert response.json()["model_id"] == model_id
        print(f"[PASS] POST /api/predict ({model_id}) ->", response.json())

    print("\nALL BACKEND API TESTS PASSED SUCCESSFULLY! :)")

if __name__ == "__main__":
    test_api()
