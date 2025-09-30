import pandas as pd
import requests
import json

X = pd.read_parquet("data/processed/test_modeling.parquet")
row = X.sample(1, random_state=42).iloc[0]
payload = {"features": row.to_dict()}

print("Sending request to server...")
print(f"Payload keys: {list(payload['features'].keys())[:10]}...")  # Show first 10 keys

resp = requests.post("http://localhost:8000/predict", json=payload, timeout=10)
print(f"Status code: {resp.status_code}")
print(f"Response text: {resp.text}")

if resp.status_code == 200:
    try:
        print("Response JSON:")
        print(json.dumps(resp.json(), indent=2))
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
else:
    print(f"Error response: {resp.text}")