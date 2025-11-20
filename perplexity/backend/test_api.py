import requests

r = requests.post("http://127.0.0.1:5000/api/chat", json={"symptoms": "hi"})
print("chat:", r.status_code, r.text)

r = requests.post("http://127.0.0.1:5000/api/predict_disease", json={"symptoms": "hi"})
print("disease:", r.status_code, r.text)

r = requests.post("http://127.0.0.1:5000/api/recommend_medicines", json={"symptoms": "hi", "topk": 3})
print("meds:", r.status_code, r.text)
