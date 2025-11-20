const input = document.getElementById("symptomInput");
const output = document.getElementById("chatOutput");
const sendBtn = document.getElementById("sendBtn");

sendBtn.addEventListener("click", () => {
  const symptoms = input.value.trim();
  if (!symptoms) {
    alert("Please enter symptoms");
    return;
  }

  fetch("http://127.0.0.1:5000/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symptoms })
  })
    .then(res => res.json())
    .then(data => {
      if (!data.success && data.error) {
        output.textContent = "Error: " + data.error;
      } else {
        output.textContent = data.message; // chatbot answer text
      }
    })
    .catch(err => {
      console.error(err);
      output.textContent = "Network error. Is backend running?";
    });
});

function callDiseaseApi(symptoms) {
  return fetch("http://127.0.0.1:5000/api/predict_disease", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symptoms })
  }).then(res => res.json());
}

function callMedicineApi(symptoms) {
  return fetch("http://127.0.0.1:5000/api/recommend_medicines", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symptoms, topk: 3 })
  }).then(res => res.json());
}
