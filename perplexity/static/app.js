// DOM references
const input        = document.getElementById("symptomInput");
const output       = document.getElementById("chatOutput");
const sendBtn      = document.getElementById("sendBtn");
const chatContainer= document.getElementById("chatContainer");
const themeToggle  = document.getElementById("themeToggle");
const themeLabel   = document.getElementById("themeLabel");
const refreshChat  = document.getElementById("refreshChat");

// Auto‑resize textarea
function autoResize() {
  input.style.height = "auto";
  input.style.height = input.scrollHeight + "px";
}
input.addEventListener("input", autoResize);

function getCurrentTime() {
  const now = new Date();
  return now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function addUserBubble(text) {
  const time = getCurrentTime();
  const wrapper = document.createElement("div");
  wrapper.className = "flex items-end gap-3 justify-end";
  wrapper.innerHTML = `
    <div class="flex flex-1 flex-col gap-1 items-end">
      <p class="text-base leading-relaxed flex max-w-lg rounded-xl rounded-br-none px-4 py-3 bg-primary text-white">
        ${text}
      </p>
      <p class="text-text-secondary-light dark:text-text-secondary-dark text-xs px-1">${time}</p>
    </div>
    <div class="bg-center bg-no-repeat aspect-square bg-cover rounded-full w-10 shrink-0"
         style="background-image:url('https://lh3.googleusercontent.com/a-/dummyUser');"></div>
  `;
  chatContainer.appendChild(wrapper);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addBotBubbleFromTextBlock(textBlock) {
  const safeHtml = textBlock
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\n/g, "<br>");

  const time = getCurrentTime();
  const wrapper = document.createElement("div");
  wrapper.className = "flex items-end gap-3 max-w-3xl";
  wrapper.innerHTML = `
    <div class="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-primary text-white">
      <span class="material-symbols-outlined text-xl">smart_toy</span>
    </div>
    <div class="flex flex-1 flex-col gap-1 items-start">
      <p class="text-sm md:text-base leading-relaxed flex max-w-3xl rounded-xl rounded-bl-none px-4 py-3 bg-chatbot-bubble text-text-light dark:bg-gray-700 dark:text-text-dark">
        ${safeHtml}
      </p>
      <p class="text-text-secondary-light dark:text-text-secondary-dark text-xs px-1">${time}</p>
    </div>
  `;
  chatContainer.appendChild(wrapper);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addTypingIndicator() {
  const wrapper = document.createElement("div");
  wrapper.id = "typingIndicator";
  wrapper.className = "flex items-end gap-3 max-w-xl";
  wrapper.innerHTML = `
    <div class="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-primary text-white">
      <span class="material-symbols-outlined text-xl">smart_toy</span>
    </div>
    <div class="flex items-center gap-1.5 rounded-xl rounded-bl-none px-4 py-3 bg-chatbot-bubble text-text-light dark:bg-gray-700 dark:text-text-dark">
      <span class="w-2 h-2 bg-text-secondary-light dark:bg-text-secondary-dark rounded-full animate-pulse [animation-delay:-0.3s]"></span>
      <span class="w-2 h-2 bg-text-secondary-light dark:bg-text-secondary-dark rounded-full animate-pulse [animation-delay:-0.15s]"></span>
      <span class="w-2 h-2 bg-text-secondary-light dark:bg-text-secondary-dark rounded-full animate-pulse"></span>
    </div>
  `;
  chatContainer.appendChild(wrapper);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function removeTypingIndicator() {
  const t = document.getElementById("typingIndicator");
  if (t) t.remove();
}

// ---- API CALLS ----

async function callChatApi(symptoms) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symptoms })
  });
  return res.json();
}

async function callDiseaseApi(symptoms) {
  const res = await fetch("/api/predict_disease", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symptoms })
  });
  return res.json();
}

async function callMedicineApi(symptoms) {
  const res = await fetch("/api/recommend_medicines", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symptoms, topk: 3 })
  });
  return res.json();
}

// MAIN SEND HANDLER
sendBtn.addEventListener("click", async () => {
  const symptoms = input.value.trim();
  if (!symptoms) {
    alert("Please enter symptoms");
    return;
  }

  addUserBubble(symptoms);
  input.value = "";
  autoResize();

  output.textContent = "Analyzing...";
  addTypingIndicator();

  try {
    const [chatResp, disease, meds] = await Promise.all([
      callChatApi(symptoms),
      callDiseaseApi(symptoms),
      callMedicineApi(symptoms)
    ]);

    removeTypingIndicator();

    let chatbotText = "";
    let diseaseText = "";
    let medsText    = "";

    if (chatResp && chatResp.message) {
      chatbotText = "Chatbot answer:\n" + chatResp.message;
    } else if (chatResp && chatResp.error) {
      chatbotText = "Chatbot error: " + chatResp.error;
    }

    if (disease && !disease.error) {
      diseaseText  = `Predicted disease: ${disease.prediction}\n`;
      diseaseText += `Description: ${disease.description}\n`;
      if (Array.isArray(disease.precautions)) {
        diseaseText += `Precautions: ${disease.precautions.join(", ")}`;
      } else if (disease.precautions) {
        diseaseText += `Precautions: ${disease.precautions}`;
      }
    } else if (disease && disease.error) {
      diseaseText = "Disease API error: " + disease.error;
    }

    if (meds && !meds.error && Array.isArray(meds.medicines)) {
      medsText = "Medicines:\n";
      meds.medicines.forEach((m, i) => {
        medsText += `${i + 1}. ${m.Name} (${m["Dosage Form"]})\n`;
      });
    } else if (meds && meds.error) {
      medsText = "Medicine API error: " + meds.error;
    }

    const combined = [chatbotText, diseaseText, medsText].filter(Boolean).join("\n\n");
    output.textContent = combined || "No data received from backend.";

    if (chatbotText) addBotBubbleFromTextBlock(chatbotText);
    if (diseaseText) addBotBubbleFromTextBlock(diseaseText);
    if (medsText)    addBotBubbleFromTextBlock(medsText);

  } catch (e) {
    console.error(e);
    removeTypingIndicator();
    output.textContent = "Error calling backend.";
    addBotBubbleFromTextBlock("There was an error contacting the backend. Please check if the Flask server is running.");
  }
});

// Enter to send
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendBtn.click();
  }
});

// THEME TOGGLE
function setTheme(mode) {
  const html = document.documentElement;
  if (mode === "dark") {
    html.classList.add("dark");
    themeLabel.textContent = "Light";
    localStorage.setItem("theme", "dark");
  } else {
    html.classList.remove("dark");
    themeLabel.textContent = "Dark";
    localStorage.setItem("theme", "light");
  }
}
const saved = localStorage.getItem("theme") || "light";
setTheme(saved);

themeToggle.addEventListener("click", () => {
  const html = document.documentElement;
  setTheme(html.classList.contains("dark") ? "light" : "dark");
});

// CLEAR CHAT
refreshChat.addEventListener("click", () => {
  chatContainer.innerHTML = "";
  output.textContent = "";
});
