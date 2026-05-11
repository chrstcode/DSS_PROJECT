const validationRules = {
  stress_score: { min: 0, max: 100, type: "number" },
  anxiety_score: { min: 0, max: 100, type: "number" },
  exam_pressure: { min: 0, max: 10, type: "number" },
  sleep_hours: { min: 0, max: 12, type: "number", step: 0.5 },
  social_support: { min: 0, max: 10, type: "number" },
  heart_rate: { min: 40, max: 200, type: "number" },
  physical_activity: { min: 0, max: 10, type: "number" },
  assignment_load: { min: 0, max: 10, type: "number" },
  study_hours: { min: 0, max: 12, type: "number", step: 0.5 },
  attendance: { min: 0, max: 100, type: "number" },
  screen_time: { min: 0, max: 24, type: "number", step: 0.5 },
  facial_emotion: {
    type: "select",
    options: ["Happy", "Sad", "Angry", "Neutral", "Surprised"],
  },
  mood_state: {
    type: "select",
    options: ["Happy", "Sad", "Anxious", "Neutral", "Excited"],
  },
  intervention_response: {
    type: "select",
    options: ["Positive", "Negative", "Neutral"],
  },
  reward_score: { min: 0, max: 100, type: "number" },
};

function updateSliderValue(slider) {
  const display = document.querySelector(`[data-for="${slider.id}"]`);
  if (display) {
    display.textContent = slider.value;
  }
}

function getFormData(form) {
  const data = new FormData(form);
  const payload = {};
  for (const [key, value] of data.entries()) {
    if (validationRules[key]?.type === "select") {
      payload[key] = value;
    } else {
      payload[key] = Number(value);
    }
  }
  return payload;
}

function validateInputs(payload) {
  for (const [field, rules] of Object.entries(validationRules)) {
    const value = payload[field];
    if (value === undefined || value === null || value === "") {
      return `Kolom ${field} wajib diisi.`;
    }

    if (rules.type === "select") {
      if (!rules.options.includes(value)) {
        return `Nilai untuk ${field} tidak valid.`;
      }
      continue;
    }

    if (Number.isNaN(value)) {
      return `Kolom ${field} harus berupa angka.`;
    }

    if (value < rules.min || value > rules.max) {
      return `Kolom ${field} harus di antara ${rules.min} dan ${rules.max}.`;
    }
  }
  return "";
}

async function submitPrediction(event) {
  event.preventDefault();

  const form = event.target;
  const results = document.getElementById("results");
  const errorMessage = document.getElementById("errorMessage");

  errorMessage.hidden = true;
  results.hidden = true;

  const payload = getFormData(form);
  const validationError = validateInputs(payload);
  if (validationError) {
    showError(validationError);
    return;
  }

  setLoading(true);
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      const message = errorData.error || `HTTP error ${response.status}`;
      throw new Error(message);
    }

    const result = await response.json();
    if (!result.success) {
      throw new Error(result.error || "Prediksi gagal");
    }

    displayResults(result);
  } catch (error) {
    showError(error.message || "Terjadi kesalahan");
  } finally {
    setLoading(false);
  }
}

function displayResults(data) {
  const results = document.getElementById("results");
  const predictions = data.predictions;

  const stressBadge = document.getElementById("stressBadge");
  const anxietyBadge = document.getElementById("anxietyBadge");
  const finalState = document.getElementById("finalState");
  const recommendation = document.getElementById("recommendation");
  const confidenceValue = document.getElementById("confidenceValue");
  const confidenceBar = document.getElementById("confidenceBar");
  const explanationText = document.getElementById("explanationText");

  stressBadge.textContent = `Stres: ${predictions.stress_level}`;
  anxietyBadge.textContent = `Kecemasan: ${predictions.anxiety_level}`;

  stressBadge.className = `badge ${getLevelClass(predictions.stress_level)}`;
  anxietyBadge.className = `badge ${getLevelClass(predictions.anxiety_level)}`;

  finalState.textContent = predictions.final_state;
  recommendation.textContent = predictions.recommended_intervention;
  const confidencePercent = Math.round((data.confidence || 0) * 100);
  confidenceValue.textContent = `${confidencePercent}%`;
  confidenceBar.style.width = `${confidencePercent}%`;
  explanationText.textContent = data.explanation;

  results.hidden = false;
  results.classList.add("fade-in");
  setTimeout(() => results.classList.remove("fade-in"), 600);
}

function setLoading(isLoading) {
  const loading = document.getElementById("loading");
  const submitButton = document.querySelector(".btn-submit");
  loading.hidden = !isLoading;
  submitButton.disabled = isLoading;
}

function showError(message) {
  const errorMessage = document.getElementById("errorMessage");
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

function getLevelClass(level) {
  const normalized = String(level).toLowerCase();
  if (normalized === "high") {
    return "badge-high";
  }
  if (normalized === "medium") {
    return "badge-medium";
  }
  return "badge-low";
}

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predictionForm");
  const sliders = document.querySelectorAll("input[type='range']");

  sliders.forEach((slider) => {
    updateSliderValue(slider);
    slider.addEventListener("input", () => updateSliderValue(slider));
  });

  form.addEventListener("submit", submitPrediction);
});
