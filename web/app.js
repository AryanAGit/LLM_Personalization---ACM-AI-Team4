const state = {
  users: [],
  selectedUser: null,
};

const els = {
  statusText: document.querySelector("#statusText"),
  userSelect: document.querySelector("#userSelect"),
  nameStat: document.querySelector("#nameStat"),
  mailboxStat: document.querySelector("#mailboxStat"),
  profileStat: document.querySelector("#profileStat"),
  queryStat: document.querySelector("#queryStat"),
  styleText: document.querySelector("#styleText"),
  promptInput: document.querySelector("#promptInput"),
  backendSelect: document.querySelector("#backendSelect"),
  backendLabel: document.querySelector("#backendLabel"),
  modelInput: document.querySelector("#modelInput"),
  baseModelInput: document.querySelector("#baseModelInput"),
  adapterInput: document.querySelector("#adapterInput"),
  adapterRootInput: document.querySelector("#adapterRootInput"),
  generateBtn: document.querySelector("#generateBtn"),
  compareBtn: document.querySelector("#compareBtn"),
  runState: document.querySelector("#runState"),
  resultText: document.querySelector("#resultText"),
  compareState: document.querySelector("#compareState"),
  baseCompareText: document.querySelector("#baseCompareText"),
  loraCompareText: document.querySelector("#loraCompareText"),
  copyBtn: document.querySelector("#copyBtn"),
  testSelect: document.querySelector("#testSelect"),
  runTestBtn: document.querySelector("#runTestBtn"),
  // Content fidelity
  rouge1Score: document.querySelector("#rouge1Score"),
  rougeLScore: document.querySelector("#rougeLScore"),
  chrfScore: document.querySelector("#chrfScore"),
  entityScore: document.querySelector("#entityScore"),
  // Style fidelity
  styleGoldScore: document.querySelector("#styleGoldScore"),
  styleUserScore: document.querySelector("#styleUserScore"),
  lengthScore: document.querySelector("#lengthScore"),
  overallScore: document.querySelector("#overallScore"),
  // Greeting / sign-off buckets
  greetingTypeScore: document.querySelector("#greetingTypeScore"),
  greetingTypeDetail: document.querySelector("#greetingTypeDetail"),
  signoffTypeScore: document.querySelector("#signoffTypeScore"),
  signoffTypeDetail: document.querySelector("#signoffTypeDetail"),
  greetingScore: document.querySelector("#greetingScore"),
  signoffScore: document.querySelector("#signoffScore"),
  testMeta: document.querySelector("#testMeta"),
  incomingText: document.querySelector("#incomingText"),
  actualText: document.querySelector("#actualText"),
  generatedTestText: document.querySelector("#generatedTestText"),
  testState: document.querySelector("#testState"),
};

function selectedBackend() {
  return els.backendSelect.value || "ollama";
}

function statusForBackend(prefix) {
  const backend = selectedBackend();
  if (backend === "ollama") return `${prefix} Ollama RAG...`;
  if (backend === "peft") return `${prefix} LoRA...`;
  return "Using template fallback...";
}

function generationLabel(payload) {
  if (payload.warning) return payload.warning;
  const backend = payload.backend || (payload.used_ollama ? "ollama" : "fallback");
  if (backend === "ollama") return `Generated with Ollama RAG ${payload.model}`;
  if (backend === "peft") {
    const adapter = payload.adapter_path
      ? ` + ${payload.adapter_path}`
      : ` + ${payload.adapter_root || "per-user adapter root"}`;
    return `Generated with LoRA ${payload.base_model || ""}${adapter}`.trim();
  }
  return "Generated with template fallback";
}

function updateBackendUi() {
  const backend = selectedBackend();
  const labels = { fallback: "Template Fallback", ollama: "Ollama RAG", peft: "LoRA" };
  els.backendLabel.textContent = labels[backend] || "Ollama RAG";
  els.modelInput.disabled = backend !== "ollama";
  els.baseModelInput.disabled = backend !== "peft";
  els.adapterInput.disabled = backend !== "peft";
  els.adapterRootInput.disabled = backend !== "peft";
}

async function loadUsers() {
  const response = await fetch("/api/users");
  const payload = await response.json();
  state.users = payload.users || [];

  els.userSelect.innerHTML = "";
  for (const user of state.users) {
    const option = document.createElement("option");
    option.value = user.user_id;
    option.textContent = `${user.inferred_name || "Unknown"} · ${user.source_user}`;
    els.userSelect.appendChild(option);
  }

  if (state.users.length) {
    selectUser(state.users[0].user_id);
    els.statusText.textContent = `${state.users.length} voice loaded from local corpus data`;
  } else {
    els.statusText.textContent = "No voices found";
  }
}

function selectUser(userId) {
  state.selectedUser = state.users.find((user) => String(user.user_id) === String(userId));
  if (!state.selectedUser) return;

  els.nameStat.textContent = state.selectedUser.inferred_name || "-";
  els.mailboxStat.textContent = state.selectedUser.source_user || "-";
  els.profileStat.textContent = `${state.selectedUser.profile_count} passages`;
  els.queryStat.textContent = `${state.selectedUser.query_count} checks`;
  els.styleText.textContent = state.selectedUser.style || "No style profile found.";
  populateTests();
}

async function generate() {
  const prompt = els.promptInput.value.trim();
  if (!prompt) {
    els.resultText.textContent = "Type a prompt first.";
    els.resultText.classList.add("error");
    return;
  }

  els.generateBtn.disabled = true;
  els.runState.textContent = statusForBackend("Calling");
  els.resultText.classList.remove("error");

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        user_id: state.selectedUser?.user_id,
        prompt,
        model: els.modelInput.value.trim() || "llama3.1:8b",
        backend: selectedBackend(),
        use_ollama: selectedBackend() === "ollama",
        base_model: els.baseModelInput.value.trim() || "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path: els.adapterInput.value.trim(),
        adapter_root: els.adapterRootInput.value.trim() || "data/lora_adapters",
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Generation failed.");

    els.resultText.textContent = payload.output;
    els.runState.textContent = generationLabel(payload);
  } catch (error) {
    els.resultText.textContent = error.message;
    els.resultText.classList.add("error");
    els.runState.textContent = "Failed";
  } finally {
    els.generateBtn.disabled = false;
  }
}

async function compareBaseAndLora() {
  const prompt = els.promptInput.value.trim();
  if (!prompt) {
    els.resultText.textContent = "Type a prompt first.";
    els.resultText.classList.add("error");
    return;
  }

  els.compareBtn.disabled = true;
  els.compareState.textContent = "comparing...";
  els.baseCompareText.classList.remove("error");
  els.loraCompareText.classList.remove("error");

  try {
    const response = await fetch("/api/compare", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        user_id: state.selectedUser?.user_id,
        prompt,
        model: els.modelInput.value.trim() || "llama3.1:8b",
        base_model: els.baseModelInput.value.trim() || "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path: els.adapterInput.value.trim(),
        adapter_root: els.adapterRootInput.value.trim() || "data/lora_adapters",
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Comparison failed.");

    els.baseCompareText.textContent = payload.base.ok ? payload.base.output : payload.base.error;
    els.loraCompareText.textContent = payload.lora.ok ? payload.lora.output : payload.lora.error;
    els.baseCompareText.classList.toggle("error", !payload.base.ok);
    els.loraCompareText.classList.toggle("error", !payload.lora.ok);
    els.compareState.textContent = `base vs ${payload.adapter_path || payload.adapter_root || "adapter"}`;
  } catch (error) {
    els.baseCompareText.textContent = error.message;
    els.baseCompareText.classList.add("error");
    els.compareState.textContent = "failed";
  } finally {
    els.compareBtn.disabled = false;
  }
}

function populateTests() {
  const queries = state.selectedUser?.queries || [];
  els.testSelect.innerHTML = "";

  for (const query of queries) {
    const option = document.createElement("option");
    option.value = query.id;
    option.textContent = `${query.id} · ${query.subject}`;
    els.testSelect.appendChild(option);
  }

  if (queries.length) {
    selectTest(queries[0].id);
  } else {
    els.incomingText.textContent = "No held-out tests found for this voice.";
    els.actualText.textContent = "-";
    els.generatedTestText.textContent = "-";
    resetScores();
  }
}

function selectTest(queryId) {
  const query = getSelectedQuery(queryId);
  if (!query) return;

  els.incomingText.textContent = query.input;
  els.actualText.textContent = query.gold;
  els.generatedTestText.textContent = "Run this test to generate a comparison.";
  els.testMeta.textContent = `${query.subject} · ${query.gold_word_count} source words`;
  els.testState.textContent = "not run";
  resetScores();
}

function getSelectedQuery(queryId = els.testSelect.value) {
  return (state.selectedUser?.queries || []).find((query) => query.id === queryId);
}

function resetScores() {
  const dashIds = [
    "rouge1Score", "rougeLScore", "chrfScore", "entityScore",
    "styleGoldScore", "styleUserScore", "lengthScore", "overallScore",
    "greetingTypeScore", "signoffTypeScore",
    "greetingScore", "signoffScore",
  ];
  for (const id of dashIds) {
    if (els[id]) els[id].textContent = "-";
  }
  if (els.greetingTypeDetail) els.greetingTypeDetail.textContent = "-";
  if (els.signoffTypeDetail) els.signoffTypeDetail.textContent = "-";
}

function renderScores(scores) {
  if (!scores) return resetScores();
  // Content fidelity
  els.rouge1Score.textContent = formatPercent(scores.rouge1);
  els.rougeLScore.textContent = formatPercent(scores.rougeL);
  els.chrfScore.textContent = formatPercent(scores.chrf);
  els.entityScore.textContent = formatPercent(scores.entity_overlap);
  // Style fidelity
  els.styleGoldScore.textContent = formatPercent(scores.style_to_gold);
  els.styleUserScore.textContent = formatPercent(scores.style_to_user);
  els.lengthScore.textContent = formatPercent(scores.length_ratio);
  els.overallScore.textContent = formatPercent(
    harmonicMean(scores.content_score, scores.style_score)
  );
  // Greeting / sign-off bucket
  els.greetingTypeScore.textContent = formatMatch(scores.greeting_type_match);
  els.greetingTypeDetail.textContent =
    `pred: ${scores.greeting_type_pred ?? "-"} · gold: ${scores.greeting_type_gold ?? "-"}`;
  els.signoffTypeScore.textContent = formatMatch(scores.signoff_type_match);
  els.signoffTypeDetail.textContent =
    `pred: ${scores.signoff_type_pred ?? "-"} · gold: ${scores.signoff_type_gold ?? "-"}`;
  // Legacy boolean cards (kept for back-compat / cross-check)
  els.greetingScore.textContent = formatMatch(scores.greeting_match);
  els.signoffScore.textContent = formatMatch(scores.signoff_match);
}

function harmonicMean(a, b) {
  if (a == null || b == null || a + b === 0) return null;
  return (2 * a * b) / (a + b);
}

function formatPercent(value) {
  if (value === undefined || value === null) return "-";
  return `${Math.round(Number(value) * 100)}%`;
}

function formatDecimal(value) {
  if (value === undefined || value === null) return "-";
  return Number(value).toFixed(2);
}

async function runSelectedTest() {
  const query = getSelectedQuery();
  if (!query) return;

  els.runTestBtn.disabled = true;
  els.testState.textContent = statusForBackend("calling").toLowerCase();
  els.generatedTestText.classList.remove("error");
  resetScores();

  try {
    const response = await fetch("/api/test", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        user_id: state.selectedUser?.user_id,
        query_id: query.id,
        model: els.modelInput.value.trim() || "llama3.1:8b",
        backend: selectedBackend(),
        use_ollama: selectedBackend() === "ollama",
        base_model: els.baseModelInput.value.trim() || "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path: els.adapterInput.value.trim(),
        adapter_root: els.adapterRootInput.value.trim() || "data/lora_adapters",
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Test failed.");

    els.incomingText.textContent = payload.query.input;
    els.actualText.textContent = payload.actual;
    els.generatedTestText.textContent = payload.generated;
    renderScores(payload.scores);
    els.testState.textContent = payload.used_ollama ? `generated with ${payload.model}` : "generated with fallback";
  } catch (error) {
    els.generatedTestText.textContent = error.message;
    els.generatedTestText.classList.add("error");
    els.testState.textContent = "failed";
  } finally {
    els.runTestBtn.disabled = false;
  }
}

els.userSelect.addEventListener("change", (event) => selectUser(event.target.value));
els.backendSelect.addEventListener("change", updateBackendUi);
els.generateBtn.addEventListener("click", generate);
els.compareBtn.addEventListener("click", compareBaseAndLora);
els.testSelect.addEventListener("change", (event) => selectTest(event.target.value));
els.runTestBtn.addEventListener("click", runSelectedTest);
els.promptInput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    generate();
  }
});

document.querySelectorAll(".sample-btn").forEach((button) => {
  button.addEventListener("click", () => {
    els.promptInput.value = button.dataset.prompt;
    els.promptInput.focus();
  });
});

els.copyBtn.addEventListener("click", async () => {
  await navigator.clipboard.writeText(els.resultText.textContent);
  els.runState.textContent = "Copied";
});

updateBackendUi();
loadUsers().catch((error) => {
  els.statusText.textContent = "Could not load users";
  els.resultText.textContent = error.message;
  els.resultText.classList.add("error");
});
