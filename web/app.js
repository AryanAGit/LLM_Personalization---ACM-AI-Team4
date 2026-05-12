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
  baseModelStat: document.querySelector("#baseModelStat"),
  styleText: document.querySelector("#styleText"),
  promptInput: document.querySelector("#promptInput"),
  modelInput: document.querySelector("#modelInput"),
  ollamaToggle: document.querySelector("#ollamaToggle"),
  generateBtn: document.querySelector("#generateBtn"),
  runState: document.querySelector("#runState"),
  resultText: document.querySelector("#resultText"),
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
    els.statusText.textContent = `${state.users.length} users loaded from processed Enron data`;
  } else {
    els.statusText.textContent = "No users found";
  }
}

function selectUser(userId) {
  state.selectedUser = state.users.find((user) => String(user.user_id) === String(userId));
  if (!state.selectedUser) return;

  els.nameStat.textContent = state.selectedUser.inferred_name || "-";
  els.mailboxStat.textContent = state.selectedUser.source_user || "-";
  els.profileStat.textContent = `${state.selectedUser.profile_count} emails`;
  els.queryStat.textContent = `${state.selectedUser.query_count} checks`;
  els.baseModelStat.textContent = state.selectedUser.base_model || "-";
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
  els.runState.textContent = els.ollamaToggle.checked ? "Calling Llama..." : "Using fallback...";
  els.resultText.classList.remove("error");

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        user_id: state.selectedUser?.user_id,
        prompt,
        model: els.modelInput.value.trim() || "llama3.1:8b",
        use_ollama: els.ollamaToggle.checked,
      }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Generation failed.");

    els.resultText.textContent = payload.output;
    els.runState.textContent = payload.used_ollama ? `Generated with ${payload.model}` : "Generated with fallback";
  } catch (error) {
    els.resultText.textContent = error.message;
    els.resultText.classList.add("error");
    els.runState.textContent = "Failed";
  } finally {
    els.generateBtn.disabled = false;
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
    els.incomingText.textContent = "No held-out tests found for this user.";
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
  els.testMeta.textContent = `${query.has_context ? "context" : "subject only"} · ${query.gold_word_count} gold words`;
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

function formatMatch(value) {
  if (value === undefined || value === null) return "-";
  return Number(value) >= 1 ? "match" : "miss";
}

async function runSelectedTest() {
  const query = getSelectedQuery();
  if (!query) return;

  els.runTestBtn.disabled = true;
  els.testState.textContent = els.ollamaToggle.checked ? "calling Llama..." : "fallback...";
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
        use_ollama: els.ollamaToggle.checked,
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
els.generateBtn.addEventListener("click", generate);
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

loadUsers().catch((error) => {
  els.statusText.textContent = "Could not load users";
  els.resultText.textContent = error.message;
  els.resultText.classList.add("error");
});
