const screens = {
    upload: document.getElementById("uploadScreen"),
    status: document.getElementById("statusScreen"),
    result: document.getElementById("resultScreen"),
    history: document.getElementById("historyScreen")
};

const nav = document.querySelectorAll(".nav-item");
const audioInput = document.getElementById("audioFile");
const analyzeBtn = document.getElementById("analyzeBtn");
const restartBtn = document.getElementById("restartBtn");
const clearBtn = document.getElementById("clearHistoryBtn");
const themeBtn = document.getElementById("themeToggle");
const ghost = document.getElementById("ghost");

let selectedFile = null;

// -------- Screen switch ----------
function switchScreen(s) {
    Object.values(screens).forEach(x => x.classList.remove("visible"));
    screens[s].classList.add("visible");
}

nav.forEach(btn => {
    btn.onclick = () => {
        const target = btn.dataset.screen;
        if (!target) return;
        switchScreen(target);
        if (target === "history") loadHistory();
    };
});

// -------- Theme ----------
themeBtn.onclick = () => {
    document.body.classList.toggle("light");
    themeBtn.textContent = document.body.classList.contains("light")
        ? "🌙 Dark Mode"
        : "☀️ Light Mode";
};

// -------- File selection ----------
audioInput.onclick = () => {
    audioInput.value = "";
};

audioInput.onchange = e => {
    selectedFile = e.target.files[0];
    document.getElementById("filename").textContent =
        selectedFile?.name || "No file selected";
    analyzeBtn.disabled = !selectedFile;
};

// -------- Waveform drawing ----------
function drawWaveform(array) {
    const canvas = document.getElementById("waveformCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "cyan";
    ctx.lineWidth = 2;
    ctx.beginPath();

    const step = Math.max(1, Math.floor(array.length / canvas.width));
    const mid = canvas.height / 2;
    let x = 0;

    for (let i = 0; i < array.length; i += step) {
        const v = array[i];
        const y = mid + v * 80;
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += 1;
        if (x > canvas.width) break;
    }
    ctx.stroke();
}

// -------- Fake Scan Progress ----------
async function fakeSteps() {
    const steps = [
        { pct: 20, text: "📂 Loading audio file..." },
        { pct: 45, text: "🎧 Extracting MFCC + LFB..." },
        { pct: 75, text: "🤖 Running SANet (demo)..." },
        { pct: 100, text: "📊 Generating spectrograms..." }
    ];

    for (let step of steps) {
        document.getElementById("progressBar").style.width = `${step.pct}%`;
        document.getElementById("progressText").textContent =
            `${step.text} (${step.pct}%)`;
        ghost && ghost.classList.toggle("blink");
        await new Promise(res => setTimeout(res, 800));
    }
}

// -------- Backend Call ----------
async function sendToBackend() {
    const fd = new FormData();
    fd.append("audio", selectedFile);

    try {
        const r = await fetch("/detect", {
            method: "POST",
            body: fd
        });

        if (!r.ok) {
            throw new Error("Server error: " + r.status);
        }

        const data = await r.json();
        console.log("Backend:", data);

        // Update result screen
        document.getElementById("predictionText").textContent = data.prediction || "N/A";
        document.getElementById("confidenceText").textContent =
            data.confidence !== undefined ? `${data.confidence}%` : "N/A";

        if (data.spectrogram1) {
            document.getElementById("plot1").src = data.spectrogram1;
        }
        if (data.spectrogram2) {
            document.getElementById("plot2").src = data.spectrogram2;
        }
        if (data.waveform) {
            drawWaveform(data.waveform);
        }

        saveHistory(data);
        loadHistory();
        switchScreen("result");
    } catch (err) {
        console.error(err);
        alert("❌ Error contacting backend. Check if Flask is running.");
        restartBtn.click();
    }
}

// -------- Analyze Button ----------
analyzeBtn.onclick = async () => {
    if (!selectedFile) return;
    switchScreen("status");
    await fakeSteps();
    await sendToBackend();
};

// -------- Restart ----------
restartBtn.onclick = () => {
    selectedFile = null;
    audioInput.value = "";
    analyzeBtn.disabled = true;
    document.getElementById("progressBar").style.width = "0%";
    document.getElementById("progressText").textContent = "Waiting...";
    switchScreen("upload");
};

// -------- Ghost eye tracking ----------
document.addEventListener("mousemove", e => {
    if (!ghost) return;
    const r = ghost.getBoundingClientRect();
    const dx = (e.clientX - (r.left + r.width / 2)) / 25;
    const dy = (e.clientY - (r.top + r.height / 2)) / 25;
    document.querySelectorAll(".eye").forEach(eye => {
        eye.style.transform = `translate(${dx}px, ${dy}px)`;
    });
});

// -------- History ----------
function saveHistory(d) {
    let h = JSON.parse(localStorage.getItem("scanHistory")) || [];
    h.push({
        prediction: d.prediction,
        confidence: d.confidence,
        time: new Date().toLocaleString()
    });
    localStorage.setItem("scanHistory", JSON.stringify(h));
}

function loadHistory() {
    const list = document.getElementById("historyList");
    const empty = document.getElementById("historyEmpty");

    let h = JSON.parse(localStorage.getItem("scanHistory")) || [];

    if (!h.length) {
        empty.style.display = "block";
        list.innerHTML = "";
        return;
    }

    empty.style.display = "none";
    list.innerHTML = h.map(item => `
        <li>
            <strong>${item.prediction}</strong> • ${item.confidence}%<br>
            <small>${item.time}</small>
        </li>
    `).join("");
}

clearBtn.onclick = () => {
    if (confirm("Clear all scan history?")) {
        localStorage.removeItem("scanHistory");
        loadHistory();
    }
};

// -------- Init ----------
switchScreen("upload");
loadHistory();
