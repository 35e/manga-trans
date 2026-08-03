const $ = (id) => document.getElementById(id);

const el = {
  pages: $("pages"), file: $("file"), page: $("page"), result: $("result"),
  boxes: $("boxes"), stage: $("stage"), viewer: $("viewer"), empty: $("empty"),
  regions: $("regions"), status: $("status"), detect: $("detect"),
  translate: $("translate"), overlay: $("overlay"), showResult: $("show-result"),
  download: $("download"),
};

const state = {
  page: null, width: 0, height: 0, regions: [], selected: null, zoom: 1, counter: 0,
};

const HANDLES = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];
const MIN_SIZE = 6;
const LOW_CONFIDENCE = 0.6;

const clamp = (value, low, high) => Math.min(Math.max(value, low), high);
const byId = (id) => state.regions.find((region) => region.id === Number(id));

async function api(path, body) {
  let options = {};
  if (body instanceof FormData) {
    options = { method: "POST", body };
  } else if (body) {
    options = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    };
  }
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function status(message, error = false) {
  el.status.textContent = message;
  el.status.classList.toggle("error", error);
}

function enable(on) {
  el.detect.disabled = el.translate.disabled = el.overlay.disabled = !on;
}

async function run(message, work) {
  status(message);
  try {
    return await work();
  } catch (error) {
    status(error.message, true);
    return null;
  }
}

// pages ---------------------------------------------------------------------

async function loadPages() {
  const data = await run("", () => api("/api/pages"));
  if (!data) return;
  el.pages.replaceChildren(...data.pages.map((name) => {
    const item = document.createElement("li");
    item.textContent = name;
    item.classList.toggle("active", name === state.page);
    item.onclick = () => openPage(name);
    return item;
  }));
  if (!data.pages.length) status("no pages yet — add some on the left");
}

async function openPage(name) {
  state.page = name;
  state.regions = [];
  state.selected = null;
  state.counter = 0;
  el.page.src = `/api/image/${encodeURIComponent(name)}`;
  el.empty.hidden = true;
  el.result.hidden = true;
  el.boxes.hidden = false;
  el.showResult.checked = false;
  el.showResult.disabled = true;
  el.download.hidden = true;
  enable(true);
  draw();
  loadPages();
  await detect();
}

// detection, reading, translating --------------------------------------------

async function detect() {
  const data = await run("detecting text…", () => api("/api/detect", { page: state.page }));
  if (!data) return;
  state.width = data.width;
  state.height = data.height;
  state.regions = data.regions.map((region) => ({
    id: ++state.counter,
    box: region.box,
    text: region.text,
    translation: "",
    confidence: region.confidence,
    approved: true,
    busy: false,
  }));
  draw();
  status(`${state.regions.length} region(s)${data.warning ? ` — ${data.warning}` : ""}`);
  if (state.regions.some((region) => region.text)) await translateAll();
}

async function translateAll() {
  const texts = state.regions.map((region) => region.text || "");
  if (!texts.some((text) => text.trim())) {
    status("nothing to translate");
    return;
  }
  const data = await run("translating…", () => api("/api/translate", { texts }));
  if (!data) return;
  state.regions.forEach((region, index) => {
    if (data.translations[index]) region.translation = data.translations[index];
  });
  drawEditor();
  status("translated — edit anything that reads badly");
}

async function reread(region) {
  region.busy = true;
  drawEditor();
  const data = await run("reading region…", () =>
    api("/api/read", { page: state.page, box: region.box }));
  region.busy = false;
  if (data) {
    region.text = data.text;
    status(data.text ? `read: ${data.text}` : "nothing read there");
  }
  drawEditor();
}

async function overlay() {
  const approved = state.regions.filter((region) => region.approved);
  if (!approved.length) {
    status("nothing approved", true);
    return;
  }
  const data = await run("overlaying…", () => api("/api/render", {
    page: state.page,
    regions: approved.map((region) => ({ box: region.box, text: region.translation })),
  }));
  if (!data) return;
  const url = `${data.url}?t=${Date.now()}`;
  el.result.src = url;
  el.showResult.disabled = false;
  el.showResult.checked = true;
  showResult();
  el.download.href = url;
  el.download.download = data.output;
  el.download.hidden = false;
  status(`written to ${data.output}`);
}

// regions -------------------------------------------------------------------

function addRegion(box) {
  const region = {
    id: ++state.counter,
    box,
    text: "",
    translation: "",
    confidence: 1,
    approved: true,
    busy: false,
  };
  state.regions.push(region);
  state.selected = region.id;
  return region;
}

function removeRegion(id) {
  state.regions = state.regions.filter((region) => region.id !== Number(id));
  if (state.selected === Number(id)) state.selected = null;
  draw();
}

function select(id) {
  state.selected = Number(id);
  draw();
  const card = el.regions.querySelector(".card.active");
  if (card) card.scrollIntoView({ block: "nearest" });
}

function draw() {
  drawBoxes();
  drawEditor();
}

function place(node, box) {
  const [x0, y0, x1, y1] = box;
  node.style.left = `${(x0 / state.width) * 100}%`;
  node.style.top = `${(y0 / state.height) * 100}%`;
  node.style.width = `${((x1 - x0) / state.width) * 100}%`;
  node.style.height = `${((y1 - y0) / state.height) * 100}%`;
}

function drawBoxes() {
  if (!state.width) {
    el.boxes.replaceChildren();
    return;
  }
  el.boxes.replaceChildren(...state.regions.map((region, index) => {
    const node = document.createElement("div");
    node.className = "region";
    node.dataset.id = region.id;
    node.classList.toggle("active", region.id === state.selected);
    node.classList.toggle("low", region.confidence < LOW_CONFIDENCE);
    node.classList.toggle("off", !region.approved);
    place(node, region.box);
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = index + 1;
    node.append(tag, ...HANDLES.map((dir) => {
      const handle = document.createElement("span");
      handle.className = "handle";
      handle.dataset.dir = dir;
      return handle;
    }));
    return node;
  }));
}

function card(region, index) {
  const node = document.createElement("li");
  node.className = "card";
  node.classList.toggle("active", region.id === state.selected);
  node.classList.toggle("busy", region.busy);
  node.innerHTML = `
    <div class="head">
      <span class="num"></span>
      <span class="conf"></span>
      <button class="drop">remove</button>
    </div>
    <div class="src"></div>
    <textarea placeholder="English…"></textarea>
    <div class="foot"><label><input type="checkbox" class="ok"> approve</label></div>`;

  node.querySelector(".num").textContent = index + 1;
  const conf = node.querySelector(".conf");
  conf.textContent = `${Math.round(region.confidence * 100)}%`;
  conf.classList.toggle("low", region.confidence < LOW_CONFIDENCE);
  node.querySelector(".src").textContent =
    region.busy ? "reading…" : region.text || "—";

  const area = node.querySelector("textarea");
  area.value = region.translation;
  area.oninput = () => { region.translation = area.value; };

  const approve = node.querySelector(".ok");
  approve.checked = region.approved;
  approve.onchange = () => { region.approved = approve.checked; drawBoxes(); };

  node.querySelector(".drop").onclick = () => removeRegion(region.id);
  node.onmousedown = (event) => {
    if (!event.target.closest("button")) select(region.id);
  };
  return node;
}

function drawEditor() {
  el.regions.replaceChildren(...state.regions.map(card));
}

// dragging ------------------------------------------------------------------

function imagePoint(event) {
  const rect = el.page.getBoundingClientRect();
  return [
    clamp(Math.round(((event.clientX - rect.left) / rect.width) * state.width), 0, state.width),
    clamp(Math.round(((event.clientY - rect.top) / rect.height) * state.height), 0, state.height),
  ];
}

function dragged(origin, mode, dx, dy) {
  let [x0, y0, x1, y1] = origin;
  if (mode === "move") {
    dx = clamp(dx, -x0, state.width - x1);
    dy = clamp(dy, -y0, state.height - y1);
    return [x0 + dx, y0 + dy, x1 + dx, y1 + dy];
  }
  if (mode.includes("n")) y0 += dy;
  if (mode.includes("s")) y1 += dy;
  if (mode.includes("w")) x0 += dx;
  if (mode.includes("e")) x1 += dx;
  return [
    clamp(Math.min(x0, x1), 0, state.width),
    clamp(Math.min(y0, y1), 0, state.height),
    clamp(Math.max(x0, x1), 0, state.width),
    clamp(Math.max(y0, y1), 0, state.height),
  ];
}

el.stage.addEventListener("pointerdown", (event) => {
  if (!state.width || el.showResult.checked || event.button !== 0) return;

  const start = imagePoint(event);
  const node = event.target.closest(".region");
  let region;
  let mode;
  let fresh = false;

  if (node) {
    region = byId(node.dataset.id);
    mode = event.target.dataset.dir || "move";
    select(region.id);
  } else {
    region = addRegion([start[0], start[1], start[0], start[1]]);
    mode = "se";
    fresh = true;
    draw();
  }

  const origin = [...region.box];
  const before = origin.join();
  event.preventDefault();
  el.stage.setPointerCapture(event.pointerId);

  const onMove = (moveEvent) => {
    const [x, y] = imagePoint(moveEvent);
    region.box = dragged(origin, mode, x - start[0], y - start[1]);
    place(el.boxes.querySelector(`[data-id="${region.id}"]`), region.box);
  };

  const onUp = () => {
    el.stage.removeEventListener("pointermove", onMove);
    el.stage.removeEventListener("pointerup", onUp);
    el.stage.releasePointerCapture(event.pointerId);
    const [x0, y0, x1, y1] = region.box;
    if (x1 - x0 < MIN_SIZE || y1 - y0 < MIN_SIZE) {
      if (fresh) removeRegion(region.id);
      else region.box = origin;
      draw();
      return;
    }
    if (region.box.join() !== before) reread(region);
  };

  el.stage.addEventListener("pointermove", onMove);
  el.stage.addEventListener("pointerup", onUp);
});

// chrome --------------------------------------------------------------------

function showResult() {
  el.result.hidden = !el.showResult.checked;
  el.boxes.hidden = el.showResult.checked;
}

function zoom(step) {
  state.zoom = clamp(state.zoom + step, 0.3, 5);
  el.stage.style.width = `${state.zoom * 100}%`;
}

function fitPage() {
  const room = el.viewer.clientWidth - 32;
  const aspect = el.page.naturalWidth / el.page.naturalHeight;
  if (!room || !aspect) return;
  const wanted = Math.min(room, (el.viewer.clientHeight - 32) * aspect);
  state.zoom = 0;
  zoom(wanted / room);
}

el.page.onload = fitPage;

el.detect.onclick = detect;
el.translate.onclick = translateAll;
el.overlay.onclick = overlay;
el.showResult.onchange = showResult;
$("zoom-in").onclick = () => zoom(0.25);
$("zoom-out").onclick = () => zoom(-0.25);

el.file.onchange = async () => {
  const form = new FormData();
  [...el.file.files].forEach((file) => form.append("file", file));
  await run("uploading…", () => api("/api/pages", form));
  el.file.value = "";
  status("added");
  loadPages();
};

window.addEventListener("keydown", (event) => {
  if (event.target.matches("input, textarea")) return;
  if ((event.key === "Delete" || event.key === "Backspace") && state.selected) {
    removeRegion(state.selected);
  }
});

loadPages();
