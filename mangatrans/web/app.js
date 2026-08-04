const $ = (id) => document.getElementById(id);

const el = {
  pages: $("pages"), file: $("file"), page: $("page"), result: $("result"),
  boxes: $("boxes"), texts: $("texts"), stage: $("stage"), viewer: $("viewer"),
  empty: $("empty"), regions: $("regions"), status: $("status"), detect: $("detect"),
  translate: $("translate"), overlay: $("overlay"), showResult: $("show-result"),
  download: $("download"), ruler: $("ruler"),
  pickBoxes: $("pick-boxes"), pickText: $("pick-text"),
};

const state = {
  page: null, width: 0, height: 0, regions: [], selected: null,
  zoom: 1, counter: 0, mode: "boxes",
};

const HANDLES = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];
const MIN_SIZE = 6;
const LOW_CONFIDENCE = 0.6;
// Kept in step with render.py, so the preview lands where the type does.
const FONT_MIN = 8;
const INSET = 0.06;

const clamp = (value, low, high) => Math.min(Math.max(value, low), high);
const byId = (id) => state.regions.find((region) => region.id === Number(id));
const shown = (layer, region) => layer.querySelector(`[data-id="${region.id}"]`);

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
  el.texts.hidden = false;
  el.showResult.checked = false;
  el.showResult.disabled = true;
  el.download.hidden = true;
  enable(true);
  draw();
  loadPages();
  await detect();
}

// detection, reading, translating --------------------------------------------

function newRegion(box, extra = {}) {
  return {
    id: ++state.counter,
    box,
    textBox: [...box],
    moved: false,
    text: "",
    translation: "",
    confidence: 1,
    approved: true,
    busy: false,
    trouble: "",
    ...extra,
  };
}

async function detect() {
  const data = await run("detecting text…", () => api("/api/detect", { page: state.page }));
  if (!data) return;
  state.width = data.width;
  state.height = data.height;
  state.regions = data.regions.map((region) => newRegion(region.box, {
    text: region.text,
    confidence: region.confidence,
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
  draw();
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
    regions: approved.map((region) => ({
      box: region.box,
      text_box: region.textBox,
      text: region.translation,
    })),
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

  // Say which regions came out bare or overrun rather than leaving them to be
  // found by eye on the finished page.
  const numbers = (indexes) => indexes
    .map((index) => state.regions.indexOf(approved[index]) + 1)
    .join(", ");
  const notes = [];
  if (data.blank.length) notes.push(`${numbers(data.blank)} covered with no English`);
  if (data.overflow.length) notes.push(`${numbers(data.overflow)} runs over its box`);
  if (data.tight.length) notes.push(`${numbers(data.tight)} had to be hyphenated`);
  status(`written to ${data.output}${notes.length ? ` — ${notes.join("; ")}` : ""}`,
    notes.length > 0);
}

// regions -------------------------------------------------------------------

function addRegion(box) {
  const region = newRegion(box);
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
  drawTexts();
  drawEditor();
}

function place(node, box) {
  const [x0, y0, x1, y1] = box;
  node.style.left = `${(x0 / state.width) * 100}%`;
  node.style.top = `${(y0 / state.height) * 100}%`;
  node.style.width = `${((x1 - x0) / state.width) * 100}%`;
  node.style.height = `${((y1 - y0) / state.height) * 100}%`;
}

function handles() {
  return HANDLES.map((dir) => {
    const handle = document.createElement("span");
    handle.className = "handle";
    handle.dataset.dir = dir;
    return handle;
  });
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
    node.append(tag, ...handles());
    return node;
  }));
}

// The English, set on the picture ---------------------------------------------

const REF = 100;

/** Width of the longest word at REF pixels — the one that forces a break. */
function widestWord(text) {
  el.ruler.style.whiteSpace = "pre";
  el.ruler.style.width = "max-content";
  el.ruler.style.fontSize = `${REF}px`;
  el.ruler.textContent = text.split(/\s+/).join("\n");
  const width = el.ruler.getBoundingClientRect().width;
  el.ruler.style.whiteSpace = "";
  return Math.max(1, width);
}

/** Largest size up to `top` at which the wrapped text is no taller than `height`. */
function largest(width, height, top) {
  let lo = FONT_MIN;
  let hi = top;
  let best = null;
  el.ruler.style.width = `${Math.max(1, width)}px`;
  while (lo <= hi) {
    const size = (lo + hi) >> 1;
    el.ruler.style.fontSize = `${size}px`;
    if (el.ruler.getBoundingClientRect().height <= height) {
      best = size;
      lo = size + 1;
    } else {
      hi = size - 1;
    }
  }
  return best;
}

/** How render.py would set `text`, in page pixels: whole words first, then broken. */
function typeSize(text, width, height) {
  const ceiling = Math.max(FONT_MIN, Math.round(height));
  // Above this size some word is wider than the line and has to be hyphenated.
  const unbroken = Math.floor((REF * width) / widestWord(text));
  el.ruler.textContent = text;
  const kept = largest(width, height, Math.min(ceiling, unbroken));
  // Nothing fits even broken: show it at the smallest size anyway, overrunning
  // the box, and mark it. That is what the renderer does, so the page matches.
  const size = kept ?? largest(width, height, ceiling);
  return { size: size ?? FONT_MIN, fits: size !== null, whole: kept !== null };
}

function setType(node, region) {
  const [x0, y0, x1, y1] = region.textBox;
  const [width, height] = [x1 - x0, y1 - y0];
  const inset = Math.max(1, Math.round(INSET * Math.min(width, height)));
  const room = [width - 2 * inset, height - 2 * inset];
  const [across, down] = room[0] >= FONT_MIN && room[1] >= FONT_MIN
    ? room : [width, height];
  const text = region.translation.trim();
  const { size, fits, whole } = typeSize(text, across, down);

  const type = node.querySelector(".type");
  const cqw = (value) => `${(value / Math.max(1, width)) * 100}cqw`;
  type.style.fontSize = cqw(size);
  type.style.padding = `${cqw((height - down) / 2)} ${cqw((width - across) / 2)}`;
  node.querySelector(".line").textContent = text;
  region.trouble = fits ? (whole ? "" : "hyphenated to fit") : "runs over its box";
  node.classList.toggle("over", !fits);
  node.classList.toggle("tight", fits && !whole);
}

function drawTexts() {
  // Only a region with English on the page can be in trouble over how it sits.
  state.regions.forEach((region) => { region.trouble = ""; });
  if (!state.width) {
    el.texts.replaceChildren();
    return;
  }
  el.texts.replaceChildren(...state.regions
    .filter((region) => region.approved && region.translation.trim())
    .map((region) => {
      const node = document.createElement("div");
      node.className = "textbox";
      node.dataset.id = region.id;
      node.classList.toggle("active", region.id === state.selected);
      place(node, region.textBox);
      const type = document.createElement("div");
      type.className = "type";
      const line = document.createElement("div");
      line.className = "line";
      type.append(line);
      node.append(type, ...handles());
      setType(node, region);
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
    <div class="foot">
      <label><input type="checkbox" class="ok"> approve</label>
      <button class="replace" hidden>re-place</button>
      <span class="note" hidden></span>
    </div>`;

  node.querySelector(".num").textContent = index + 1;
  const conf = node.querySelector(".conf");
  conf.textContent = `${Math.round(region.confidence * 100)}%`;
  conf.classList.toggle("low", region.confidence < LOW_CONFIDENCE);
  node.querySelector(".src").textContent =
    region.busy ? "reading…" : region.text || "—";

  const area = node.querySelector("textarea");
  area.value = region.translation;
  area.oninput = () => {
    region.translation = area.value;
    // Typing re-sets this one box; a box that has just appeared, or just been
    // emptied, needs the layer built again.
    const text = shown(el.texts, region);
    if (text && area.value.trim()) setType(text, region);
    else drawTexts();
  };

  const approve = node.querySelector(".ok");
  approve.checked = region.approved;
  approve.onchange = () => {
    region.approved = approve.checked;
    drawBoxes();
    drawTexts();
  };

  const replace = node.querySelector(".replace");
  replace.hidden = !region.moved;
  replace.title = "put the English back over its region";
  replace.onclick = () => {
    region.textBox = [...region.box];
    region.moved = false;
    draw();
  };

  const note = node.querySelector(".note");
  const trouble = region.approved && !region.translation.trim()
    ? "will only be cleaned" : region.trouble;
  note.hidden = !trouble;
  note.textContent = trouble;

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

/** Move the region's box, bringing its English along unless that was placed. */
function moveBox(region, box) {
  region.box = box;
  place(shown(el.boxes, region), box);
  if (region.moved) return;
  region.textBox = [...box];
  const text = shown(el.texts, region);
  if (text) {
    place(text, region.textBox);
    setType(text, region);
  }
}

function moveText(region, box) {
  region.textBox = box;
  region.moved = true;
  const text = shown(el.texts, region);
  place(text, box);
  setType(text, region);
}

function grab(event, mode, origin, apply, drop) {
  const from = imagePoint(event);
  const before = origin.join();
  event.preventDefault();
  el.stage.setPointerCapture(event.pointerId);

  const onMove = (moveEvent) => {
    const [x, y] = imagePoint(moveEvent);
    apply(dragged(origin, mode, x - from[0], y - from[1]));
  };

  const onUp = () => {
    el.stage.removeEventListener("pointermove", onMove);
    el.stage.removeEventListener("pointerup", onUp);
    el.stage.releasePointerCapture(event.pointerId);
    drop(before);
  };

  el.stage.addEventListener("pointermove", onMove);
  el.stage.addEventListener("pointerup", onUp);
}

const tooSmall = ([x0, y0, x1, y1]) => x1 - x0 < MIN_SIZE || y1 - y0 < MIN_SIZE;

el.stage.addEventListener("pointerdown", (event) => {
  if (!state.width || el.showResult.checked || event.button !== 0) return;

  if (state.mode === "text") {
    const hit = event.target.closest(".textbox");
    if (!hit) return;
    const region = byId(hit.dataset.id);
    const origin = [...region.textBox];
    const placed = region.moved;
    select(region.id);
    grab(event, event.target.dataset.dir || "move", origin,
      (box) => moveText(region, box),
      () => {
        if (tooSmall(region.textBox)) {
          region.textBox = origin;
          region.moved = placed;
        }
        draw();
      });
    return;
  }

  const hit = event.target.closest(".region");
  let region;
  let mode;
  let fresh = false;

  if (hit) {
    region = byId(hit.dataset.id);
    mode = event.target.dataset.dir || "move";
    select(region.id);
  } else {
    const [x, y] = imagePoint(event);
    region = addRegion([x, y, x, y]);
    mode = "se";
    fresh = true;
    draw();
  }

  const origin = [...region.box];
  grab(event, mode, origin, (box) => moveBox(region, box), (before) => {
    if (tooSmall(region.box)) {
      if (fresh) return removeRegion(region.id);
      moveBox(region, origin);
      draw();
      return;
    }
    draw();
    if (region.box.join() !== before) reread(region);
  });
});

// chrome --------------------------------------------------------------------

function showResult() {
  el.result.hidden = !el.showResult.checked;
  el.boxes.hidden = el.showResult.checked;
  el.texts.hidden = el.showResult.checked;
}

function mode(which) {
  state.mode = which;
  el.stage.className = which;
  el.pickBoxes.classList.toggle("on", which === "boxes");
  el.pickText.classList.toggle("on", which === "text");
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
el.pickBoxes.onclick = () => mode("boxes");
el.pickText.onclick = () => mode("text");
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
// Sizes measured against the fallback font would be wrong; measure again once
// the real one is in.
document.fonts.load("16px page").then(drawTexts, () => {});
