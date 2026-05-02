// Mermaid render + CSS-transform pan/zoom for Zensical docs.
//
// Background bugs we work around:
//   1. Zensical's bundle transforms each `<pre class="mermaid"><code>...` block
//      into an empty `<div class="mermaid" data-processed="true">` BEFORE
//      mermaid runs, dropping the diagram source on the floor.
//   2. Whatever mermaid Zensical loads has a broken `mermaid.run()` code path
//      for flowchart-v2 (renders error placeholders with NaN transforms),
//      while `mermaid.render()` works correctly.
//   3. Zensical's parallel mermaid pipeline appends orphan error SVGs to body.
//
// The pan/zoom approach is the same one modelcontextprotocol.io uses: wrap
// the .mermaid div in a positioned container, apply zoom/pan via CSS transform
// on the inner div (transform: translate(x,y) scale(s)), and provide a small
// floating control panel. Much simpler than svg-pan-zoom and doesn't touch
// the SVG at all.

import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";

mermaid.initialize({
  startOnLoad: false,
  securityLevel: "loose",
  flowchart: { htmlLabels: true, useMaxWidth: true },
});

const ZOOM_STEP = 1.2;
const ZOOM_MIN = 0.3;
const ZOOM_MAX = 5;
const PAN_STEP = 50;

async function fetchDiagramSources() {
  const r = await fetch(window.location.href, { cache: "no-store" });
  const html = await r.text();
  const doc = new DOMParser().parseFromString(html, "text/html");
  return Array.from(doc.querySelectorAll("pre.mermaid > code")).map((c) => c.textContent);
}

function setTransform(div, state) {
  div.style.transform = `translate(${state.x}px, ${state.y}px) scale(${state.scale})`;
}

function makeButton(label, svgPath) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.setAttribute("aria-label", label);
  btn.title = label;
  btn.className = "mermaid-zoom-btn";
  btn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${svgPath}</svg>`;
  return btn;
}

const ICONS = {
  zoomIn: '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/>',
  zoomOut: '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="8" y1="11" x2="14" y2="11"/>',
  reset: '<polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/>',
  panUp: '<polyline points="18 15 12 9 6 15"/>',
  panDown: '<polyline points="6 9 12 15 18 9"/>',
  panLeft: '<polyline points="15 18 9 12 15 6"/>',
  panRight: '<polyline points="9 18 15 12 9 6"/>',
};

function attachZoom(div) {
  if (div.dataset.zoomAttached === "1") return;
  div.dataset.zoomAttached = "1";

  const state = { x: 0, y: 0, scale: 1 };

  // Wrap the .mermaid div in a positioned container
  const wrapper = document.createElement("div");
  wrapper.className = "mermaid-zoom-wrapper";
  div.parentNode.insertBefore(wrapper, div);
  wrapper.appendChild(div);

  div.style.transformOrigin = "center center";
  div.style.transition = "transform 0.15s ease-out";
  setTransform(div, state);

  // Floating control panel
  const controls = document.createElement("div");
  controls.className = "mermaid-zoom-controls";
  controls.innerHTML = `
    <div class="mermaid-zoom-row">
      <span></span>
      <button type="button" aria-label="Pan up" class="mermaid-zoom-btn">${`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS.panUp}</svg>`}</button>
      <button type="button" aria-label="Zoom in" class="mermaid-zoom-btn">${`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS.zoomIn}</svg>`}</button>
    </div>
    <div class="mermaid-zoom-row">
      <button type="button" aria-label="Pan left" class="mermaid-zoom-btn">${`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS.panLeft}</svg>`}</button>
      <button type="button" aria-label="Reset view" class="mermaid-zoom-btn">${`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS.reset}</svg>`}</button>
      <button type="button" aria-label="Pan right" class="mermaid-zoom-btn">${`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS.panRight}</svg>`}</button>
    </div>
    <div class="mermaid-zoom-row">
      <span></span>
      <button type="button" aria-label="Pan down" class="mermaid-zoom-btn">${`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS.panDown}</svg>`}</button>
      <button type="button" aria-label="Zoom out" class="mermaid-zoom-btn">${`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${ICONS.zoomOut}</svg>`}</button>
    </div>
  `;
  wrapper.appendChild(controls);

  const handlers = {
    "Pan up": () => { state.y += PAN_STEP; setTransform(div, state); },
    "Pan down": () => { state.y -= PAN_STEP; setTransform(div, state); },
    "Pan left": () => { state.x += PAN_STEP; setTransform(div, state); },
    "Pan right": () => { state.x -= PAN_STEP; setTransform(div, state); },
    "Zoom in": () => { state.scale = Math.min(ZOOM_MAX, state.scale * ZOOM_STEP); setTransform(div, state); },
    "Zoom out": () => { state.scale = Math.max(ZOOM_MIN, state.scale / ZOOM_STEP); setTransform(div, state); },
    "Reset view": () => { state.x = 0; state.y = 0; state.scale = 1; setTransform(div, state); },
  };

  controls.querySelectorAll("button").forEach((btn) => {
    const label = btn.getAttribute("aria-label");
    btn.addEventListener("click", (e) => { e.preventDefault(); handlers[label]?.(); });
  });

  // Scroll-wheel zoom on hover
  wrapper.addEventListener("wheel", (e) => {
    if (!e.ctrlKey && !e.metaKey) return;  // require Ctrl/Cmd to zoom
    e.preventDefault();
    if (e.deltaY < 0) handlers["Zoom in"]();
    else handlers["Zoom out"]();
  }, { passive: false });
}

async function renderAll() {
  const divs = document.querySelectorAll(".mermaid");
  if (divs.length === 0) return;

  const sources = await fetchDiagramSources();
  if (sources.length === 0) {
    console.warn("[mermaid pan-zoom] no diagram sources found in raw HTML");
    return;
  }

  for (let i = 0; i < divs.length; i++) {
    const div = divs[i];
    const source = sources[i];
    if (!source) continue;
    try {
      const id = `padwan-mermaid-${i}-${Date.now()}`;
      const { svg, bindFunctions } = await mermaid.render(id, source);
      div.innerHTML = svg;
      // Lock the div so Zensical's later mermaid pass doesn't clobber our render.
      div.dataset.processed = "true";
      div.classList.remove("mermaid");
      div.classList.add("mermaid-rendered");
      const svgEl = div.querySelector("svg");
      if (svgEl && bindFunctions) bindFunctions(svgEl);
      attachZoom(div);
    } catch (e) {
      console.warn("[mermaid pan-zoom] render failed for diagram", i, e);
    }
  }
}

// Clean up error SVGs that Zensical's parallel mermaid pipeline appends to body.
function cleanupOrphanedErrors() {
  document.querySelectorAll("body > div > svg[aria-roledescription='error']")
    .forEach((svg) => svg.parentElement?.remove());
}

renderAll();

const cleanupObserver = new MutationObserver(cleanupOrphanedErrors);
cleanupObserver.observe(document.body, { childList: true });
for (const delay of [500, 1500, 3000]) setTimeout(cleanupOrphanedErrors, delay);

console.debug("[mermaid pan-zoom] installed");
