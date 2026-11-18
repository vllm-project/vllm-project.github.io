"use strict";

// Shared controls for the two figures. Playback is reader-initiated and stops at
// the last step; the diagram is an explanation, not an inference-time simulation.
function createDuplexPlayer({ steps, overview, render }) {
  const host = document.querySelector("#player");
  host.innerHTML = '<div class="controls"><button type="button" data-action="play">Play</button>' +
    '<button type="button" data-action="prev" aria-label="Previous step">← Previous</button>' +
    '<button type="button" data-action="next" aria-label="Next step">Next →</button>' +
    '<button type="button" data-action="all" aria-pressed="true">Full view</button><span class="count"></span></div>' +
    '<div class="steps" role="group" aria-label="Walkthrough steps"></div>' +
    '<section class="explain" aria-live="polite" aria-atomic="true"><h2></h2><p></p></section>';
  const find = selector => host.querySelector(selector);
  let current = -1, playing = false, timer;
  const buttons = steps.map((step, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `${index + 1} · ${step.label}`;
    button.setAttribute("aria-pressed", "false");
    button.onclick = () => select(index);
    find(".steps").appendChild(button);
    return button;
  });
  function stop() {
    playing = false;
    clearTimeout(timer);
    find('[data-action="play"]').textContent = current === steps.length - 1 ? "Replay" : "Play";
  }
  function show(index) {
    current = index;
    const step = index < 0 ? overview : steps[index];
    find(".explain h2").textContent = step.title;
    find(".explain p").textContent = step.text;
    find(".count").textContent = index < 0 ? `${steps.length} steps · full view` : `${index + 1} / ${steps.length}`;
    find('[data-action="all"]').setAttribute("aria-pressed", String(index < 0));
    find('[data-action="prev"]').disabled = index <= 0;
    find('[data-action="next"]').disabled = index === steps.length - 1;
    buttons.forEach((button, i) => button.setAttribute("aria-pressed", String(i === index)));
    render(step, index);
    if (!playing) stop();
  }
  function select(index) { stop(); show(Math.max(-1, Math.min(index, steps.length - 1))); }
  function advance() {
    timer = setTimeout(() => {
      if (current === steps.length - 1) { stop(); return; }
      show(current + 1);
      advance();
    }, 5200);
  }
  find('[data-action="play"]').onclick = () => {
    if (playing) { stop(); return; }
    playing = true;
    if (current < 0 || current === steps.length - 1) show(0);
    find('[data-action="play"]').textContent = "Pause";
    advance();
  };
  find('[data-action="prev"]').onclick = () => select(current - 1);
  find('[data-action="next"]').onclick = () => select(current + 1);
  find('[data-action="all"]').onclick = () => select(-1);
  host.onkeydown = event => {
    if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
      event.preventDefault();
      select(Math.max(0, current + (event.key === "ArrowRight" ? 1 : -1)));
    }
  };
  const hide = () => { if (document.hidden) stop(); };
  document.addEventListener("visibilitychange", hide);
  show(-1);
  return { select, dispose() { stop(); document.removeEventListener("visibilitychange", hide); } };
}

function bindDiagramSteps(root, player) {
  root.querySelectorAll("[data-step]").forEach(node => {
    node.setAttribute("role", "button");
    node.setAttribute("tabindex", "0");
    node.addEventListener("click", () => player.select(Number(node.dataset.step)));
    node.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        player.select(Number(node.dataset.step));
      }
    });
  });
}

function highlightDiagram(root, step) {
  root.querySelectorAll(".node, .edge").forEach(node => {
    node.classList.toggle("active", [...step.nodes, ...step.edges].includes(node.id));
  });
  const viewport = root.parentElement;
  if (viewport.scrollWidth <= viewport.clientWidth) return;
  if (!step.nodes.length) { viewport.scrollLeft = 0; return; }
  const selected = root.querySelector("#" + step.nodes[0]);
  const box = selected.getBoundingClientRect();
  // Keep a selected module visible on phones without moving the surrounding page.
  if (box.width < viewport.clientWidth) {
    viewport.scrollLeft += box.left - viewport.getBoundingClientRect().left + box.width / 2 - viewport.clientWidth / 2;
  }
}

// Match the blog's existing iframe resize protocol. No external dependencies.
if (window.parent !== window) {
  let lastHeight = 0;
  new ResizeObserver(() => {
    const height = Math.ceil(document.body.getBoundingClientRect().height);
    if (height !== lastHeight) {
      lastHeight = height;
      window.parent.postMessage({ type: "vllm-embed-resize", height }, window.location.origin);
    }
  }).observe(document.body);
}
