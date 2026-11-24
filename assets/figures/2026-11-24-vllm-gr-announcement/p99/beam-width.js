// With no script, the two controls remain ordinary links to the figures.
const overview = document.getElementById("gr-p99-overview");
if (overview) {
  const panel = overview.querySelector("img");
  const links = [...overview.querySelectorAll("p a[href]")];
  if (panel && links.length === 2 && links.some(link => link.href === panel.src)) {
    const group = document.createElement("div");
    group.className = "beam-width-controls";
    group.setAttribute("role", "group");
    group.setAttribute("aria-label", "Beam width");
    panel.id = "gr-p99-chart";

    const choices = links.map(link => {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = link.textContent;
      button.setAttribute("aria-controls", panel.id);
      button.setAttribute("aria-pressed", String(link.href === panel.src));
      group.append(button);
      return { link, button };
    });

    function select(choice) {
      for (const item of choices) {
        item.button.setAttribute("aria-pressed", String(item === choice));
      }
      panel.src = choice.link.href;
      panel.alt = choice.link.dataset.chartAlt;
    }

    for (const [index, choice] of choices.entries()) {
      choice.button.addEventListener("click", () => select(choice));
      choice.button.addEventListener("keydown", event => {
        const targets = new Map([
          ["ArrowLeft", (index + choices.length - 1) % choices.length],
          ["ArrowRight", (index + 1) % choices.length],
          ["Home", 0],
          ["End", choices.length - 1],
        ]);
        const next = targets.get(event.key);
        if (next === undefined) return;
        event.preventDefault();
        choices[next].button.focus();
        select(choices[next]);
      });
    }
    links[0].parentElement.replaceWith(group);
  }
}
