// Shortcut overlay shared by the editor pages. Each page passes its own list;
// everything else -- the button, the "?" and F1 keys, Escape to close, and
// swallowing keystrokes so the editor underneath doesn't act on them while the
// overlay is up -- is the same everywhere and lives here.
//
// sections: [{ title, rows: [[keys, what it does], ...] }]
// `keys` is split on "+" and each part rendered as its own <kbd>, so
// "Ctrl + drag" comes out as two keycaps and "double-click" as one.
function initShortcutHelp(button, sections) {
  let backdrop = null;

  function build() {
    backdrop = document.createElement("div");
    backdrop.className = "gw-help-backdrop";
    backdrop.hidden = true;

    const panel = document.createElement("div");
    panel.className = "gw-help-panel";
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-modal", "true");
    panel.setAttribute("aria-label", "Keyboard and mouse shortcuts");

    const head = document.createElement("div");
    head.className = "gw-help-head";
    const title = document.createElement("h2");
    title.textContent = "Keyboard & mouse shortcuts";
    const close = document.createElement("button");
    close.type = "button";
    close.className = "btn btn-outline-secondary btn-sm";
    close.textContent = "Close";
    close.addEventListener("click", () => toggle(false));
    head.append(title, close);
    panel.appendChild(head);

    sections.forEach((section) => {
      const heading = document.createElement("div");
      heading.className = "gw-help-section";
      heading.textContent = section.title;
      panel.appendChild(heading);

      const table = document.createElement("table");
      section.rows.forEach(([keys, description]) => {
        const tr = document.createElement("tr");
        const keyCell = document.createElement("td");
        keys
          .split("+")
          .map((part) => part.trim())
          .filter(Boolean)
          .forEach((part, i) => {
            if (i) keyCell.append(" + ");
            const kbd = document.createElement("kbd");
            kbd.textContent = part;
            keyCell.appendChild(kbd);
          });
        const textCell = document.createElement("td");
        textCell.textContent = description;
        tr.append(keyCell, textCell);
        table.appendChild(tr);
      });
      panel.appendChild(table);
    });

    backdrop.appendChild(panel);
    // Clicking the dimmed area closes; clicking the panel itself doesn't.
    backdrop.addEventListener("mousedown", (e) => {
      if (e.target === backdrop) toggle(false);
    });
    document.body.appendChild(backdrop);
  }

  function isOpen() {
    return backdrop && !backdrop.hidden;
  }

  function toggle(open) {
    if (!backdrop) build();
    backdrop.hidden = !(open === undefined ? backdrop.hidden : open);
    if (button) button.setAttribute("aria-expanded", String(!backdrop.hidden));
  }

  if (button) button.addEventListener("click", () => toggle());

  // Capture phase: while the overlay is up, the page underneath must not see
  // Delete, arrows or anything else the user aims at the dialog.
  window.addEventListener(
    "keydown",
    (e) => {
      const tag = (e.target.tagName || "").toLowerCase();
      const typing = tag === "input" || tag === "textarea" || tag === "select";
      if (isOpen()) {
        e.stopPropagation();
        if (e.key === "Escape" || e.key === "?" || e.key === "F1") {
          e.preventDefault();
          toggle(false);
        }
        return;
      }
      if (typing) return;
      if (e.key === "?" || e.key === "F1") {
        e.preventDefault();
        e.stopPropagation();
        toggle(true);
      }
    },
    true
  );

  return { open: () => toggle(true), close: () => toggle(false) };
}
