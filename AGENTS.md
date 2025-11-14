# Agent Guidelines for This Repo

- This repo contains teaching materials for an internal AI / ML course (slides + notebooks) aimed at practitioners with basic Python and data skills, not research specialists.
- When editing `slides/`:
  - Favor clear, visual explanations (diagrams, simple equations, short code) over dense theory.
  - Keep tone professional and concise; avoid jokes or overly casual language in the slide content itself.
  - Prefer PyTorch for code examples; keep examples small enough to run quickly on a laptop.
  - Put reusable graphics under `slides/assets/` and reference them from the slides rather than inlining large SVG snippets.
  - For network/flow diagrams, prefer:
    - Graph structure: DOT/Graphviz snippets rendered via Viz.js, saved as `.svg` or `.png` in `slides/assets/`.
    - Function plots (e.g., ReLU, sigmoid): generate with `matplotlib` in the `venv` and save to `slides/assets/` as `.png`.
    - Keep diagrams simple (2–3 layers, small node counts) and sized to fit a single slide without overflow.
- When editing `notebooks/`:
  - Mirror the conceptual flow of the corresponding lesson’s slides.
  - Keep cells short and focused; prioritize readability for learners stepping through execution.
  - Use the same conventions as the slides (ReLU in hidden layers, sigmoid for binary outputs in Lesson 1, PyTorch instead of Keras).
- In general, optimize for teaching clarity and consistency across slides and notebooks, not for maximal performance or advanced tricks. 
