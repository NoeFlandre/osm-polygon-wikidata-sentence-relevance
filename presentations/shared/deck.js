(() => {
  const slides = [...document.querySelectorAll(".slide")];
  const counter = document.querySelector(".counter");
  let current = 0;

  function show(index) {
    current = Math.max(0, Math.min(index, slides.length - 1));
    slides.forEach((slide, i) => slide.classList.toggle("active", i === current));
    counter.textContent = `${current + 1} / ${slides.length}`;
    document.title = `${document.body.dataset.title} · ${current + 1}/${slides.length}`;
  }

  document.querySelector("[data-action=prev]").addEventListener("click", () => show(current - 1));
  document.querySelector("[data-action=next]").addEventListener("click", () => show(current + 1));
  document.addEventListener("keydown", (event) => {
    if (["ArrowRight", "PageDown", " "].includes(event.key)) show(current + 1);
    if (["ArrowLeft", "PageUp"].includes(event.key)) show(current - 1);
    if (event.key === "Home") show(0);
    if (event.key === "End") show(slides.length - 1);
  });
  show(0);
})();
