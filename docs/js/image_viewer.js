export function loadImageViewer(containerId, images, fps = 2) {
  const container = document.getElementById(containerId);
  if (!container) {
    console.error(`Container '${containerId}' not found.`);
    return;
  }

  let current = 0;
  let playing = false;
  let interval = null;

  container.innerHTML = `
    <div style="display:flex; align-items:center; justify-content:center;">
      <button id="prevBtn" style="font-size:2em;">◀</button>
      <img id="viewerImage" src="${images[current]}" style="max-width:80%; max-height:80%;">
      <button id="nextBtn" style="font-size:2em;">▶</button>
      <button id="playBtn" style="font-size:1.5em; margin-left:10px;">Play</button>
    </div>
  `;

  const img = container.querySelector('#viewerImage');
  const playBtn = container.querySelector('#playBtn');

  function updateImage(newIndex) {
    current = (newIndex + images.length) % images.length;
    img.src = images[current];
  }

  function start() {
    playing = true;
    playBtn.textContent = "Pause";
    interval = setInterval(() => {
      updateImage(current + 1);
    }, 1000 / fps);
  }

  function stop() {
    playing = false;
    playBtn.textContent = "Play";
    clearInterval(interval);
  }

  container.querySelector('#prevBtn').addEventListener('click', () => {
    updateImage(current - 1);
    stop();
  });

  container.querySelector('#nextBtn').addEventListener('click', () => {
    updateImage(current + 1);
    stop();
  });

  playBtn.addEventListener('click', () => {
    playing ? stop() : start();
  });

  window.addEventListener('keydown', (e) => {
    if (!container.matches(':hover')) return;
    if (e.key === 'ArrowRight') { updateImage(current + 1); stop(); }
    if (e.key === 'ArrowLeft') { updateImage(current - 1); stop(); }
    if (e.key === ' ') {
      e.preventDefault();
      playing ? stop() : start();
    }
  });

  console.log(`Simple image player loaded with ${images.length} frames at ${fps} fps.`);
}
