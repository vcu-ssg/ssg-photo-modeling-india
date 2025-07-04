export async function loadSplatViewer(containerId, pathToSplatFile) {
  const SPLAT = await import("https://cdn.jsdelivr.net/npm/gsplat@1.2.3?module");

  const canvas = document.getElementById(containerId);
  if (!canvas) {
    console.error(`Canvas element with ID '${containerId}' not found.`);
    return;
  }

  const response = await fetch(pathToSplatFile);
  if (!response.ok) {
    console.error(`Failed to load .splat file from '${pathToSplatFile}'`);
    return;
  }

  const arrayBuffer = await response.arrayBuffer();
  const binaryData = new Uint8Array(arrayBuffer);

  const renderer = new SPLAT.WebGLRenderer(canvas);
  const scene = new SPLAT.Scene();
  const camera = new SPLAT.Camera();

  camera.position = new SPLAT.Vector3(5, 0, 0);  // move camera along Y axis
  

  const controls = new SPLAT.OrbitControls(camera, canvas);

  const splat = new SPLAT.Splat(SPLAT.SplatData.Deserialize(binaryData));


  scene.addObject(splat);

  function resize() {
    renderer.setSize(canvas.clientWidth, canvas.clientHeight);
  }
  window.addEventListener("resize", resize);
  resize();

  function animate() {
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }

  animate();
}
