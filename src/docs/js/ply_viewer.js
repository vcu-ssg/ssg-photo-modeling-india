export async function loadPLYViewer(containerId, plyPath) {
  const SAMPLE_COUNT = 20000;
  const CAMERA_DISTANCE_SCALE = 5;

  const container = document.getElementById(containerId);
  if (!container) {
    console.error(`Container with ID '${containerId}' not found.`);
    return;
  }

  const THREE = await import('https://unpkg.com/three@0.165.0/build/three.module.js');
  const { PLYLoader } = await import('https://unpkg.com/three@0.165.0/examples/jsm/loaders/PLYLoader.js?module');
  const { OrbitControls } = await import('https://unpkg.com/three@0.165.0/examples/jsm/controls/OrbitControls.js?module');

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf0f0f0);

  const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.screenSpacePanning = false;
  controls.minDistance = 0.1;
  controls.maxDistance = 500;

  scene.add(new THREE.AmbientLight(0x999999));
  const light = new THREE.DirectionalLight(0xffffff, 0.8);
  light.position.set(5, 10, 5);
  scene.add(light);

  const axisRings = [];
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let selectedAxis = null;
  let isDragging = false;
  let prevPointerX = null;

  function createAxisCircle(radius, segments, color, rotation, axisName) {
    const points = [];
    for (let i = 0; i <= segments; i++) {
      const theta = (i / segments) * Math.PI * 2;
      points.push(new THREE.Vector3(radius * Math.cos(theta), radius * Math.sin(theta), 0));
    }
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ color });
    const circle = new THREE.LineLoop(geometry, material);
    if (rotation) circle.rotation.set(rotation.x, rotation.y, rotation.z);
    circle.userData.axis = axisName;
    axisRings.push(circle);
    return circle;
  }

  const loader = new PLYLoader();
  let modelGroup = null;

  loader.load(plyPath, (geometry) => {
    const positionAttr = geometry.attributes.position;
    const count = positionAttr.count;

    // Compute center of mass
    const center = new THREE.Vector3();
    for (let i = 0; i < count; i++) {
      center.x += positionAttr.getX(i);
      center.y += positionAttr.getY(i);
      center.z += positionAttr.getZ(i);
    }
    center.divideScalar(count);

    // Compute RMS distance
    const indices = Array.from({ length: count }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    const sampleIndices = indices.slice(0, Math.min(SAMPLE_COUNT, count));
    let sumSq = 0;
    for (const i of sampleIndices) {
      const dx = positionAttr.getX(i) - center.x;
      const dy = positionAttr.getY(i) - center.y;
      const dz = positionAttr.getZ(i) - center.z;
      sumSq += dx * dx + dy * dy + dz * dz;
    }
    const rms = Math.sqrt(sumSq / sampleIndices.length);

    // Center and render points
    const material = new THREE.PointsMaterial({
      size: 0.005,
      vertexColors: true,
    });

    const points = new THREE.Points(geometry, material);
    points.position.sub(center);

    modelGroup = new THREE.Group();
    modelGroup.add(points);
    scene.add(modelGroup);

    // Axis rings
    const radius = rms * 1.2;
    const segments = 64;
    modelGroup.add(createAxisCircle(radius, segments, 0xff0000, new THREE.Euler(0, Math.PI / 2, 0), 'x'));
    modelGroup.add(createAxisCircle(radius, segments, 0x00ff00, new THREE.Euler(Math.PI / 2, 0, 0), 'y'));
    modelGroup.add(createAxisCircle(radius, segments, 0x0000ff, new THREE.Euler(0, 0, 0), 'z'));

    // Set camera
    camera.position.set(center.x + rms * CAMERA_DISTANCE_SCALE, center.y, center.z + rms * CAMERA_DISTANCE_SCALE);
    controls.target.set(0, 0, 0);
    controls.update();
  });

  // === Rotation logic ===
  const spinDirections = { local: { x: 1, y: 1, z: 1 }, global: { x: 1, y: 1, z: 1 } };
  const spinning = { local: { x: false, y: false, z: false }, global: { x: false, y: false, z: false } };
  const keys = {};

  window.addEventListener('keydown', (e) => {
    const k = e.key.toLowerCase();
    const isShift = e.shiftKey;

    if (!['x', 'y', 'z'].includes(k)) {
      if (k === 's') autoShowcase = !autoShowcase;
      keys[k] = true;
      return;
    }

    const mode = isShift ? 'global' : 'local';
    if (!spinning[mode][k]) {
      spinDirections[mode][k] *= -1;
      spinning[mode][k] = true;
    }
  });

  window.addEventListener('keyup', (e) => {
    const k = e.key.toLowerCase();
    const isShift = e.shiftKey;

    if (!['x', 'y', 'z'].includes(k)) {
      keys[k] = false;
      return;
    }

    const mode = isShift ? 'global' : 'local';
    spinning[mode][k] = false;
  });

  // === Mouse interaction for axis rings ===
  renderer.domElement.addEventListener('pointerdown', (event) => {
    pointer.x = (event.clientX / container.clientWidth) * 2 - 1;
    pointer.y = -(event.clientY / container.clientHeight) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const intersects = raycaster.intersectObjects(axisRings, false);
    if (intersects.length > 0) {
      selectedAxis = intersects[0].object.userData.axis;
      prevPointerX = event.clientX;
      isDragging = true;
      controls.enabled = false;
    }
  });

  renderer.domElement.addEventListener('pointermove', (event) => {
    if (!isDragging || !selectedAxis || !modelGroup) return;
    const delta = event.clientX - prevPointerX;
    prevPointerX = event.clientX;
    const angle = delta * 0.005;
    if (selectedAxis === 'x') modelGroup.rotation.x += angle;
    if (selectedAxis === 'y') modelGroup.rotation.y += angle;
    if (selectedAxis === 'z') modelGroup.rotation.z += angle;
  });

  renderer.domElement.addEventListener('pointerup', () => {
    isDragging = false;
    selectedAxis = null;
    controls.enabled = true;
  });

  // === Showcase animation ===
  let autoShowcase = true;
  const showcaseSpeed = 0.025;

  function animate() {
    requestAnimationFrame(animate);

    if (modelGroup) {
      const rotSpeed = 0.08;

      if (spinning.local.x) modelGroup.rotation.x += rotSpeed * spinDirections.local.x;
      if (spinning.local.y) modelGroup.rotation.y += rotSpeed * spinDirections.local.y;
      if (spinning.local.z) modelGroup.rotation.z += rotSpeed * spinDirections.local.z;

      if (spinning.global.x) modelGroup.rotateOnWorldAxis(new THREE.Vector3(1, 0, 0), rotSpeed * spinDirections.global.x);
      if (spinning.global.y) modelGroup.rotateOnWorldAxis(new THREE.Vector3(0, 1, 0), rotSpeed * spinDirections.global.y);
      if (spinning.global.z) modelGroup.rotateOnWorldAxis(new THREE.Vector3(0, 0, 1), rotSpeed * spinDirections.global.z);

      if (autoShowcase) {
        modelGroup.rotation.x += showcaseSpeed;
        modelGroup.rotation.y += showcaseSpeed * 0.7;
        modelGroup.rotation.z += showcaseSpeed * 0.3;
      }

      const zoomSpeed = 0.1;
      if (keys['arrowup'] || keys['='] || keys[']']) camera.position.z -= zoomSpeed;
      if (keys['arrowdown'] || keys['-'] || keys['[']) camera.position.z += zoomSpeed;
    }

    controls.update();
    renderer.render(scene, camera);
  }

  animate();

  window.addEventListener('resize', () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });
}
