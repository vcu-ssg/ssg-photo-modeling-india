export async function loadGLBViewer(containerId, glbPath, texturePath, {
  mode = "textured",
  metalness = 0.3,
  roughness = 0.7
} = {}) {
  const THREE = await import('https://unpkg.com/three@0.165.0/build/three.module.js');
  const { OrbitControls } = await import('https://unpkg.com/three@0.165.0/examples/jsm/controls/OrbitControls.js?module');
  const { GLTFLoader } = await import('https://unpkg.com/three@0.165.0/examples/jsm/loaders/GLTFLoader.js?module');

  const container = document.getElementById(containerId);
  if (!container) {
    console.error(`Container with ID '${containerId}' not found.`);
    return;
  }

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf0f0f0);

  const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  scene.add(new THREE.AmbientLight(0xffffff,1.5));
  const directionalLight = new THREE.DirectionalLight(0xffffff, 2.0);
  directionalLight.position.set(5, 10, 5);
  scene.add(directionalLight);

  // Load the texture
  let texture;
  if (mode === "textured") {
    const textureLoader = new THREE.TextureLoader();
    try {
      texture = await textureLoader.loadAsync(texturePath);
      texture.flipY = false; // important for GL convention
      texture.encoding = THREE.sRGBEncoding;
    } catch (err) {
      console.error(`❌ Failed to load texture from ${texturePath}`, err);
      return;
    }
  }

  // Load the GLB
  const loader = new GLTFLoader();
  loader.setPath(glbPath.substring(0, glbPath.lastIndexOf('/') + 1));

  loader.load(glbPath.split('/').pop(), (gltf) => {
    const model = gltf.scene;

    model.traverse((child) => {
      if (child.isMesh) {
        child.geometry = child.geometry.clone(); // clone to avoid VAO reuse bugs

        if (mode === "wireframe") {
          if (child.material) child.material.dispose();
          child.material = new THREE.MeshBasicMaterial({
            color: 0x000000,
            wireframe: true
          });

        } else if (mode === "shaded") {
          if (child.material) child.material.dispose();
          child.material = new THREE.MeshStandardMaterial({
            color: 0x999999,
            metalness: metalness,
            roughness: roughness
          });

        } else if (mode === "textured" && texture) {
          if (child.material) child.material.dispose();
          child.material = new THREE.MeshStandardMaterial({
            map: texture,
            metalness: metalness,
            roughness: roughness
          });


          // Make sure texture wrapping and UVs are correct
          texture.wrapS = THREE.RepeatWrapping;
          texture.wrapT = THREE.RepeatWrapping;
          texture.offset.set(0,0);
          texture.repeat.set(1,1);
          texture.needsUpdate = true;

          if (child.geometry.attributes.uv) {
            child.geometry.attributes.uv.needsUpdate = true;
            child.geometry.uvsNeedUpdate = true;
          }
          if (child.geometry.attributes.uv) {
            const uvAttr = child.geometry.attributes.uv;
            const uvs = [];
            for (let i = 0; i < Math.min(uvAttr.count, 100); i++) {
              uvs.push([uvAttr.getX(i), uvAttr.getY(i)]);
            }
            console.log("Sample UVs:", uvs);
          }
        }

        console.log("✅ Mesh:", child);
        console.log("✅ Material:", child.material);
        console.log("✅ UVs:", child.geometry.attributes.uv);

        child.material.needsUpdate = true;
      }
    });

    model.scale.y *= -1;


    scene.add(model);

    // Center & zoom
    const box = new THREE.Box3().setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const radius = size.length() * 0.5;

    model.position.sub(center);
    camera.position.set(radius * 1.5, radius * 1.5, radius * 1.5);
    controls.target.set(0, 0, 0);
    controls.update();
  }, undefined, (err) => {
    console.error(`❌ Failed to load GLB file at ${glbPath}:`, err);
  });

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }

  animate();

  window.addEventListener("resize", () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });
}
