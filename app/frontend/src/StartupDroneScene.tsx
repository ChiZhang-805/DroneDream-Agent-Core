import { useEffect, useRef } from "react";
import * as THREE from "three";

function cylinderBetween(
  start: THREE.Vector3,
  end: THREE.Vector3,
  radius: number,
  material: THREE.Material,
) {
  const direction = new THREE.Vector3().subVectors(end, start);
  const mesh = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, direction.length(), 18),
    material,
  );
  mesh.position.copy(start).add(end).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    direction.clone().normalize(),
  );
  return mesh;
}

function createDrone(accent: THREE.Color) {
  const drone = new THREE.Group();
  const carbon = new THREE.MeshPhysicalMaterial({
    color: 0x15131a,
    metalness: 0.72,
    roughness: 0.25,
    clearcoat: 0.8,
    clearcoatRoughness: 0.16,
  });
  const graphite = new THREE.MeshStandardMaterial({
    color: 0x39333d,
    metalness: 0.62,
    roughness: 0.32,
  });
  const metal = new THREE.MeshStandardMaterial({
    color: 0x827781,
    metalness: 0.88,
    roughness: 0.2,
  });
  const glow = new THREE.MeshBasicMaterial({ color: accent, toneMapped: false });
  const glass = new THREE.MeshPhysicalMaterial({
    color: 0x220913,
    emissive: 0x5c0b24,
    emissiveIntensity: 0.72,
    metalness: 0.15,
    roughness: 0.06,
    transmission: 0.18,
  });

  const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.52, 0.92, 10, 28), carbon);
  body.rotation.x = Math.PI / 2;
  body.scale.set(1.08, 0.7, 1.26);
  drone.add(body);

  const shell = new THREE.Mesh(new THREE.SphereGeometry(0.72, 36, 20), graphite);
  shell.scale.set(1.02, 0.38, 1.18);
  shell.position.set(0, 0.29, -0.04);
  drone.add(shell);

  const accentRail = new THREE.Mesh(new THREE.BoxGeometry(0.72, 0.025, 0.2), glow);
  accentRail.position.set(0, 0.51, -0.18);
  drone.add(accentRail);

  const motorPositions = [
    new THREE.Vector3(-1.62, 0.05, -1.3),
    new THREE.Vector3(1.62, 0.05, -1.3),
    new THREE.Vector3(-1.62, 0.05, 1.3),
    new THREE.Vector3(1.62, 0.05, 1.3),
  ];
  const rotors: THREE.Group[] = [];
  motorPositions.forEach((position, index) => {
    const elbow = position.clone().multiplyScalar(0.68);
    elbow.y = 0.08;
    drone.add(cylinderBetween(position.clone().multiplyScalar(0.28), elbow, 0.1, carbon));
    drone.add(cylinderBetween(elbow, position, 0.082, graphite));

    const motor = new THREE.Mesh(new THREE.CylinderGeometry(0.19, 0.22, 0.3, 24), carbon);
    motor.position.copy(position);
    motor.position.y += 0.14;
    drone.add(motor);

    const rotor = new THREE.Group();
    rotor.position.copy(position);
    rotor.position.y += 0.34;
    const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.095, 0.1, 18), metal);
    rotor.add(hub);
    const bladeMaterial = new THREE.MeshPhysicalMaterial({
      color: index < 2 ? 0x68606d : 0x49434e,
      transparent: true,
      opacity: 0.78,
      metalness: 0.46,
      roughness: 0.25,
    });
    const blade = new THREE.Mesh(new THREE.BoxGeometry(1.48, 0.026, 0.11), bladeMaterial);
    blade.position.y = 0.05;
    rotor.add(blade);
    const second = blade.clone();
    second.rotation.y = Math.PI / 2;
    rotor.add(second);
    const wash = new THREE.Mesh(
      new THREE.RingGeometry(0.34, 0.78, 64),
      new THREE.MeshBasicMaterial({
        color: accent,
        transparent: true,
        opacity: index % 2 === 0 ? 0.18 : 0.11,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    );
    wash.rotation.x = -Math.PI / 2;
    wash.position.y = -0.04;
    rotor.add(wash);
    rotors.push(rotor);
    drone.add(rotor);
  });

  const gimbal = new THREE.Mesh(new THREE.BoxGeometry(0.44, 0.34, 0.38), graphite);
  gimbal.position.set(0, -0.68, 0.75);
  drone.add(gimbal);
  const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.15, 0.1, 28), glass);
  lens.rotation.x = Math.PI / 2;
  lens.position.set(0, -0.68, 0.98);
  drone.add(lens);

  const landingMaterial = new THREE.MeshStandardMaterial({
    color: 0x26212a,
    metalness: 0.5,
    roughness: 0.38,
  });
  [-0.62, 0.62].forEach((x) => {
    const front = new THREE.Vector3(x, -0.45, 0.54);
    const back = new THREE.Vector3(x, -0.45, -0.52);
    const frontFoot = new THREE.Vector3(x, -0.98, 0.72);
    const backFoot = new THREE.Vector3(x, -0.98, -0.7);
    drone.add(cylinderBetween(front, frontFoot, 0.045, landingMaterial));
    drone.add(cylinderBetween(back, backFoot, 0.045, landingMaterial));
    drone.add(cylinderBetween(frontFoot, backFoot, 0.055, landingMaterial));
  });

  drone.traverse((object) => {
    if (object instanceof THREE.Mesh) {
      object.castShadow = true;
      object.receiveShadow = true;
    }
  });
  return { drone, rotors };
}

export function StartupDroneScene({ progress }: { progress: number }) {
  const hostRef = useRef<HTMLDivElement>(null);
  const progressRef = useRef(progress);
  progressRef.current = progress;

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return undefined;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(34, 1, 0.1, 100);
    camera.position.set(0, 2.2, 7.6);
    camera.lookAt(0, 0, 0);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.12;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(renderer.domElement);

    const accent = new THREE.Color(0xe3264f);
    const hotPink = new THREE.Color(0xff6a88);
    const { drone, rotors } = createDrone(accent);
    drone.rotation.set(-0.16, -0.45, 0.02);
    scene.add(drone);

    scene.add(new THREE.HemisphereLight(0xffeff3, 0x20151c, 2.7));
    const key = new THREE.DirectionalLight(0xffffff, 4.8);
    key.position.set(3.5, 6, 5);
    key.castShadow = true;
    scene.add(key);
    const rim = new THREE.PointLight(0xe3264f, 36, 12, 2);
    rim.position.set(-3, 1.4, -1.8);
    scene.add(rim);
    const fill = new THREE.PointLight(0xff91a8, 20, 10, 2);
    fill.position.set(3.2, -0.5, 2.5);
    scene.add(fill);

    const ringMaterial = new THREE.MeshBasicMaterial({
      color: accent,
      transparent: true,
      opacity: 0.24,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const ring = new THREE.Mesh(new THREE.RingGeometry(2.9, 2.93, 128), ringMaterial);
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = -1.22;
    scene.add(ring);
    const innerRing = new THREE.Mesh(new THREE.RingGeometry(1.9, 1.915, 128), ringMaterial.clone());
    (innerRing.material as THREE.MeshBasicMaterial).opacity = 0.12;
    innerRing.rotation.x = -Math.PI / 2;
    innerRing.position.y = -1.21;
    scene.add(innerRing);

    const particleCount = 720;
    const positions = new Float32Array(particleCount * 3);
    for (let index = 0; index < particleCount; index += 1) {
      const radius = 2.5 + Math.random() * 2.8;
      const angle = Math.random() * Math.PI * 2;
      positions[index * 3] = Math.cos(angle) * radius;
      positions[index * 3 + 1] = (Math.random() - 0.5) * 3.8;
      positions[index * 3 + 2] = Math.sin(angle) * radius;
    }
    const particleGeometry = new THREE.BufferGeometry();
    particleGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const particles = new THREE.Points(
      particleGeometry,
      new THREE.PointsMaterial({
        color: hotPink,
        size: 0.026,
        transparent: true,
        opacity: 0.46,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    );
    scene.add(particles);

    const pointer = new THREE.Vector2();
    const target = new THREE.Vector2();
    const onPointerMove = (event: PointerEvent) => {
      const bounds = host.getBoundingClientRect();
      target.set(
        ((event.clientX - bounds.left) / bounds.width - 0.5) * 2,
        ((event.clientY - bounds.top) / bounds.height - 0.5) * 2,
      );
    };
    host.addEventListener("pointermove", onPointerMove);

    const resize = () => {
      const width = Math.max(1, host.clientWidth);
      const height = Math.max(1, host.clientHeight);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const clock = new THREE.Clock();
    let frame = 0;
    const render = () => {
      const elapsed = clock.getElapsedTime();
      pointer.lerp(target, 0.035);
      const readiness = progressRef.current / 100;
      const rotorSpeed = reducedMotion ? 0 : 0.12 + readiness * 0.45;
      rotors.forEach((rotor, index) => {
        rotor.rotation.y += (index % 2 ? -1 : 1) * rotorSpeed;
      });
      if (!reducedMotion) {
        drone.position.y = Math.sin(elapsed * 1.05) * 0.08;
        drone.rotation.y = -0.45 + Math.sin(elapsed * 0.34) * 0.08 + pointer.x * 0.12;
        drone.rotation.x = -0.16 + pointer.y * 0.07;
        particles.rotation.y = elapsed * 0.035;
        particles.rotation.x = Math.sin(elapsed * 0.2) * 0.05;
        ring.rotation.z = elapsed * 0.06;
        innerRing.rotation.z = -elapsed * 0.045;
      }
      (ring.material as THREE.MeshBasicMaterial).opacity = 0.12 + readiness * 0.2;
      renderer.render(scene, camera);
      frame = requestAnimationFrame(render);
    };
    render();

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      host.removeEventListener("pointermove", onPointerMove);
      scene.traverse((object) => {
        if (!(object instanceof THREE.Mesh || object instanceof THREE.Points)) return;
        object.geometry?.dispose();
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        materials.forEach((material) => material.dispose());
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  return <div ref={hostRef} className="startup-drone-scene" aria-hidden="true" />;
}
