"use client";

import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
} from "react";

import * as THREE from "three";

import { TrackballControls } from
  "three/addons/controls/TrackballControls.js";

import { PLYLoader } from
  "three/addons/loaders/PLYLoader.js";

import type { SpatialResult } from "@/lib/types";

export type PointCloudControls = {
  reset: () => void;
  focusSelected: () => void;
};

type Props = {
  data: ArrayBuffer;
  pointSize: number;
  overlay?: SpatialResult | null;
  selectedObjectId?: string | null;
  showFloor?: boolean;
  alignToFloor?: boolean;
};

type WorldTransform = {
  center: THREE.Vector3;
  scale: number;
};


function isXYZ(value: number[] | undefined): value is [number, number, number] {
  return Boolean(value && value.length === 3 && value.every(Number.isFinite));
}

// Pick the normal that points into the reconstructed room, not toward an
// arbitrary Pi3X coordinate axis. Floor points themselves do not vote.
function roomFacingFloor(
  plane: NonNullable<SpatialResult["floor"]>,
  geometry: THREE.BufferGeometry,
  world: WorldTransform
) {
  const normal = new THREE.Vector3(...plane.normal);
  const length = normal.length();
  normal.divideScalar(length);
  let offset = plane.offset / length;
  let displayOffset = world.scale * (normal.dot(world.center) + offset);
  const positions = geometry.getAttribute("position");
  const diagonal = geometry.boundingBox!.getSize(new THREE.Vector3()).length();
  const residual = Number.isFinite(plane.rms_residual) ? Math.abs(plane.rms_residual) : 0;
  const band = Math.max(diagonal * 0.002, residual * world.scale * 3, 1e-6);
  const distances: number[] = [];
  const step = Math.max(1, Math.ceil(positions.count / 12000));
  for (let i = 0; i < positions.count; i += step) {
    const distance = normal.x * positions.getX(i) +
      normal.y * positions.getY(i) + normal.z * positions.getZ(i) + displayOffset;
    if (Number.isFinite(distance) && Math.abs(distance) > band) distances.push(distance);
  }
  distances.sort((a, b) => a - b);
  // Median rejects isolated below-floor artifacts and ignores dense floor samples.
  // If the median straddles zero, use the robust extent on each side.
  let side = 0;
  if (distances.length) {
    const middle = Math.floor(distances.length / 2);
    side = distances.length % 2 ? distances[middle] :
      (distances[middle - 1] + distances[middle]) / 2;
    if (Math.abs(side) <= band) {
      side = distances[Math.floor((distances.length - 1) * 0.1)] +
        distances[Math.ceil((distances.length - 1) * 0.9)];
    }
  }
  // An entirely flat cloud has no geometric evidence for either side.
  if (side < 0 || (Math.abs(side) <= 1e-10 && normal.y < 0)) {
    normal.negate();
    offset = -offset;
    displayOffset = -displayOffset;
  }
  return { normal, offset, displayOffset };
}

function fitCamera(
  camera: THREE.PerspectiveCamera,
  controls: TrackballControls,
  bounds: THREE.Box3,
  resetDirection = false
) {
  const center = bounds.getCenter(new THREE.Vector3());
  const radius = Math.max(bounds.getSize(new THREE.Vector3()).length() / 2, 0.04);
  const verticalHalfFov = THREE.MathUtils.degToRad(camera.fov / 2);
  const horizontalHalfFov = Math.atan(Math.tan(verticalHalfFov) * camera.aspect);
  const distance = radius * 1.2 / Math.sin(Math.min(verticalHalfFov, horizontalHalfFov));
  const direction = resetDirection ? new THREE.Vector3(6.4, 3.5, 7.8) :
    camera.position.clone().sub(controls.target);
  if (direction.lengthSq() < 1e-10) direction.set(6.4, 3.5, 7.8);
  direction.normalize();
  // Clear Trackball's pending zoom/pan before assigning a new target.
  controls.reset();
  camera.up.set(0, 1, 0);
  controls.target.copy(center);
  camera.position.copy(center).addScaledVector(direction, distance);
  controls.minDistance = Math.min(0.08, distance * 0.1);
  controls.maxDistance = Math.max(140, distance * 4);
  camera.near = Math.max(0.0001, Math.min(0.01, distance / 1000));
  camera.far = Math.max(3000, distance * 10);
  camera.updateProjectionMatrix();
  camera.lookAt(center);
  controls.update();
}

function disposeOverlay(group: THREE.Group) {
  for (const object of [...group.children]) {
    group.remove(object);

    if (
      object instanceof THREE.Points ||
      object instanceof THREE.Mesh
    ) {
      object.geometry.dispose();

      if (Array.isArray(object.material)) {
        object.material.forEach((m) => m.dispose());
      } else {
        object.material.dispose();
      }
    }
  }
}

export const PointCloudViewer =
  forwardRef<PointCloudControls, Props>(
    function PointCloudViewer(
      {
        data,
        pointSize,
        overlay = null,
        selectedObjectId = null,
        showFloor = true,
        alignToFloor = false,
      },
      ref
    ) {
      const mountRef = useRef<HTMLDivElement>(null);

      const materialRef =
        useRef<THREE.PointsMaterial | null>(null);

      const overlayRef =
        useRef<THREE.Group | null>(null);

      const worldRef =
        useRef<WorldTransform | null>(null);

      const sceneState = useRef<{
        root: THREE.Group;
        grid: THREE.GridHelper;
        camera: THREE.PerspectiveCamera;
        controls: TrackballControls;
        geometry: THREE.BufferGeometry;
      } | null>(null);
      const focusRef = useRef<() => void>(() => {});

      const resetRef =
        useRef<() => void>(() => {});

      useImperativeHandle(
        ref,
        () => ({
          reset: () => resetRef.current(),
          focusSelected: () => focusRef.current(),
        }),
        []
      );

      useEffect(() => {
        if (materialRef.current) {
          materialRef.current.size = pointSize;
          materialRef.current.needsUpdate = true;
        }
      }, [pointSize]);

      // --------------------------------------------------
      // BASE 3D SCENE
      // --------------------------------------------------

      useEffect(() => {
        const mount = mountRef.current;
        if (!mount) return;

        const scene = new THREE.Scene();

        scene.background = new THREE.Color(
          "#171717"
        );

        const camera =
          new THREE.PerspectiveCamera(
            46,
            Math.max(mount.clientWidth, 1) /
              Math.max(mount.clientHeight, 1),
            0.01,
            3000
          );

        camera.position.set(6.4, 3.5, 7.8);

        const renderer = new THREE.WebGLRenderer({
          antialias: true,
          powerPreference: "high-performance",
        });

        renderer.setPixelRatio(
          Math.min(window.devicePixelRatio || 1, 2)
        );

        renderer.outputColorSpace =
          THREE.SRGBColorSpace;

        renderer.setSize(
          Math.max(mount.clientWidth, 1),
          Math.max(mount.clientHeight, 1)
        );

        mount.appendChild(renderer.domElement);

        const controls = new TrackballControls(
          camera,
          renderer.domElement
        );

        controls.staticMoving = true;
        controls.rotateSpeed = 2;
        controls.zoomSpeed = 1.2;
        controls.panSpeed = 0.3;
        controls.minDistance = 0.08;
        controls.maxDistance = 140;

        // Base geometry and semantic overlays share one display-only transform.
        const root = new THREE.Group();
        scene.add(root);
        const overlays = new THREE.Group();
        root.add(overlays);
        const grid = new THREE.GridHelper(10, 20, 0x639c77, 0x384d40);
        grid.visible = false;
        scene.add(grid);

        overlayRef.current = overlays;

        let geometry:
          THREE.BufferGeometry | null = null;

        let material:
          THREE.PointsMaterial | null = null;

        let observer:
          ResizeObserver | null = null;

        let errorLabel:
          HTMLDivElement | null = null;

        let frame = 0;

        try {
          geometry = new PLYLoader().parse(
            data.slice(0)
          );

          if (!geometry.getAttribute("position")) {
            throw new Error(
              "PLY has no vertex positions."
            );
          }

          geometry.computeBoundingBox();

          const bounds = geometry.boundingBox;

          if (!bounds || bounds.isEmpty()) {
            throw new Error(
              "PLY contains no valid points."
            );
          }

          const size = bounds.getSize(
            new THREE.Vector3()
          );

          const center = bounds.getCenter(
            new THREE.Vector3()
          );

          const scale =
            5 /
            Math.max(
              size.x,
              size.y,
              size.z,
              0.0001
            );

          // The same transformation will be used
          // for SAM and RANSAC points.
          worldRef.current = {
            center,
            scale,
          };

          geometry.translate(
            -center.x,
            -center.y,
            -center.z
          );

          geometry.scale(
            scale,
            scale,
            scale
          );

          const vertexColors = Boolean(
            geometry.getAttribute("color")
          );

          // Keep the full original environment visible.
          material = new THREE.PointsMaterial({
            size: pointSize,
            sizeAttenuation: true,
            vertexColors,
            color: vertexColors
              ? 0xffffff
              : 0xc6b5f6,
            depthTest: true,
            depthWrite: true,
          });

          materialRef.current = material;

          const base = new THREE.Points(
            geometry,
            material
          );

          base.frustumCulled = false;

          root.add(base);
          geometry.computeBoundingBox();
          sceneState.current = {
            root, grid, camera, controls,
            geometry,
          };
          resetRef.current = () => {
            root.updateMatrixWorld(true);
            fitCamera(camera, controls,
              geometry!.boundingBox!.clone().applyMatrix4(root.matrixWorld), true);
          };
          resetRef.current();

          observer = new ResizeObserver(() => {
            if (
              !mount.clientWidth ||
              !mount.clientHeight
            ) {
              return;
            }

            camera.aspect =
              mount.clientWidth /
              mount.clientHeight;

            camera.updateProjectionMatrix();

            renderer.setSize(
              mount.clientWidth,
              mount.clientHeight
            );

            controls.handleResize();
          });

          observer.observe(mount);

          const animate = () => {
            controls.update();
            renderer.render(scene, camera);

            frame = requestAnimationFrame(
              animate
            );
          };

          animate();

        } catch (error) {
          errorLabel = document.createElement(
            "div"
          );

          errorLabel.className =
            "viewer-error-inline";

          errorLabel.textContent =
            error instanceof Error
              ? error.message
              : "Cannot display this point cloud.";

          mount.appendChild(errorLabel);
        }

        return () => {
          cancelAnimationFrame(frame);

          observer?.disconnect();
          controls.dispose();

          disposeOverlay(overlays);
          root.remove(overlays);
          grid.geometry.dispose();
          const gridMaterials = Array.isArray(grid.material) ? grid.material : [grid.material];
          gridMaterials.forEach((m) => m.dispose());
          sceneState.current = null;
          focusRef.current = () => {};

          overlayRef.current = null;
          worldRef.current = null;

          resetRef.current = () => {};

          geometry?.dispose();
          material?.dispose();

          materialRef.current = null;

          renderer.dispose();
          renderer.domElement.remove();

          errorLabel?.remove();
        };
      }, [data]);

      // Always derive the pose from the original normalized geometry.
      // Repeated toggles never accumulate rotations or translations.
      useEffect(() => {
        const state = sceneState.current;
        const world = worldRef.current;
        if (!state || !world) return;
        state.root.quaternion.identity();
        state.root.position.set(0, 0, 0);
        const plane = overlay?.floor;
        const validPlane = Boolean(plane && isXYZ(plane.normal) &&
          Math.hypot(...plane.normal) > 1e-8 && Number.isFinite(plane.offset));
        state.grid.visible = alignToFloor && validPlane;
        if (alignToFloor && plane && validPlane) {
          const { normal, displayOffset } = roomFacingFloor(plane, state.geometry, world);
          state.root.quaternion.setFromUnitVectors(normal, new THREE.Vector3(0, 1, 0));
          // After rotation, y = n.q. Translate the candidate floor to y = 0;
          // the room-facing half-space then lies above it.
          state.root.position.y = displayOffset;
        }
        state.root.updateMatrixWorld(true);
        resetRef.current();
      }, [data, overlay?.floor, alignToFloor]);

      useEffect(() => {
        focusRef.current = () => {
          const state = sceneState.current;
          const world = worldRef.current;
          const selected = overlay?.objects.find(
            (object) => String(object.object_id) === selectedObjectId);
          if (!state || !world || !selected) return;
          const box = new THREE.Box3();
          if (isXYZ(selected.bbox_min_xyz) && isXYZ(selected.bbox_max_xyz) &&
              selected.bbox_min_xyz.every((v, i) => v <= selected.bbox_max_xyz![i])) {
            box.set(new THREE.Vector3(...selected.bbox_min_xyz),
              new THREE.Vector3(...selected.bbox_max_xyz));
          } else {
            selected.sample_xyz.filter(isXYZ).forEach((xyz) =>
              box.expandByPoint(new THREE.Vector3(...xyz)));
          }
          if (box.isEmpty() && isXYZ(selected.center_xyz)) {
            box.expandByPoint(new THREE.Vector3(...selected.center_xyz));
          }
          if (box.isEmpty()) return;
          box.min.sub(world.center).multiplyScalar(world.scale);
          box.max.sub(world.center).multiplyScalar(world.scale);
          state.root.updateMatrixWorld(true);
          box.applyMatrix4(state.root.matrixWorld);
          fitCamera(state.camera, state.controls, box);
        };
        return () => { focusRef.current = () => {}; };
      }, [data, overlay, selectedObjectId]);

      // --------------------------------------------------
      // SAM / RANSAC OVERLAYS
      // --------------------------------------------------

      useEffect(() => {
        const group = overlayRef.current;
        const world = worldRef.current;

        if (!group || !world) return;

        disposeOverlay(group);

        if (!overlay) return;

        const toDisplay = (xyz: number[]) =>
          new THREE.Vector3(
            xyz[0],
            xyz[1],
            xyz[2]
          )
            .sub(world.center)
            .multiplyScalar(world.scale);

        function addHighlight(
          points: number[][],
          color: number,
          size: number,
          opacity: number
        ) {
          // Render every returned sample (up to 5,000 in the current backend).
          const valid = points.filter(isXYZ);

          if (!valid.length) return;

          const geo =
            new THREE.BufferGeometry().setFromPoints(
              valid.map(toDisplay)
            );

          const mat = new THREE.PointsMaterial({
            color,
            size,
            sizeAttenuation: true,
            transparent: opacity < 1,
            opacity,
            depthTest: true,
            depthWrite: false,
          });

          const cloud = new THREE.Points(
            geo,
            mat
          );

          cloud.frustumCulled = false;
          cloud.renderOrder = 2;

          group?.add(cloud);
        }

        // Optional floor-plane evidence.
        // This is NOT confirmed traversable geometry.
        if (showFloor && overlay.floor) {
          addHighlight(
            overlay.floor.sample_xyz,
            0xb1efc7,
            0.05,
            0.75
          );
        }

        // Only the selected SAM object is highlighted.
        const selected = overlay.objects.find(
          (object) =>
            String(object.object_id) ===
            selectedObjectId
        );

        if (!selected) return;

        addHighlight(
          selected.sample_xyz,
          0xffb099,
          0.08,
          1
        );

        // Small center marker on the selected object.
        if (
          selected.center_xyz.length === 3 &&
          selected.center_xyz.every(Number.isFinite)
        ) {
          const marker = new THREE.Mesh(
            new THREE.SphereGeometry(
              0.055,
              16,
              12
            ),
            new THREE.MeshBasicMaterial({
              color: 0xffb099,
              depthTest: true,
              depthWrite: false,
            })
          );

          marker.position.copy(
            toDisplay(selected.center_xyz)
          );

          marker.renderOrder = 3;

          group.add(marker);
        }

      }, [
        data,
        overlay,
        selectedObjectId,
        showFloor,
      ]);

      return (
        <div
          ref={mountRef}
          className="pointcloud-surface"
          role="img"
          aria-label="3D point cloud with selected SAM object and optional candidate floor overlay"
        />
      );
    }
  );
