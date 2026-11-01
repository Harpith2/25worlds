
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
};

type Props = {
  data: ArrayBuffer;
  pointSize: number;
  overlay?: SpatialResult | null;
  selectedObjectId?: string | null;
  showFloor?: boolean;
};

type WorldTransform = {
  center: THREE.Vector3;
  scale: number;
};

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

      const resetRef =
        useRef<() => void>(() => {});

      useImperativeHandle(
        ref,
        () => ({
          reset: () => resetRef.current(),
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

        const overlays = new THREE.Group();
        scene.add(overlays);

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

          scene.add(base);

          resetRef.current = () => controls.reset();

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
          scene.remove(overlays);

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
          const valid = points.filter(
            (xyz) =>
              xyz.length === 3 &&
              xyz.every(Number.isFinite)
          );

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
