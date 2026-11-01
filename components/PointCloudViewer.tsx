"use client";

import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import * as THREE from "three";
import { TrackballControls } from "three/addons/controls/TrackballControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

export type PointCloudControls = { reset: () => void };

type Props = { data: ArrayBuffer; pointSize: number };

export const PointCloudViewer = forwardRef<PointCloudControls, Props>(function PointCloudViewer({ data, pointSize }, ref) {
  const mount = useRef<HTMLDivElement>(null);
  const resetFn = useRef<() => void>(() => {});
  const materialRef = useRef<THREE.PointsMaterial | null>(null);

  useImperativeHandle(ref, () => ({ reset: () => resetFn.current() }), []);
  useEffect(() => {
    if (materialRef.current) {
      materialRef.current.size = pointSize;
      materialRef.current.needsUpdate = true;
    }
  }, [pointSize]);

  useEffect(() => {
    const container = mount.current;
    if (!container) return;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#171717");
    const camera = new THREE.PerspectiveCamera(46, container.clientWidth / Math.max(container.clientHeight, 1), 0.01, 3000);
    camera.position.set(6.4, 3.5, 7.8);
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const controls = new TrackballControls(camera, renderer.domElement);
    // Trackball rotation can cross both poles and roll fully upside down.
    controls.staticMoving = true;
    controls.rotateSpeed = 2;
    controls.zoomSpeed = 1.2;
    controls.panSpeed = 0.3;
    controls.minDistance = 0.08;
    controls.maxDistance = 140;

    let geometry: THREE.BufferGeometry | undefined;
    let material: THREE.PointsMaterial | undefined;
    let raf = 0;
    let resizeObserver: ResizeObserver | undefined;
    try {
      geometry = new PLYLoader().parse(data.slice(0));
      if (!geometry.getAttribute("position")) throw new Error("PLY is missing vertex positions.");
      geometry.computeBoundingBox();
      const bounds = geometry.boundingBox;
      if (!bounds || bounds.isEmpty()) throw new Error("PLY does not contain valid geometry.");
      const size = bounds.getSize(new THREE.Vector3());
      const center = bounds.getCenter(new THREE.Vector3());
      const longest = Math.max(size.x, size.y, size.z, 0.0001);
      geometry.translate(-center.x, -center.y, -center.z);
      // Geometry is centered and normalized for convenient view; the downloaded PLY is untouched.
      geometry.scale(5 / longest, 5 / longest, 5 / longest);
      const vertexColors = Boolean(geometry.getAttribute("color"));
      material = new THREE.PointsMaterial({
        size: pointSize,
        sizeAttenuation: true,
        vertexColors,
        color: vertexColors ? 0xffffff : 0xc6b5f6,
        transparent: false,
        depthWrite: true,
      });
      materialRef.current = material;
      const points = new THREE.Points(geometry, material);
      points.frustumCulled = false;
      scene.add(points);
      const reset = () => {
        controls.reset();
      };
      resetFn.current = reset;
      reset();
      const render = () => {
        controls.update();
        renderer.render(scene, camera);
        raf = requestAnimationFrame(render);
      };
      render();
      resizeObserver = new ResizeObserver(() => {
        if (!container.clientWidth || !container.clientHeight) return;
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
        controls.handleResize();
      });
      resizeObserver.observe(container);
    } catch (error) {
      container.setAttribute("data-error", error instanceof Error ? error.message : "Unable to open PLY.");
      // The parent already displays a clear status; this makes an invalid PLY visible in-browser.
      const fallback = document.createElement("div");
      fallback.className = "viewer-error-inline";
      fallback.textContent = error instanceof Error ? error.message : "Could not render PLY.";
      container.appendChild(fallback);
    }
    return () => {
      if (raf) cancelAnimationFrame(raf);
      resizeObserver?.disconnect();
      controls.dispose();
      geometry?.dispose();
      material?.dispose();
      materialRef.current = null;
      renderer.dispose();
      renderer.domElement.remove();
      container.querySelector(".viewer-error-inline")?.remove();
    };
  }, [data]); // Intentionally keep scene persistent while point-size control changes.

  return <div className="pointcloud-surface" ref={mount} role="img" aria-label="Interactive colored point cloud. Drag to rotate freely through 360 degrees, including upside down. Scroll to zoom and right-drag to pan." />;
});
