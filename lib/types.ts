export type JobStatus = "queued" | "preparing" | "reconstructing" | "exporting" | "completed" | "failed";
export type UIMode = "ready" | "uploading" | "loading" | JobStatus;

export interface ReconstructionJob {
  id: string;
  name: string;
  status: JobStatus;
  message?: string;
  error?: string;
  spatial_ready?: boolean;
  spatial_frames?: number;
  frame_count?: number;
  total_points?: number;
  preview_points?: number;
  duration_seconds?: number;
  created_at?: string;
}


export interface SpatialObject {
  object_id: string | number;
  label: string;
  sam_confidence: number | null;
  bounds_xyxy: [number, number, number, number];
  mask_pixels: number;
  grounded_points: number;
  center_xyz: [number, number, number];
  bbox_min_xyz?: [number, number, number];
  bbox_max_xyz?: [number, number, number];
  sample_xyz: number[][];
  distance_to_floor_plane?: number;
}

export interface SpatialPlane {
  normal: [number, number, number];
  offset: number;
  inlier_count: number;
  inlier_ratio: number;
  rms_residual: number;
  sample_xyz: number[][];
}

export interface SpatialResult {
  frame_index: number;
  image_width: number;
  image_height: number;
  object_concept: string;
  source: string;
  floor: SpatialPlane | null;
  objects: SpatialObject[];
  warnings: string[];
}

export type SpatialResponse = {
  status: "queued" | "failed" | "completed";
  error?: string;
  result?: SpatialResult;
  frame_image_url?: string | null;
};


export const statuses: Record<UIMode, { title: string; description: string; step: number }> = {
  ready: { title: "Ready when you are", description: "Upload a walkthrough to create your first 3D world.", step: 0 },
  uploading: { title: "Uploading video", description: "Sending your walkthrough to the cloud securely.", step: 1 },
  queued: { title: "Finding a GPU", description: "Your job is queued for reconstruction.", step: 1 },
  preparing: { title: "Preparing your frames", description: "Selecting overlapping views for reconstruction.", step: 1 },
  reconstructing: { title: "Building the geometry", description: "Reconstructing the 3D scene from your video.", step: 1 },
  exporting: { title: "Preparing your 3D scene", description: "Generating the point cloud and browser-ready preview.", step: 1 },
  loading: { title: "Opening your world", description: "Loading the reconstructed points into the viewer.", step: 2 },
  completed: { title: "Your world is ready", description: "Orbit, zoom, inspect, and download the reconstruction.", step: 2 },
  failed: { title: "Reconstruction stopped", description: "Review the error and try another walkthrough.", step: 0 },
};
