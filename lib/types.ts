export type JobStatus = "queued" | "preparing" | "reconstructing" | "exporting" | "completed" | "failed";
export type UIMode = "ready" | "uploading" | "loading" | JobStatus;

export interface ReconstructionJob {
  id: string;
  name: string;
  status: JobStatus;
  message?: string;
  error?: string;
  frame_count?: number;
  total_points?: number;
  preview_points?: number;
  duration_seconds?: number;
  created_at?: string;
}

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
