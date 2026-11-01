# 1 world — Next.js Geometry Studio

Video walkthrough → cloud Pi3X reconstruction → interactive colored PLY point cloud.

**Scope:** This release implements *only* the geometric reconstruction experience. It does not claim to have SAM segmentation, Muse Spark reasoning, a safe navigable floor map, or autonomous retrieval. There are no bundled example videos, rooms, point clouds, NPZ files, or personal media.

## Architecture

- **Next.js 16 App Router + React + TypeScript:** accessible upload, real video preview, job-state display, interactive Three.js point-cloud viewer.
- **Next.js Route Handlers:** same-origin API, stream uploads to Modal, validate a demo access code on the server, proxy PLY downloads, and keep the Modal URL and access secret off the browser.
- **Modal:** a small FastAPI ASGI endpoint receives video uploads, extracts 12 frames with FFmpeg, queues a separate L40S GPU Pi3X job, and serves a preview PLY and full PLY from a Volume.
- **Pi3X:** pretrained geometry model; no fine-tuning for each room. The worker outputs a real 3D point cloud; no demo-shaped geometry is invented.

```
Browser (Next.js) → /api/jobs → Next.js BFF → Modal FastAPI → Modal .spawn.aio() GPU
                                           ↖ status + PLY ← Modal Volume ← Pi3X
```

## Requirements

- Node 20.9+ (Node 22 recommended), npm
- Python 3.11+ for the Modal CLI
- Modal account with credits, GPU access, and permission to run the model weights
- A short MP4/MOV 4–30 seconds, <=60 MB

## 1. Deploy the Modal backend

From `cloud/`:

```bash
python -m pip install 'modal>=1.0'
modal setup
modal secret create world-auth WORLD_ACCESS_CODE='a-long-random-secret-you-choose'
modal deploy modal_app.py
```

Record the HTTPS URL Modal prints for `web()` (ends in something like `--worldpilot-reconstruct-web.modal.run`). It will be the value of `MODAL_API_URL`.

`modal serve modal_app.py` can be used during development, but its temporary URL changes when stopped. Modal may need time to build the Pi3X container and fetch the weights on the first call.

## 2. Start Next.js

From the project root:

```bash
npm install
cp .env.example .env.local
```

In `.env.local` set:

```
MODAL_API_URL=https://YOUR-PUBLISHED-MODAL-WEB-URL
WORLD_ACCESS_CODE=a-long-random-secret-you-choose
```

Use the **same** access code as the Modal secret. **Do not set a `NEXT_PUBLIC_` secret**. Neither the Modal URL nor the code are sent to the client; the user enters a temporary demo code that the Next.js server verifies.

```bash
npm run dev
```

Open http://localhost:3000, upload a video, enter the access code, then click **Reconstruct my world**. The UI will poll actual job status and load the returned PLY.

### Deployment notes

- The Next.js frontend can be deployed to a Node server on Railway, Render, or your own VPS. Set the same two server-side environment variables. On serverless platforms with small request body limits (for example Vercel), **60MB videos may not fit** through a Route Handler. For production, use a signed direct-to-storage upload flow instead. The current code is suitable for local Next.js development and Node deployments that permit streaming request bodies.
- Modal GPU time, network egress, and storage are billable; stop unused resources and protect the access code.
- Input files and completed jobs are kept in the Modal Volume and periodically removed after a 24-hour retention window. This is a shared-access demo, **not** a production multi-user private-storage system. Use per-user authentication before public launch.
- The original Pi3X model weights are **CC BY-NC 4.0**; check the license before commercial use.
- Browser PLY is uniformly downsampled to a maximum of 75,000 real vertices. Full output is downloadable; high-density point clouds can be large.
- Video-derived geometry is not calibrated for robot-safe navigation.

## Project layout

```
app/                 Next.js pages, globals, server-side API proxy routes
components/          Studio UI, Three.js point viewer
lib/                 shared types, secure Modal forwarding
cloud/modal_app.py   Modal GPU job and ASGI web endpoint
cloud/service.py     video validation, frame extraction, job and PLY APIs
cloud/tests/         backend API tests
```

## Testing

Run the cloud API offline tests:

```bash
cd cloud
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

For Next.js:

```bash
npm run typecheck
npm run build
```

**Verification boundary:** The frontend source and cloud Python are included, but this archive has not been deployed to your Modal account and cloud Pi3X inference has not been confirmed. You should verify a real video end-to-end before the hackathon demo.
