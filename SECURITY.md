# Privacy and security scope

This is a hackathon demo, not a production multi-tenant service.

- The source archive excludes all previously uploaded real room images, videos, PLY, NPZ, frames and personal data.
- `.gitignore` excludes common capture and reconstruction file types and `.env.local`.
- Next.js checks the demo code server-side before proxying requests, and Modal validates it again. Do not use a trivial code or embed it into client JavaScript.
- A shared access code does not isolate users from one another. Add real user auth and ownership checks before public invitations.
- Modal's Volume stores uploaded videos and results; completed/failed jobs are deleted during daily cleanup after 24 hours, and can be deleted manually once completed. Verify cloud retention policy for your app.
- Uploads are limited to 60MB and videos to 4–30 seconds; GPU concurrency is restricted to one job.
- No model weights are distributed in this repository.
