# Modal Backend

Deploy from this folder:

```bash
pip install 'modal>=1.0'
modal setup
modal secret create world-auth WORLD_ACCESS_CODE='your-long-random-code'
modal deploy modal_app.py
```

The Modal worker does not host the frontend. It exposes `/api/jobs` and returns real PLY assets; the Next.js app acts as the authorized frontend and proxies requests. Never embed your access code in client-side JavaScript.

Useful documentation:
- https://modal.com/docs/guide/webhooks
- https://modal.com/docs/sdk/py/latest/asgi_app
- https://modal.com/docs/sdk/py/latest/Function
- https://modal.com/docs/guide/volumes

## Verify the deployed API source

The API source is baked into the web image. After editing it, redeploy this exact
checkout (PowerShell, from `cloud/`):

```powershell
modal deploy --env main .\modal_app.py
```

Open `<the web URL printed by deployment>/api/health`. Its `revision` must match
this local source hash:

```powershell
(Get-FileHash .\service.py -Algorithm SHA256).Hash.Substring(0, 12).ToLower()
```

The same revision is printed on API startup and on HTTP 400 validation errors.
If it differs or is missing, check that Next.js `MODAL_API_URL` points to the
newly deployed web URL and restart the Next.js server after changing `.env.local`.
Inspect a fresh call after deployment: historical log entries do not change.
The API uses `await volume.reload.aio()` and `await volume.commit.aio()`;
blocking Volume calls in historical logs indicate a different source revision.
HTTP 400 details are returned to the frontend and logged separately from Modal
async usage warnings, so the actual video-validation failure is visible.
