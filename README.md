# Hosted Face Attendance MVP

A local, hosted face-attendance service. Client websites use the small browser SDK to open a camera-verification page; they do not implement recognition or receive face images.

## Start locally

1. Install Python 3.11+ and Node 20+.
2. Create the local environment and install the backend:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   Copy-Item .env.example .env
   ```

3. Download the two ONNX models into `models/`:

   ```powershell
   .\scripts\download-models.ps1
   ```

   This uses OpenCV Zoo's YuNet detector and SFace recognizer. Review their model licences before commercial deployment; never silently replace them with unknown model files.

4. Run the API:

   ```powershell
   py -m uvicorn app.main:app --reload --port 8000
   ```

5. Build the hosted UI into the FastAPI static folder:

   ```powershell
   cd frontend
   npm install
   npm.cmd run build
   ```

Open `http://localhost:8000`. FastAPI serves the built hosted UI from `app/static`, including `/verify/<token>`. For frontend-only hot reload during development, use `npm.cmd run dev`; it remains available at `http://localhost:5173`.

With an explicit `APP_ENV=development` in `.env`, opening that root URL shows a test-session launcher. Enter an Organization ID (stable identifier) and Organization Name. For **Enroll face** only, also enter an employee ID/name to label the face. For check-in/out, it creates an organization-only session and identifies the person from enrolled faces. The launcher API is unavailable whenever `APP_ENV` is not `development`.

## Integration model

Your customer's **server** creates a session using its `X-Client-Key`; it must not expose the key in browser code. Its site then passes the returned `token` to the browser SDK. See [examples/client-integration.html](examples/client-integration.html).

The SDK opens `HOSTED_ORIGIN/verify/<token>` in a popup. Completion is returned through `postMessage` and, optionally, the customer can verify the signed result JWT or receive the configured webhook.

An end-to-end example customer website is in [angular-client](angular-client/README.md). It is a separate Angular app with a small server-side `/api/attendance/session` route that keeps the organization client key out of browser code.

## Local admin flow

The seeded client key is `DEFAULT_CLIENT_KEY` in `.env`.

1. Create an employee: `POST /v1/admin/employees` with header `X-Admin-Key`.
2. Create their enrollment session using `POST /v1/admin/enrollment-sessions`.
3. Create any attendance session using `POST /v1/sessions` with only `action` and `return_origin`. Its authenticated client key selects the organization; no employee ID is accepted or needed.

The API docs are at `http://localhost:8000/docs`.

## What is intentionally local in this MVP

- SQLite stores tenants, employees, face embeddings, sessions and attendance events.
- Embeddings are compared in process against enrolled people from the token's organization. For production, retain the repository interface and replace it with PostgreSQL + pgvector.
- Images are decoded only for verification and never persisted.
- Check-in/out use browser-side MediaPipe landmarks to automatically capture only when one face is centered, close enough, level, and facing forward. Enrollment remains a manual action. MediaPipe's WASM and face-landmark model are fetched from its public CDN/model host on first browser use; self-host these assets before an offline or restricted production deployment.
- This MVP does **not** claim photo/video anti-spoof protection. Add a tested liveness component before using it for high-trust, unsupervised attendance.

## Security notes

Set real secrets in `.env`, run only over HTTPS in production, allowlist return origins, use a customer backend to create sessions, and obtain informed biometric consent. The browser UI never displays an API key.
