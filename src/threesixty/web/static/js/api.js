// Talking to the server.
//
// One place that knows how errors come back, so every caller gets a real Error with a
// useful message rather than a bare 400. `AlreadyRunning` is modelled explicitly
// because the UI needs to *do* something with it -- offer to go to the stage that is
// busy, rather than repeat "something is already running".

export class ApiError extends Error {
  constructor(message, status, payload = {}) {
    super(message);
    this.status = status;
    this.payload = payload;
    this.runningStage = payload.running_stage || null;
  }

  get isAlreadyRunning() {
    return this.status === 409;
  }
}

async function request(path, options) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (cause) {
    throw new ApiError("The application server is not responding.", 0);
  }

  const type = response.headers.get("Content-Type") || "";
  if (!type.includes("application/json")) {
    if (!response.ok) throw new ApiError(response.statusText, response.status);
    return response;
  }

  const payload = await response.json();
  if (!response.ok || payload.error) {
    throw new ApiError(payload.error || response.statusText, response.status, payload);
  }
  return payload;
}

export function get(path) {
  return request(path, { method: "GET" });
}

export function post(path, body = {}) {
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** Ask the server to raise a native file dialog; the browser cannot supply real paths. */
export async function pick(mode, title, kind = "media", initial = "") {
  const { paths } = await post("/api/pick", { mode, title, kind, initial });
  return paths;
}

export const jobs = {
  all: (logLines = 0) => get(`/api/jobs?log=${logLines}`),
  status: (stage, logLines = 400) => post("/api/job/status", { stage, log: logLines }),
  cancel: (stage) => post("/api/job/cancel", { stage }),
};
