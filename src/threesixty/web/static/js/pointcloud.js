// A tiny self-contained WebGL point-cloud viewer -- no framework, no build step.
//
// Feeds on the compact binary /api/reconstruct/points serves: [f64 mtime][u32 count]
// [f32 xyz...][u8 rgb...]. Orbit with the left mouse, zoom with the wheel. It exists so
// the reconstruction can be watched as its sparse cloud is built, instead of a console.

export function PointCloud(canvas) {
  const noop = { ok: false, load() {}, clear() {}, start() {}, stop() {}, count: 0 };
  const gl = canvas.getContext("webgl", { antialias: true, alpha: true, depth: true });
  let program;
  try {
    if (!gl) throw new Error("no webgl");
    program = build(gl,
      `attribute vec3 position; attribute vec3 color;
       uniform mat4 mvp; uniform float pointSize; varying vec3 vColor;
       void main() {
         gl_Position = mvp * vec4(position, 1.0);
         gl_PointSize = pointSize; vColor = color;
       }`,
      `precision mediump float; varying vec3 vColor;
       void main() {
         vec2 d = gl_PointCoord - vec2(0.5);
         if (dot(d, d) > 0.25) discard;              // round dots
         gl_FragColor = vec4(vColor, 1.0);
       }`);
  } catch {
    return noop;   // no WebGL here -- the stage still works, just without the cloud
  }

  const posBuf = gl.createBuffer();
  const colBuf = gl.createBuffer();
  const loc = {
    position: gl.getAttribLocation(program, "position"),
    color: gl.getAttribLocation(program, "color"),
    mvp: gl.getUniformLocation(program, "mvp"),
    pointSize: gl.getUniformLocation(program, "pointSize"),
  };

  let count = 0;
  let center = [0, 0, 0];
  let radius = 1;
  const cam = { yaw: 0.7, pitch: 0.45, dist: 2.6 };

  function load(buffer) {
    const view = new DataView(buffer);
    const n = view.getUint32(8, true);
    if (!n) { count = 0; return; }
    const positions = new Float32Array(buffer.slice(12, 12 + n * 12));
    const rgb = new Uint8Array(buffer, 12 + n * 12, n * 3);
    const colors = new Float32Array(n * 3);
    for (let i = 0; i < colors.length; i++) colors[i] = rgb[i] / 255;

    [center, radius] = bounds(positions, n);
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, colBuf);
    gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);
    count = n;
  }

  function clear() { count = 0; }

  function frame() {
    const w = canvas.clientWidth || 1, h = canvas.clientHeight || 1;
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!count) return;

    gl.enable(gl.DEPTH_TEST);
    gl.useProgram(program);
    gl.uniformMatrix4fv(loc.mvp, false, mvp(w / h));
    gl.uniform1f(loc.pointSize, Math.max(1.5, (canvas.height / 500) * 2));
    attrib(loc.position, posBuf);
    attrib(loc.color, colBuf);
    gl.drawArrays(gl.POINTS, 0, count);
  }

  function attrib(index, buffer) {
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.enableVertexAttribArray(index);
    gl.vertexAttribPointer(index, 3, gl.FLOAT, false, 0, 0);
  }

  function mvp(aspect) {
    const d = cam.dist * radius;
    const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const eye = [center[0] + d * cp * Math.sin(cam.yaw),
                 center[1] + d * sp,
                 center[2] + d * cp * Math.cos(cam.yaw)];
    return multiply(perspective(0.9, aspect, radius * 0.01, radius * 100),
                    lookAt(eye, center, [0, 1, 0]));
  }

  // ── orbit / zoom ────────────────────────────────────────────────────────
  let dragging = false, lastX = 0, lastY = 0;
  canvas.addEventListener("mousedown", (e) => {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
  });
  window.addEventListener("mouseup", () => { dragging = false; });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    cam.yaw -= (e.clientX - lastX) * 0.008;
    cam.pitch = Math.max(-1.5, Math.min(1.5, cam.pitch + (e.clientY - lastY) * 0.008));
    lastX = e.clientX; lastY = e.clientY;
  });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    cam.dist = Math.max(0.2, Math.min(30, cam.dist * (1 + Math.sign(e.deltaY) * 0.1)));
  }, { passive: false });

  let running = false;
  function start() {
    if (running) return;
    running = true;
    const tick = () => { if (!running) return; frame(); requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  }
  function stop() { running = false; }

  return { ok: true, load, clear, start, stop, get count() { return count; } };
}

// ── helpers ────────────────────────────────────────────────────────────────

function build(gl, vsSource, fsSource) {
  const compile = (type, source) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader) || "shader compile failed");
    }
    return shader;
  };
  const program = gl.createProgram();
  gl.attachShader(program, compile(gl.VERTEX_SHADER, vsSource));
  gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fsSource));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program) || "program link failed");
  }
  return program;
}

function bounds(positions, n) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < 3; a++) {
      const v = positions[i * 3 + a];
      if (v < min[a]) min[a] = v;
      if (v > max[a]) max[a] = v;
    }
  }
  const center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  const radius = Math.max(
    Math.hypot(max[0] - center[0], max[1] - center[1], max[2] - center[2]), 1e-3);
  return [center, radius];
}

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return [f / aspect, 0, 0, 0,
          0, f, 0, 0,
          0, 0, (far + near) * nf, -1,
          0, 0, 2 * far * near * nf, 0];
}

function lookAt(eye, target, up) {
  const z = normalize([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return [x[0], y[0], z[0], 0,
          x[1], y[1], z[1], 0,
          x[2], y[2], z[2], 0,
          -dot(x, eye), -dot(y, eye), -dot(z, eye), 1];
}

function multiply(a, b) {
  const out = new Array(16);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      out[c * 4 + r] = a[0 * 4 + r] * b[c * 4 + 0] + a[1 * 4 + r] * b[c * 4 + 1]
                     + a[2 * 4 + r] * b[c * 4 + 2] + a[3 * 4 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1],
                         a[2] * b[0] - a[0] * b[2],
                         a[0] * b[1] - a[1] * b[0]];
function normalize(v) {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}
