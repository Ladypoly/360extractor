// Spherical geometry for the rig editor overlay.
//
// This decides where a camera's footprint is drawn on the panorama, which is what
// the user reads to answer "is the car inside this camera?". If it disagrees with
// what ffmpeg actually extracts, the UI lies about the one thing it exists for --
// so tests/test_overlay_geometry.py runs this exact file under node and checks it
// against real extractions.
//
// Equirect maps bearing to x and elevation to y:
//   x = (yaw + 180) / 360 * W,  y = (90 - pitch) / 180 * H
// Directions are y-up: dir(yaw, pitch) = (sin y cos p, sin p, cos y cos p).

export const RAD = Math.PI / 180;
export const DEG = 180 / Math.PI;

export function dirFrom(yaw, pitch) {
  const cy = Math.cos(yaw * RAD), sy = Math.sin(yaw * RAD);
  const cp = Math.cos(pitch * RAD), sp = Math.sin(pitch * RAD);
  return [sy * cp, sp, cy * cp];
}

export function angleOf(d) {
  return {
    yaw: Math.atan2(d[0], d[2]) * DEG,
    pitch: Math.asin(Math.max(-1, Math.min(1, d[1]))) * DEG,
  };
}

export const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
export const scale = (a, k) => [a[0]*k, a[1]*k, a[2]*k];
export const add = (a, b) => [a[0]+b[0], a[1]+b[1], a[2]+b[2]];
export function norm(a) {
  const l = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0]/l, a[1]/l, a[2]/l];
}

const ZERO = { yaw: 0, pitch: 0, roll: 0 };

/** World-space camera basis, with rig orientation folded in. */
export function basisOf(camera, orientation = ZERO) {
  const yaw = camera.yaw + (orientation.yaw || 0);
  const pitch = camera.pitch + (orientation.pitch || 0);
  const roll = (camera.roll || 0) + (orientation.roll || 0);

  const forward = dirFrom(yaw, pitch);
  // `right` is taken directly from the bearing so it never degenerates, even when
  // the camera looks straight down and a world-up cross product would collapse.
  let right = dirFrom(yaw + 90, 0);
  // forward x right, not right x forward -- the other order points down and mirrors
  // the footprint vertically. Symmetric fovs hide that; roll does not.
  let up = norm(cross(forward, right));

  if (roll) {
    const c = Math.cos(roll * RAD), s = Math.sin(roll * RAD);
    const rolled = add(scale(right, c), scale(up, s));
    up = add(scale(up, c), scale(right, -s));
    right = rolled;
  }
  return { forward, right, up };
}

/** Direction through a normalized image-plane coordinate in [-1, 1]. */
export function rayThrough(camera, basis, nx, ny) {
  const tx = Math.tan(camera.h_fov / 2 * RAD);
  const ty = Math.tan(camera.v_fov / 2 * RAD);
  return norm(add(basis.forward,
                  add(scale(basis.right, nx * tx), scale(basis.up, ny * ty))));
}

/** Bearing and elevation of a point on the image plane. */
export function anglesAt(camera, orientation, nx, ny) {
  return angleOf(rayThrough(camera, basisOf(camera, orientation), nx, ny));
}

/**
 * Footprint outline in equirect pixel coordinates.
 *
 * The seam is unwrapped so the polygon stays continuous; callers draw it at
 * -width, 0 and +width so whichever copy is on screen appears.
 */
export function footprint(camera, orientation, width, height, steps = 20) {
  const basis = basisOf(camera, orientation);
  const edge = [];
  for (let i = 0; i < steps; i++) edge.push([-1 + 2*i/steps, -1]);
  for (let i = 0; i < steps; i++) edge.push([1, -1 + 2*i/steps]);
  for (let i = 0; i < steps; i++) edge.push([1 - 2*i/steps, 1]);
  for (let i = 0; i < steps; i++) edge.push([-1, 1 - 2*i/steps]);

  const points = edge.map(([nx, ny]) => {
    const a = angleOf(rayThrough(camera, basis, nx, ny));
    return { x: (a.yaw + 180) / 360 * width, y: (90 - a.pitch) / 180 * height };
  });

  for (let i = 1; i < points.length; i++) {
    while (points[i].x - points[i-1].x > width / 2) points[i].x -= width;
    while (points[i].x - points[i-1].x < -width / 2) points[i].x += width;
  }
  return points;
}

/** Fraction of a camera's view falling below -angle elevation. */
export function occlusionFraction(camera, orientation, angle, steps = 11) {
  if (angle <= 0) return 0;
  const basis = basisOf(camera, orientation);
  let inside = 0, total = 0;
  for (let iy = 0; iy < steps; iy++) {
    for (let ix = 0; ix < steps; ix++) {
      const nx = -1 + 2 * ix / (steps - 1);
      const ny = -1 + 2 * iy / (steps - 1);
      if (angleOf(rayThrough(camera, basis, nx, ny)).pitch < -angle) inside++;
      total++;
    }
  }
  return inside / total;
}

/** Centre, edge midpoints and corners of a camera, for tests and readouts. */
export function probeEdges(camera, orientation = ZERO) {
  const probes = {
    center: [0, 0],
    left: [-1, 0], right: [1, 0], top: [0, 1], bottom: [0, -1],
    tl: [-1, 1], tr: [1, 1], bl: [-1, -1], br: [1, -1],
  };
  const basis = basisOf(camera, orientation);
  const out = {};
  for (const [name, [nx, ny]] of Object.entries(probes)) {
    const a = angleOf(rayThrough(camera, basis, nx, ny));
    out[name] = { yaw: a.yaw, pitch: a.pitch };
  }
  return out;
}

// ── views: where a direction lands in the picture on screen ──────────────
//
// Two pictures carry the same sphere. The panorama maps bearing to x and elevation
// to y. A raw dual-fisheye frame maps each hemisphere onto a circle, and it is worth
// drawing the rig on *that* when that is what the camera shot: the stitch line is
// visible in it, and a camera placed across the seam is a camera whose images will be
// worst. Both views expose the same three operations, so the editor draws, hit-tests
// and drags through whichever it is showing.
//
// The dual-fisheye numbers are not a guess. `v360=e:dfisheye` was run over markers at
// known bearings and the landing pixels read back (tests/test_overlay_geometry.py keeps
// doing it): the FRONT hemisphere is the right-hand circle, the back is the left-hand
// one and mirrored in x, the radius is linear in the angle from the lens axis
// (equidistant), and the full half-width is the lens's own field of view.

export const equirectView = {
  kind: "panorama",
  wraps: true,
  project(yaw, pitch, width, height) {
    return { x: (yaw + 180) / 360 * width, y: (90 - pitch) / 180 * height, lens: 0 };
  },
  unproject(x, y, width, height) {
    return { yaw: x / width * 360 - 180, pitch: 90 - y / height * 180 };
  },
};

/** The two-circle view of a dual-fisheye source, `lensFov` degrees per lens. */
export function dualFisheyeView(lensFov = 190) {
  const half = Math.max(lensFov, 1) / 2 * RAD;
  return {
    kind: "lenses",
    wraps: false,
    lensFov,

    project(yaw, pitch, width, height) {
      const [dx, dy, dz] = dirFrom(yaw, pitch);
      const front = dz >= 0;
      const theta = Math.acos(Math.max(-1, Math.min(1, front ? dz : -dz)));
      const span = Math.hypot(dx, dy) || 1e-9;
      const ux = dx / span, uy = dy / span;
      const rx = theta / half * (width / 4);
      const ry = theta / half * (height / 2);
      // Front on the right, back on the left and mirrored in x -- measured, not assumed.
      const cx = front ? 3 * width / 4 : width / 4;
      return { x: cx + (front ? rx * ux : -rx * ux), y: height / 2 - ry * uy,
               lens: front ? 1 : 0 };
    },

    unproject(x, y, width, height) {
      const front = x >= width / 2;
      const cx = front ? 3 * width / 4 : width / 4;
      const dx = (front ? x - cx : cx - x) / (width / 4);
      const dy = (height / 2 - y) / (height / 2);
      const r = Math.hypot(dx, dy);
      const theta = Math.min(r, 1.4) * half;
      const span = r || 1e-9;
      const st = Math.sin(theta), ct = Math.cos(theta);
      // `dx` already carries the back lens's mirror, so the direction needs no
      // second flip -- applying one sent every bearing behind the rig to its opposite.
      return angleOf([st * dx / span, st * dy / span, front ? ct : -ct]);
    },

    /** Circle centre and radii for one lens, for drawing the frame of the picture. */
    circle(lens, width, height) {
      return { x: lens ? 3 * width / 4 : width / 4, y: height / 2,
               rx: width / 4, ry: height / 2 };
    },
  };
}

/**
 * Footprint outline as one or more paths, in the pixel space of `view`.
 *
 * The panorama needs the seam unwrapped so its single polygon stays continuous. The
 * lens view instead *splits*: a camera looking across the stitch is genuinely two
 * shapes, one on each circle, and joining them would draw a line straight through the
 * middle of a picture where no such edge exists.
 */
export function footprintPaths(camera, orientation, view, width, height, steps = 24) {
  const basis = basisOf(camera, orientation);
  const edge = [];
  for (let i = 0; i < steps; i++) edge.push([-1 + 2*i/steps, -1]);
  for (let i = 0; i < steps; i++) edge.push([1, -1 + 2*i/steps]);
  for (let i = 0; i < steps; i++) edge.push([1 - 2*i/steps, 1]);
  for (let i = 0; i < steps; i++) edge.push([-1, 1 - 2*i/steps]);

  const projected = edge.map(([nx, ny]) => {
    const a = angleOf(rayThrough(camera, basis, nx, ny));
    return view.project(a.yaw, a.pitch, width, height);
  });

  if (view.wraps) {
    const points = projected.map((p) => ({ x: p.x, y: p.y }));
    for (let i = 1; i < points.length; i++) {
      while (points[i].x - points[i-1].x > width / 2) points[i].x -= width;
      while (points[i].x - points[i-1].x < -width / 2) points[i].x += width;
    }
    return [points];
  }

  // Walk the closed outline and cut it wherever it changes lens.
  const paths = [];
  let run = [];
  for (let i = 0; i < projected.length + 1; i++) {
    const point = projected[i % projected.length];
    const previous = run.length ? projected[(i - 1) % projected.length] : null;
    if (previous && point.lens !== previous.lens) {
      if (run.length > 1) paths.push(run);
      run = [];
    }
    run.push({ x: point.x, y: point.y });
  }
  if (run.length > 1) paths.push(run);
  return paths;
}
