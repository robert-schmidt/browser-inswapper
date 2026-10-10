/// <reference lib="webworker" />
// GPU compositing (WebGL2 on an OffscreenCanvas): the camera frame, then the latest swapped face drawn as a mesh.
// Each mesh vertex sits on a face landmark of *this* frame and samples the swapped face where that landmark was
// when the swap ran. So between model runs (~10 per second) the last result still follows head pose, mouth,
// eyes and brows at the camera's full frame rate, instead of sliding around as a flat picture.

import type { Mesh } from './mesh.ts';
import type { FaceMask } from './protocol.ts';

const FRAME_VS = `#version 300 es
out vec2 uv;
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  uv = vec2(p.x, 1.0 - p.y);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;
const FRAME_FS = `#version 300 es
precision highp float;
in vec2 uv;
uniform sampler2D frame;
out vec4 color;
void main() { color = texture(frame, uv); }`;

const FACE_VS = `#version 300 es
in vec2 pos;      // pixels, this frame
in vec2 tex;      // 0..1 in the swapped face
in float weight;  // 0 at the contour, 1 inside
uniform vec2 size;
out vec2 uv;
out float w;
void main() {
  uv = tex; w = weight;
  gl_Position = vec4(pos.x / size.x * 2.0 - 1.0, 1.0 - pos.y / size.y * 2.0, 0.0, 1.0);
}`;
const FACE_FS = `#version 300 es
precision highp float;
in vec2 uv;
in float w;
uniform sampler2D face;
uniform sampler2D skin;   // 0 where something is in front of the face, over the whole frame (white when occlusion is off)
uniform vec2 size;
uniform float strength;
out vec4 color;
void main() {
  // Fade near the crop's border too, where the model has no face to draw.
  vec2 e = smoothstep(0.0, 0.06, uv) * smoothstep(0.0, 0.06, 1.0 - uv);
  // A hand or mic in front of the face stays in front.
  float s = smoothstep(0.3, 0.7, texture(skin, vec2(gl_FragCoord.x / size.x, 1.0 - gl_FragCoord.y / size.y)).r);
  color = vec4(texture(face, uv).rgb, w * e.x * e.y * s * strength);
}`;

function program(gl: WebGL2RenderingContext, vs: string, fs: string) {
  const p = gl.createProgram()!;
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]] as const) {
    const s = gl.createShader(type)!;
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) ?? 'shader');
    gl.attachShader(p, s);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p) ?? 'program');
  return p;
}

function texture(gl: WebGL2RenderingContext) {
  const t = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}

// A new swap result fades in over this long instead of replacing the last one at once: at ~10 results a
// second, hard cuts read as shimmer around the eyes and mouth.
const CROSSFADE_MS = 70;

type Slot = { tex: WebGLTexture; texBuf: WebGLBuffer; vao: WebGLVertexArrayObject; since: number };

export class Compositor {
  readonly canvas: OffscreenCanvas;
  private gl: WebGL2RenderingContext;
  private frameProg: WebGLProgram;
  private faceProg: WebGLProgram;
  private frameTex: WebGLTexture;
  private skinTex: WebGLTexture;
  private posBuf: WebGLBuffer;
  private slots: Slot[];
  private current = -1; // newest slot; the other one is the result fading out
  private hasPrevious = false;
  private count: number;

  constructor(mesh: Mesh) {
    this.canvas = new OffscreenCanvas(2, 2);
    const gl = this.canvas.getContext('webgl2', { alpha: false, antialias: true, premultipliedAlpha: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error('WebGL2 is needed to draw the swapped face.');
    this.gl = gl;
    this.frameProg = program(gl, FRAME_VS, FRAME_FS);
    this.faceProg = program(gl, FACE_VS, FACE_FS);
    this.frameTex = texture(gl);
    this.skinTex = texture(gl);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    this.setSkin(null);
    const n = mesh.weights.length;
    const buffer = (data: Float32Array | Uint16Array, target: number, usage: number) => {
      const b = gl.createBuffer()!;
      gl.bindBuffer(target, b);
      gl.bufferData(target, data, usage);
      return b;
    };
    this.posBuf = buffer(new Float32Array(n * 2), gl.ARRAY_BUFFER, gl.DYNAMIC_DRAW);
    const weightBuf = buffer(mesh.weights, gl.ARRAY_BUFFER, gl.STATIC_DRAW);
    const indexBuf = buffer(mesh.triangles, gl.ELEMENT_ARRAY_BUFFER, gl.STATIC_DRAW);
    const bind = (name: string, b: WebGLBuffer, size: number) => {
      const loc = gl.getAttribLocation(this.faceProg, name);
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    };
    // Two slots (shared mesh positions, own texture and texture coordinates), for the crossfade.
    this.slots = [0, 1].map(() => {
      const texBuf = buffer(new Float32Array(n * 2), gl.ARRAY_BUFFER, gl.DYNAMIC_DRAW);
      const vao = gl.createVertexArray()!;
      gl.bindVertexArray(vao);
      bind('pos', this.posBuf, 2);
      bind('tex', texBuf, 2);
      bind('weight', weightBuf, 1);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuf);
      gl.bindVertexArray(null);
      return { tex: texture(gl), texBuf, vao, since: 0 };
    });
    this.count = mesh.triangles.length;
  }

  /** A new swapped face, and where each landmark was inside it (0..1, x/y pairs). */
  setFace(face: TexImageSource, uv: Float32Array) {
    const gl = this.gl;
    this.hasPrevious = this.current >= 0;
    this.current = (this.current + 1) % 2;
    const slot = this.slots[this.current];
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, slot.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, face);
    gl.bindBuffer(gl.ARRAY_BUFFER, slot.texBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, uv);
    slot.since = performance.now();
  }
  clearFace() { this.current = -1; this.hasPrevious = false; }

  private setSkin(mask: FaceMask | null) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.skinTex);
    if (mask) gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, mask.width, mask.height, 0, gl.RED, gl.UNSIGNED_BYTE, mask.data);
    else gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array([255]));
  }

  /** Draws `frame`, then the face on `points` (pixels, x/y pairs for every landmark) when there is one. */
  draw(frame: TexImageSource, width: number, height: number, points: Float32Array | null, strength: number, mask: FaceMask | null = null) {
    const gl = this.gl;
    if (this.canvas.width !== width || this.canvas.height !== height) { this.canvas.width = width; this.canvas.height = height; }
    gl.viewport(0, 0, width, height);
    gl.disable(gl.BLEND);
    gl.useProgram(this.frameProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.frameTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, frame);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (!points || this.current < 0 || strength <= 0.001) return;
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(this.faceProg);
    gl.uniform2f(gl.getUniformLocation(this.faceProg, 'size'), width, height);
    gl.activeTexture(gl.TEXTURE1);
    this.setSkin(mask);
    gl.uniform1i(gl.getUniformLocation(this.faceProg, 'skin'), 1);
    gl.uniform1i(gl.getUniformLocation(this.faceProg, 'face'), 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, points);
    const cur = this.slots[this.current];
    const t = Math.min(1, (performance.now() - cur.since) / CROSSFADE_MS);
    const drawSlot = (slot: Slot, alpha: number) => {
      gl.uniform1f(gl.getUniformLocation(this.faceProg, 'strength'), alpha);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, slot.tex);
      gl.bindVertexArray(slot.vao);
      gl.drawElements(gl.TRIANGLES, this.count, gl.UNSIGNED_SHORT, 0);
    };
    // The outgoing result underneath at full strength, the new one over it fading in.
    if (this.hasPrevious && t < 1) drawSlot(this.slots[(this.current + 1) % 2], strength);
    drawSlot(cur, this.hasPrevious ? strength * t : strength);
    gl.bindVertexArray(null);
  }
}
