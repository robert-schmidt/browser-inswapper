/// <reference lib="webworker" />
// GPU compositing (WebGL2 on an OffscreenCanvas): the camera frame, then the latest swapped face drawn as a mesh.
// Each mesh vertex sits on a face landmark of *this* frame and samples the swapped face where that landmark was
// when the swap ran. So between model runs (~10 per second) the last result still follows head pose, mouth,
// eyes and brows at the camera's full frame rate, instead of sliding around as a flat picture.

import type { Mesh } from './mesh.ts';

const FRAME_VS = `#version 300 es
out vec2 uv;
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  uv = vec2(p.x, 1.0 - p.y);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;
const FRAME_FS = `#version 300 es
precision mediump float;
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
precision mediump float;
in vec2 uv;
in float w;
uniform sampler2D face;
uniform float strength;
out vec4 color;
void main() {
  // Fade near the crop's border too, where the model has no face to draw.
  vec2 e = smoothstep(0.0, 0.06, uv) * smoothstep(0.0, 0.06, 1.0 - uv);
  color = vec4(texture(face, uv).rgb, w * e.x * e.y * strength);
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

export class Compositor {
  readonly canvas: OffscreenCanvas;
  private gl: WebGL2RenderingContext;
  private frameProg: WebGLProgram;
  private faceProg: WebGLProgram;
  private frameTex: WebGLTexture;
  private faceTex: WebGLTexture;
  private vao: WebGLVertexArrayObject;
  private posBuf: WebGLBuffer;
  private texBuf: WebGLBuffer;
  private count: number;
  private hasFace = false;

  constructor(mesh: Mesh) {
    this.canvas = new OffscreenCanvas(2, 2);
    const gl = this.canvas.getContext('webgl2', { alpha: false, antialias: true, premultipliedAlpha: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error('WebGL2 is needed to draw the swapped face.');
    this.gl = gl;
    this.frameProg = program(gl, FRAME_VS, FRAME_FS);
    this.faceProg = program(gl, FACE_VS, FACE_FS);
    this.frameTex = texture(gl);
    this.faceTex = texture(gl);
    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);
    const attr = (name: string, size: number, data: Float32Array, usage: number) => {
      const b = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, data, usage);
      const loc = gl.getAttribLocation(this.faceProg, name);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
      return b;
    };
    const n = mesh.weights.length;
    this.posBuf = attr('pos', 2, new Float32Array(n * 2), gl.DYNAMIC_DRAW);
    this.texBuf = attr('tex', 2, new Float32Array(n * 2), gl.DYNAMIC_DRAW);
    attr('weight', 1, mesh.weights, gl.STATIC_DRAW);
    const ib = gl.createBuffer()!;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.triangles, gl.STATIC_DRAW);
    this.count = mesh.triangles.length;
    gl.bindVertexArray(null);
  }

  /** A new swapped face, and where each landmark was inside it (0..1, x/y pairs). */
  setFace(face: TexImageSource, uv: Float32Array) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.faceTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, face);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.texBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, uv);
    this.hasFace = true;
  }
  clearFace() { this.hasFace = false; }

  /** Draws `frame`, then the face on `points` (pixels, x/y pairs for every landmark) when there is one. */
  draw(frame: TexImageSource, width: number, height: number, points: Float32Array | null, strength: number) {
    const gl = this.gl;
    if (this.canvas.width !== width || this.canvas.height !== height) { this.canvas.width = width; this.canvas.height = height; }
    gl.viewport(0, 0, width, height);
    gl.disable(gl.BLEND);
    gl.useProgram(this.frameProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.frameTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, frame);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (!points || !this.hasFace || strength <= 0.001) return;
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(this.faceProg);
    gl.uniform2f(gl.getUniformLocation(this.faceProg, 'size'), width, height);
    gl.uniform1f(gl.getUniformLocation(this.faceProg, 'strength'), strength);
    gl.bindTexture(gl.TEXTURE_2D, this.faceTex);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, points);
    gl.bindVertexArray(this.vao);
    gl.drawElements(gl.TRIANGLES, this.count, gl.UNSIGNED_SHORT, 0);
    gl.bindVertexArray(null);
  }
}
