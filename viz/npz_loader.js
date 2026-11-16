/**
 * Read a NumPy .npz in the browser and map EMG arrays onto the viewer blocks.
 * Expects emg_data (samples × channels) plus emg_timestamps. IMU arrays are skipped.
 */
import { datasetFromSides } from "./parquet_loader.js";

export { datasetFromSides };

const NPY_MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
const DATA_KEYS = ["emg_data", "emg", "samples"];
const TIME_KEYS = ["emg_timestamps", "emg_timestamp", "timestamps", "timestamp", "time"];
const NOMINAL_FS = 2000;

const DTYPES = {
  "|i1": { size: 1, read: (view, offset) => view.getInt8(offset) },
  "|u1": { size: 1, read: (view, offset) => view.getUint8(offset) },
  "<i2": { size: 2, read: (view, offset) => view.getInt16(offset, true) },
  "<u2": { size: 2, read: (view, offset) => view.getUint16(offset, true) },
  "<i4": { size: 4, read: (view, offset) => view.getInt32(offset, true) },
  "<u4": { size: 4, read: (view, offset) => view.getUint32(offset, true) },
  "<f4": { size: 4, read: (view, offset) => view.getFloat32(offset, true) },
  "<f8": { size: 8, read: (view, offset) => view.getFloat64(offset, true) },
  "<i8": { size: 8, read: (view, offset) => Number(view.getBigInt64(offset, true)) },
  "<u8": { size: 8, read: (view, offset) => Number(view.getBigUint64(offset, true)) },
};

export function roleFromFilename(filename) {
  const base = String(filename).split(/[/\\]/).pop() || "";
  if (/right|右腕|右手/i.test(base)) return "right";
  if (/left|左腕|左手/i.test(base)) return "left";
  return "left";
}

export async function streamFromNpz(buffer, filename) {
  const arrays = await readNpz(buffer);
  const dataKey = pickDataKey(arrays);
  if (!dataKey) throw new Error(`${filename} 里没有 emg_data`);
  const packed = asChannels(arrays[dataKey]);
  if (packed.n < 2) throw new Error(`${filename} 的样本太少`);
  const timeKey = pickTimeKey(arrays, dataKey, packed.n);
  const notes = [];
  let time;
  if (timeKey) {
    time = relativeTime(arrays[timeKey].data);
  } else {
    time = new Float64Array(packed.n);
    for (let i = 0; i < packed.n; i++) time[i] = i / NOMINAL_FS;
    notes.push("没有时间戳，已按 2000 Hz 排列");
  }
  if (Object.keys(arrays).some((key) => /^imu_/i.test(key))) {
    notes.push("未显示 IMU（采样率与 EMG 不同）");
  }
  const prefix = roleFromFilename(filename) === "right" ? "R" : "L";
  const channels = packed.channels;
  return {
    role: roleFromFilename(filename),
    note: notes.join("；"),
    block: {
      kind: "emg",
      names: channels.map((_, index) => `${prefix}${index}`),
      channels,
      mean: channels.map(mean),
      std: channels.map(std),
      time,
      fs_hz: sampleRateHz(time),
      duration_s: time[packed.n - 1] - time[0],
      filename,
    },
  };
}

export async function readNpz(buffer) {
  const entries = await unzip(buffer);
  const arrays = {};
  for (const [name, bytes] of entries) {
    if (!name.toLowerCase().endsWith(".npy")) continue;
    const key = name.slice(0, -4).split("/").pop();
    if (!key || key.startsWith("._")) continue;
    arrays[key] = parseNpy(bytes);
  }
  if (!Object.keys(arrays).length) throw new Error("npz 里没有 npy 数组");
  return arrays;
}

export function parseNpy(bytes) {
  const data = asBytes(bytes);
  if (data.length < 10) throw new Error("npy 太短");
  for (let i = 0; i < NPY_MAGIC.length; i++) {
    if (data[i] !== NPY_MAGIC[i]) throw new Error("不是 npy 数组");
  }
  const major = data[6];
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  let headerLen;
  let offset;
  if (major === 1) {
    headerLen = view.getUint16(8, true);
    offset = 10 + headerLen;
  } else if (major === 2 || major === 3) {
    headerLen = view.getUint32(8, true);
    offset = 12 + headerLen;
  } else {
    throw new Error(`不支持的 npy 版本 ${major}`);
  }
  if (offset > data.length) throw new Error("npy 头损坏");
  const header = new TextDecoder("latin1").decode(data.subarray(offset - headerLen, offset));
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const dtype = descr ? DTYPES[descr] : null;
  if (!dtype) throw new Error(`不支持的 dtype ${descr || "未知"}`);
  const fortran = /'fortran_order':\s*True/.test(header);
  const shapeText = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1] ?? "";
  const shape = shapeText
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => Number(part));
  if (!shape.length || shape.some((dim) => !Number.isInteger(dim) || dim < 0)) {
    throw new Error("npy shape 无法解析");
  }
  const count = shape.reduce((total, dim) => total * dim, 1);
  const needed = offset + count * dtype.size;
  if (needed > data.length) throw new Error("npy 数据截断");
  const values = new Float64Array(count);
  for (let i = 0; i < count; i++) values[i] = dtype.read(view, offset + i * dtype.size);
  return { shape, fortran, data: values };
}

function pickDataKey(arrays) {
  const known = findKey(arrays, DATA_KEYS);
  if (known) return known;
  const ranked = Object.keys(arrays)
    .filter((key) => !/imu|arrival|(^|_)sn$/i.test(key) && !key.startsWith("._"))
    .map((key) => ({ key, array: arrays[key] }))
    .filter(({ array }) => array.shape.length === 1 || array.shape.length === 2)
    .sort((a, b) => {
      const rank = (array) => (array.shape.length === 2 ? 2 : 1);
      return rank(b.array) - rank(a.array) || b.array.data.length - a.array.data.length;
    });
  return ranked[0]?.key || null;
}

function pickTimeKey(arrays, dataKey, n) {
  const known = findKey(arrays, TIME_KEYS);
  if (known && isSeries(arrays[known], n)) return known;
  return (
    Object.keys(arrays).find(
      (key) => key !== dataKey && /time/i.test(key) && isSeries(arrays[key], n)
    ) || null
  );
}

function findKey(arrays, names) {
  const keys = Object.keys(arrays);
  for (const name of names) {
    const hit = keys.find((key) => key.toLowerCase() === name);
    if (hit) return hit;
  }
  return null;
}

function isSeries(array, n) {
  return array.shape.length === 1 && array.shape[0] === n;
}

function asChannels(array) {
  const { shape, fortran, data } = array;
  if (shape.length === 1) {
    return { n: shape[0], channels: [Float32Array.from(data)] };
  }
  if (shape.length !== 2) {
    throw new Error(`emg_data 是 ${shape.join("×")} 维，需要 1 维或 2 维`);
  }
  const [first, second] = shape;
  const sampleAxis = first >= second ? 0 : 1;
  const n = shape[sampleAxis];
  const channels = shape[1 - sampleAxis];
  if (channels < 1 || channels > 64) throw new Error(`通道数 ${channels} 无法显示`);
  const columns = Array.from({ length: channels }, () => new Float32Array(n));
  for (let i = 0; i < n; i++) {
    for (let channel = 0; channel < channels; channel++) {
      const coords = sampleAxis === 0 ? [i, channel] : [channel, i];
      columns[channel][i] = data[flatIndex(shape, fortran, coords)];
    }
  }
  return { n, channels: columns };
}

function flatIndex(shape, fortran, coords) {
  let index = 0;
  if (fortran) {
    let stride = 1;
    for (let axis = 0; axis < shape.length; axis++) {
      index += coords[axis] * stride;
      stride *= shape[axis];
    }
    return index;
  }
  let stride = 1;
  for (let axis = shape.length - 1; axis >= 0; axis--) {
    index += coords[axis] * stride;
    stride *= shape[axis];
  }
  return index;
}

function zip64Sizes(extra, compressedSize, uncompressedSize) {
  const view = new DataView(extra.buffer, extra.byteOffset, extra.byteLength);
  let cursor = 0;
  while (cursor + 4 <= extra.length) {
    const id = view.getUint16(cursor, true);
    const size = view.getUint16(cursor + 2, true);
    const body = cursor + 4;
    if (body + size > extra.length) break;
    if (id === 1) {
      let pos = body;
      if (uncompressedSize === 0xffffffff) {
        uncompressedSize = Number(view.getBigUint64(pos, true));
        pos += 8;
      }
      if (compressedSize === 0xffffffff) {
        compressedSize = Number(view.getBigUint64(pos, true));
      }
      return { compressedSize, uncompressedSize };
    }
    cursor = body + size;
  }
  throw new Error("npz 使用了 Zip64，但缺少扩展长度");
}

async function unzip(buffer) {
  const bytes = asBytes(buffer);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const files = new Map();
  let offset = 0;
  while (offset + 30 <= bytes.length) {
    const signature = view.getUint32(offset, true);
    if (signature === 0x02014b50 || signature === 0x06054b50) break;
    if (signature !== 0x04034b50) throw new Error("npz 不是 zip 压缩包");
    const flags = view.getUint16(offset + 6, true);
    const method = view.getUint16(offset + 8, true);
    let compressedSize = view.getUint32(offset + 18, true);
    let uncompressedSize = view.getUint32(offset + 22, true);
    const nameLength = view.getUint16(offset + 26, true);
    const extraLength = view.getUint16(offset + 28, true);
    if (flags & 0x8) throw new Error("不支持带 data descriptor 的 npz");
    const nameStart = offset + 30;
    const extra = bytes.subarray(nameStart + nameLength, nameStart + nameLength + extraLength);
    if (compressedSize === 0xffffffff || uncompressedSize === 0xffffffff) {
      const zip64 = zip64Sizes(extra, compressedSize, uncompressedSize);
      compressedSize = zip64.compressedSize;
      uncompressedSize = zip64.uncompressedSize;
    }
    const dataStart = nameStart + nameLength + extraLength;
    if (dataStart + compressedSize > bytes.length) throw new Error("npz 数据截断");
    const name = new TextDecoder("utf-8").decode(bytes.subarray(nameStart, nameStart + nameLength));
    const compressed = bytes.subarray(dataStart, dataStart + compressedSize);
    let raw;
    if (method === 0) raw = compressed;
    else if (method === 8) raw = await inflateRaw(compressed);
    else throw new Error(`不支持的 npz 压缩方式 ${method}`);
    if (uncompressedSize !== 0xffffffff && raw.length !== uncompressedSize) {
      throw new Error(`npy ${name} 解压后长度不符`);
    }
    const base = name.replace(/\\/g, "/").split("/").pop();
    if (base && !base.endsWith("/")) files.set(base, raw);
    offset = dataStart + compressedSize;
  }
  if (!files.size) throw new Error("npz 是空的");
  return files;
}

async function inflateRaw(bytes) {
  if (typeof DecompressionStream !== "function") {
    throw new Error("当前浏览器不能解压 npz，请改用未压缩的 np.savez");
  }
  try {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  } catch {
    throw new Error("当前浏览器不能解压 npz，请改用未压缩的 np.savez");
  }
}

function asBytes(buffer) {
  if (buffer instanceof Uint8Array) return buffer;
  return new Uint8Array(buffer);
}

function relativeTime(values) {
  const out = new Float64Array(values.length);
  let origin = NaN;
  for (let i = 0; i < values.length; i++) {
    out[i] = values[i];
    if (!Number.isFinite(origin) && Number.isFinite(values[i])) origin = values[i];
  }
  if (!Number.isFinite(origin)) throw new Error("timestamp 无效");
  for (let i = 0; i < out.length; i++) out[i] -= origin;
  return out;
}

function sampleRateHz(time) {
  const n = time.length;
  if (n < 2) return 0;
  const stride = Math.max(1, Math.floor((n - 1) / 8000));
  const diffs = [];
  for (let i = stride; i < n; i += stride) {
    const dt = time[i] - time[i - 1];
    if (dt > 0) diffs.push(dt);
  }
  if (!diffs.length) return 0;
  diffs.sort((a, b) => a - b);
  return 1 / diffs[Math.floor(diffs.length / 2)];
}

function mean(arr) {
  let sum = 0;
  for (let i = 0; i < arr.length; i++) sum += arr[i];
  return arr.length ? sum / arr.length : 0;
}

function std(arr) {
  const center = mean(arr);
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    const delta = arr[i] - center;
    sum += delta * delta;
  }
  return Math.sqrt(sum / (arr.length || 1));
}
