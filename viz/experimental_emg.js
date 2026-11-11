/* Viewer-only offline experiment. No file writes; legacy DSP is independent. */
(function (root) {
  "use strict";
  const FS = 2000, TAPS = 8001, HALF = 4000, FFT = 32768, HOP = FFT - TAPS + 1;
  function fft(re, im, inverse = false) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
    }
    for (let len = 2; len <= n; len *= 2) {
      const angle = (inverse ? 2 : -2) * Math.PI / len, cr = Math.cos(angle), ci = Math.sin(angle);
      for (let i = 0; i < n; i += len) {
        let wr = 1, wi = 0;
        for (let k = 0; k < len / 2; k++) {
          const a = i + k, b = a + len / 2, vr = re[b] * wr - im[b] * wi, vi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - vr; im[b] = im[a] - vi; re[a] += vr; im[a] += vi;
          const next = wr * cr - wi * ci; wi = wr * ci + wi * cr; wr = next;
        }
      }
    }
    if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  }
  function biquad(cutoff, q, high) {
    const w = 2 * Math.PI * cutoff / FS, c = Math.cos(w), a = Math.sin(w) / (2 * q), d = 1 + a;
    const b = (high ? 1 + c : 1 - c) / 2;
    return [b / d, (high ? -2 : 2) * b / d, b / d, -2 * c / d, (1 - a) / d];
  }
  function design() {
    const factor = Math.pow(Math.SQRT2 - 1, 1 / 4);
    const hp = FS / Math.PI * Math.atan(Math.tan(Math.PI * 20 / FS) * factor);
    const lp = FS / Math.PI * Math.atan(Math.tan(Math.PI * 450 / FS) / Math.pow(Math.SQRT2 - 1, 1 / 8));
    return { hp, lp, sos: [biquad(hp, Math.SQRT1_2, true), biquad(lp, 1 / (2 * Math.cos(Math.PI / 8)), false), biquad(lp, 1 / (2 * Math.cos(3 * Math.PI / 8)), false)] };
  }
  function filter(sos, x) {
    const y = Float64Array.from(x);
    for (const [b0, b1, b2, a1, a2] of sos) {
      const dc = (b0 + b1 + b2) / (1 + a1 + a2), first = y[0];
      let z0 = (dc - b0) * first, z1 = (b2 - a2 * dc) * first;
      for (let i = 0; i < y.length; i++) {
        const v = y[i], out = b0 * v + z0;
        z0 = b1 * v - a1 * out + z1; z1 = b2 * v - a2 * out; y[i] = out;
      }
    }
    return y;
  }
  function bandpass(x) {
    const pad = Math.min(x.length - 1, HALF), z = new Float64Array(x.length + 2 * pad);
    z.set(x, pad);
    for (let i = 0; i < pad; i++) { z[pad - 1 - i] = 2 * x[0] - x[i + 1]; z[pad + x.length + i] = 2 * x[x.length - 1] - x[x.length - 2 - i]; }
    const sos = design().sos, y = filter(sos, filter(sos, z).reverse()).reverse();
    return y.slice(pad, pad + x.length);
  }
  function median(a) { a.sort((x, y) => x - y); const n = a.length; return n % 2 ? a[n >> 1] : (a[n / 2 - 1] + a[n / 2]) / 2; }
  function psd(x) {
    const win = 8000, nfft = 8192, bins = nfft / 2 + 1, p = new Float64Array(bins);
    const margin = Math.min(8000, Math.floor((x.length - win) / 2));
    if (margin < HALF) throw new Error("片段不足 8 秒，无法可靠自适应检测");
    const re = new Float64Array(nfft), im = new Float64Array(nfft), hann = new Float64Array(win);
    for (let i = 0; i < win; i++) hann[i] = .5 - .5 * Math.cos(2 * Math.PI * i / win);
    let count = 0;
    for (let start = margin; start + win <= x.length - margin; start += win / 2) {
      let mean = 0;
      for (let i = 0; i < win; i++) mean += x[start + i] / win;
      re.fill(0); im.fill(0);
      for (let i = 0; i < win; i++) re[i] = (x[start + i] - mean) * hann[i];
      fft(re, im);
      for (let k = 0; k < bins; k++) p[k] += re[k] ** 2 + im[k] ** 2;
      count++;
    }
    for (let k = 0; k < bins; k++) p[k] /= count;
    return { p, df: FS / nfft };
  }
  function excess(spectrum, h, near, lo, hi) {
    const { p, df } = spectrum; let peak = 0; const background = [];
    for (let k = Math.max(0, Math.ceil((h - hi) / df)); k <= Math.min(p.length - 1, Math.floor((h + hi) / df)); k++) {
      const d = Math.abs(k * df - h);
      if (d <= near) peak = Math.max(peak, p[k]);
      if (d >= lo && d <= hi) background.push(p[k]);
    }
    return 10 * Math.log10((peak + 1e-30) / (median(background) + 1e-30));
  }
  function i0(x) { let term = 1, sum = 1; for (let k = 1; k < 50; k++) { term *= (x * x / 4) / (k * k); sum += term; if (term < sum * 1e-16) break; } return sum; }
  function sinc(x) { return Math.abs(x) < 1e-12 ? 1 : Math.sin(Math.PI * x) / (Math.PI * x); }
  // Analytic trapezoid impulse response, Kaiser windowed (same target as firwin2).
  function kernel(widths) {
    const b = new Float64Array(TAPS), norm = i0(8.6);
    for (let i = 0; i < TAPS; i++) {
      const k = i - HALF, t = k / FS; let v = k === 0 ? 1 : 0;
      for (let j = 0; j < 9; j++) if (widths[j]) {
        const a = 2 * widths[j] + 1.5;
        v -= 2 * a / FS * sinc(a * t) * sinc(1.5 * t) * Math.cos(2 * Math.PI * (j + 1) * 50 * t);
      }
      b[i] = v * i0(8.6 * Math.sqrt(Math.max(0, 1 - (k / HALF) ** 2))) / norm;
    }
    return b;
  }
  function convolver(y) {
    const z = new Float64Array(y.length + 2 * HALF); z.set(y, HALF);
    for (let i = 0; i < HALF; i++) { z[HALF - 1 - i] = y[i + 1]; z[HALF + y.length + i] = y[y.length - 2 - i]; }
    const chunks = [];
    for (let start = 0; start < z.length; start += HOP) {
      const re = new Float64Array(FFT), im = new Float64Array(FFT), len = Math.min(HOP, z.length - start);
      re.set(z.subarray(start, start + len)); fft(re, im); chunks.push({ re, im, start, len });
    }
    return (b) => {
      const hr = new Float64Array(FFT), hi = new Float64Array(FFT); hr.set(b); fft(hr, hi);
      const full = new Float64Array(z.length + TAPS - 1);
      for (const chunk of chunks) {
        const re = new Float64Array(FFT), im = new Float64Array(FFT);
        for (let k = 0; k < FFT; k++) { re[k] = chunk.re[k] * hr[k] - chunk.im[k] * hi[k]; im[k] = chunk.re[k] * hi[k] + chunk.im[k] * hr[k]; }
        fft(re, im, true);
        for (let k = 0; k < chunk.len + TAPS - 1; k++) full[chunk.start + k] += re[k];
      }
      return full.slice(2 * HALF, 2 * HALF + y.length);
    };
  }
  function processChannel(samples, time, progress = () => {}) {
    const result = Float64Array.from(samples), segments = [], n = samples.length;
    if (time.length !== n) throw new Error("时间戳与样本数不一致");
    for (let i = 0; i < n; i++) if (!Number.isFinite(samples[i]) || !Number.isFinite(time[i]) || (i && time[i] <= time[i - 1])) throw new Error("实验方案要求有限信号和严格递增时间戳");
    const bounds = [0];
    for (let i = 1; i < n; i++) if (time[i] - time[i - 1] > 3 / FS) bounds.push(i);
    bounds.push(n);
    for (let s = 0; s < bounds.length - 1; s++) {
      const start = bounds[s], end = bounds[s + 1], rec = { start, end, widths: [], unresolved: [], validStart: start, validEnd: start, short: end - start < 8 * FS };
      if (!rec.short) {
        const y = bandpass(samples.subarray(start, end)), spectrum = psd(y), selected = Array.from({ length: 9 }, (_, j) => excess(spectrum, (j + 1) * 50, 1, 3, 8) > 6);
        const widths = selected.map(v => v ? 4 : 0), responses = [], conv = convolver(y);
        for (let width = 1; width <= 4; width++) {
          const candidate = conv(kernel(selected.map(v => v ? width : 0))), p = psd(candidate);
          responses.push(selected.map((_, j) => excess(p, (j + 1) * 50, 5, 7, 12)));
          progress(`片段 ${s + 1}/${bounds.length - 1} · 半宽 ${width}/4 Hz`);
        }
        for (let j = 0; j < 9; j++) if (selected[j]) {
          const chosen = responses.findIndex(v => v[j] <= 6); widths[j] = chosen < 0 ? 4 : chosen + 1;
        }
        const output = conv(kernel(widths)), final = psd(output);
        rec.widths = widths; rec.qcDb = selected.map((_, j) => excess(final, (j + 1) * 50, 5, 7, 12));
        rec.unresolved = rec.qcDb.map((v, j) => widths[j] && v > 6 ? (j + 1) * 50 : 0).filter(Boolean);
        rec.validStart = start + HALF; rec.validEnd = end - HALF;
        result.set(output, start);
      }
      segments.push(rec);
    }
    return { samples: result, segments };
  }
  const api = { fft, design, bandpass, kernel, convolver, psd, excess, processChannel };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof WorkerGlobalScope !== "undefined" && root instanceof WorkerGlobalScope) {
    root.onmessage = ({ data }) => {
      try {
        const sides = {};
        for (const block of data.sides) {
          let elapsed = 0, intervals = 0;
          for (let i = 1; i < block.time.length; i++) {
            const dt = block.time[i] - block.time[i - 1];
            if (dt > 0 && dt <= 3 / FS) { elapsed += dt; intervals++; }
          }
          const nativeRate = elapsed > 0 ? intervals / elapsed : block.meanRate;
          if (!(nativeRate >= 1900 && nativeRate <= 2100)) throw new Error(`${block.key} 不是约 2 kHz EMG，拒绝套用实验方案`);
          const channels = [], diagnostics = [];
          for (let c = 0; c < block.channels.length; c++) {
            const output = processChannel(block.channels[c], block.time, detail => root.postMessage({ progress: `${block.key} 通道 ${c + 1}/${block.channels.length} · ${detail}` }));
            channels.push(output.samples); diagnostics.push(output.segments);
          }
          sides[block.key] = { channels, diagnostics };
        }
        root.postMessage({ sides }, Object.values(sides).flatMap(b => b.channels.map(c => c.buffer)));
      } catch (error) { root.postMessage({ error: error.message }); }
    };
  }
})(typeof self !== "undefined" ? self : globalThis);
