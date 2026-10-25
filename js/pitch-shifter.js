// AudioWorklet that shifts pitch without changing speed. Two read heads
// sweep through a short delay line at the new rate; each is faded in and
// out with a Hann window, so the jump where a head wraps around is silent
// and the two always add up to full volume.
const WINDOW_SECONDS = 0.1;
const DRY_FADE_SECONDS = 0.05;

class PitchShifterProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      {
        name: "ratio",
        defaultValue: 1,
        minValue: 0.25,
        maxValue: 4,
        automationRate: "k-rate",
      },
    ];
  }

  constructor() {
    super();
    this.windowSize = Math.round(sampleRate * WINDOW_SECONDS);
    let size = 1;
    while (size < this.windowSize * 2) size *= 2;
    this.mask = size - 1;
    this.buffers = [new Float32Array(size), new Float32Array(size)];
    this.writeIndex = 0;
    this.phase = 0;
    // At ratio 1 the two heads would comb-filter, so the untouched signal
    // is crossfaded in instead.
    this.wet = 0;
  }

  read(buffer, delay) {
    const position = this.writeIndex - delay;
    const base = Math.floor(position);
    const frac = position - base;
    const a = buffer[base & this.mask];
    const b = buffer[(base + 1) & this.mask];
    return a + (b - a) * frac;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    const ratio = parameters.ratio[0];
    const targetWet = Math.abs(ratio - 1) < 1e-4 ? 0 : 1;
    const wetStep = 1 / (sampleRate * DRY_FADE_SECONDS);
    const phaseStep = (1 - ratio) / this.windowSize;
    const frames = output[0].length;

    for (let i = 0; i < frames; i++) {
      for (let c = 0; c < 2; c++) {
        const channel = input[c] || input[0];
        this.buffers[c][this.writeIndex] = channel ? channel[i] : 0;
      }

      if (this.wet < targetWet) this.wet = Math.min(targetWet, this.wet + wetStep);
      else if (this.wet > targetWet) this.wet = Math.max(targetWet, this.wet - wetStep);

      const phaseA = this.phase;
      const phaseB = phaseA < 0.5 ? phaseA + 0.5 : phaseA - 0.5;
      const s = Math.sin(Math.PI * phaseA);
      const gainA = s * s;
      const gainB = 1 - gainA;
      const delayA = phaseA * this.windowSize;
      const delayB = phaseB * this.windowSize;

      for (let c = 0; c < output.length; c++) {
        const buffer = this.buffers[Math.min(c, 1)];
        const dry = buffer[this.writeIndex];
        const shifted =
          this.read(buffer, delayA) * gainA + this.read(buffer, delayB) * gainB;
        output[c][i] = dry + (shifted - dry) * this.wet;
      }

      this.phase += phaseStep;
      if (this.phase >= 1) this.phase -= 1;
      else if (this.phase < 0) this.phase += 1;
      this.writeIndex = (this.writeIndex + 1) & this.mask;
    }
    return true;
  }
}

registerProcessor("pitch-shifter", PitchShifterProcessor);
