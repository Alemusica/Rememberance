#include "Core.h"
#include <algorithm>

namespace golden {

static inline float tiltFactor(float f, float tiltDbPerOct, float refHz=1000.0f) {
    if (std::abs(tiltDbPerOct) < 1e-6f) return 1.0f;
    float log2ratio = std::log2(std::max(f, 1e-3f) / refHz);
    return std::pow(10.0f, (tiltDbPerOct * log2ratio) / 20.0f);
}

void Partials::prepare(double sampleRate, int nPartials, uint32_t seed, float low, float high, float tiltDbPerOct) {
    sr = sampleRate;
    lo = low; hi = high; tilt = tiltDbPerOct;
    freqs.resize(nPartials);
    amps.resize(nPartials);
    phases.resize(nPartials);
    incs.resize(nPartials);
    // Simple LCG
    auto rng = [state = seed]() mutable {
        state = 1664525u * state + 1013904223u;
        return state;
    };
    for (int i = 0; i < nPartials; ++i) {
        float u = (rng() / float(UINT32_MAX));
        freqs[i] = low + u * (high - low);
    }
    std::sort(freqs.begin(), freqs.end());
    // 1/k roll-off with tilt across frequency
    float sum = 0.0f;
    for (int k = 0; k < nPartials; ++k) { sum += (1.0f / float(k + 1)) * tiltFactor(freqs[k], tiltDbPerOct); }
    for (int k = 0; k < nPartials; ++k) {
        amps[k] = ((1.0f / float(k + 1)) * tiltFactor(freqs[k], tiltDbPerOct)) / sum;
        phases[k] = (rng() / float(UINT32_MAX)) * 2.0f * float(M_PI);
        incs[k] = 2.0f * float(M_PI) * freqs[k] / float(sr);
    }
}

} // namespace golden
