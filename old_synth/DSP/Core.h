// Minimal DSP core for Golden Integration Synth
#pragma once

#include <vector>
#include <cmath>
#include <cstdint>

namespace golden {

inline float dbToLin(float dB) { return std::pow(10.0f, dB / 20.0f); }

struct Smoothed {
    void prepare(double sampleRate, double timeSeconds) {
        sr = sampleRate;
        steps = std::max<int>(1, int(timeSeconds * sr));
        remain = 0;
        cur = target;
        inc = 0.0;
    }
    void reset(double value) {
        cur = target = value; remain = 0; inc = 0.0; }
    void setTarget(double value) {
        target = value;
        remain = steps;
        inc = (target - cur) / double(std::max<int>(1, remain));
    }
    inline float getNext() {
        if (remain > 0) { cur += inc; --remain; }
        return (float) cur;
    }
    double sr { 48000.0 };
    int steps { 1 };
    int remain { 0 };
    double cur { 0.0 };
    double target { 0.0 };
    double inc { 0.0 };
};

struct Partials {
    void prepare(double sampleRate, int nPartials, uint32_t seed, float low=300.0f, float high=1500.0f, float tiltDbPerOct=0.0f);
    inline float next() {
        float s = 0.0f;
        for (size_t i = 0; i < freqs.size(); ++i) {
            phases[i] += incs[i];
            if (phases[i] > float(2.0 * M_PI)) phases[i] -= float(2.0 * M_PI);
            s += amps[i] * std::sin(phases[i]);
        }
        return s;
    }
    std::vector<float> freqs, amps, phases, incs;
    double sr { 48000.0 };
    float lo { 300.0f }, hi { 1500.0f }, tilt { 0.0f };
};

struct EnvState {
    double sr { 48000.0 };
    double p1 { 0.0 }, p2 { 0.0 }, pB { 0.0 };
    inline float stepSine(float freqHz) {
        if (freqHz <= 0.0f) return 0.0f;
        double inc = 2.0 * M_PI * freqHz / sr;
        p1 += inc; if (p1 > 2.0 * M_PI) p1 -= 2.0 * M_PI;
        return std::sin((float)p1);
    }
};

inline float raisedSine(float phase01) {
    // phase01 in [0..1)
    return 0.5f * (1.0f + std::sin(2.0f * float(M_PI) * phase01));
}

struct Modulators {
    void prepare(double sampleRate) {
        sr = sampleRate;
        p1 = p2 = pB = 0.0;
        pD = 0.0;
    }
    inline float envA(float am1Hz, float d1, float am2Hz, float d2, float bridgeHz, float dB,
                      float diffHz=0.0f, float dDiff=0.0f) {
        float e = 1.0f;
        if (am1Hz > 0.0f && d1 > 0.0f) {
            p1 += 2.0 * M_PI * am1Hz / sr; if (p1 > 2.0 * M_PI) p1 -= 2.0 * M_PI;
            e *= 1.0f - d1 + d1 * 0.5f * (1.0f + std::sin((float)p1));
        }
        if (am2Hz > 0.0f && d2 > 0.0f) {
            p2 += 2.0 * M_PI * am2Hz / sr; if (p2 > 2.0 * M_PI) p2 -= 2.0 * M_PI;
            e *= 1.0f - d2 + d2 * 0.5f * (1.0f + std::sin((float)p2));
        }
        if (bridgeHz > 0.0f && dB > 0.0f) {
            pB += 2.0 * M_PI * bridgeHz / sr; if (pB > 2.0 * M_PI) pB -= 2.0 * M_PI;
            e *= 1.0f - dB + dB * 0.5f * (1.0f + std::sin((float)pB));
        }
        if (diffHz > 0.0f && dDiff > 0.0f) {
            pD += 2.0 * M_PI * diffHz / sr; if (pD > 2.0 * M_PI) pD -= 2.0 * M_PI;
            e *= 1.0f - dDiff + dDiff * 0.5f * (1.0f + std::sin((float)pD));
        }
        return e;
    }
    double sr { 48000.0 };
    double p1 { 0.0 }, p2 { 0.0 }, pB { 0.0 }, pD { 0.0 };
};

struct Limiter {
    float softK { 1.5f };
    float headroomDb { -1.0f };
    inline float process(float x) const {
        float y = std::tanh(softK * x) / std::tanh(softK);
        return y * dbToLin(headroomDb);
    }
};

} // namespace golden
