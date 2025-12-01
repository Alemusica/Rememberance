#pragma once

namespace golden {

struct GoldenBridgeConfig {
    float am1Hz { 6.85f };
    float am2Hz { 11.10f };
    float bridgeHz { 29.0f }; // approx f1 + f2
    float d1 { 0.25f };
    float d2 { 0.25f };
    float dBridge { 0.20f };
};

} // namespace golden

