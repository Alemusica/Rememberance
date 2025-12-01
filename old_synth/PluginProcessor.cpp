#include "PluginProcessor.h"
#include "PluginEditor.h"

GoldenIntegrationSynthAudioProcessor::GoldenIntegrationSynthAudioProcessor()
    : juce::AudioProcessor (BusesProperties().withOutput("Output", juce::AudioChannelSet::stereo(), true))
{
    // Minimal file logger to diagnose host load issues
    fileLogger.reset(juce::FileLogger::createDefaultAppLogger("GoldenIntegrationSynth",
                                                             "GoldenIntegrationSynth",
                                                             "init"));
    juce::Logger::writeToLog("[GIS] Constructed AudioProcessor");
}

void GoldenIntegrationSynthAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    juce::ignoreUnused(samplesPerBlock);
    currentSR = sampleRate;
    requestedPartials = (int) apvts.getRawParameterValue("partials_n")->load();
    partialSeed = (uint32_t) apvts.getRawParameterValue("partials_seed")->load();
    currentLo = apvts.getRawParameterValue("carrier_lo_hz")->load();
    currentHi = apvts.getRawParameterValue("carrier_hi_hz")->load();
    currentTilt = apvts.getRawParameterValue("tone_tilt_db_per_oct")->load();
    partials.prepare(sampleRate, requestedPartials, partialSeed, currentLo, currentHi, currentTilt);
    modulators.prepare(sampleRate);
    juce::Logger::writeToLog("[GIS] prepareToPlay sr=" + juce::String(sampleRate) +
                             " nPartials=" + juce::String(requestedPartials));

    smDepth1.prepare(sampleRate, 0.01);
    smDepth2.prepare(sampleRate, 0.01);
    smDepthB.prepare(sampleRate, 0.01);
    smSide.prepare(sampleRate, 0.01);
    smAm1.prepare(sampleRate, 0.01);
    smAm2.prepare(sampleRate, 0.01);
    smBridge.prepare(sampleRate, 0.01);
    smDiff.prepare(sampleRate, 0.01);
    smDepthDiff.prepare(sampleRate, 0.01);

    smDepth1.reset(apvts.getRawParameterValue("depth1")->load());
    smDepth2.reset(apvts.getRawParameterValue("depth2")->load());
    smDepthB.reset(apvts.getRawParameterValue("depth_bridge")->load());
    smSide.reset(apvts.getRawParameterValue("side_gain")->load());
    smAm1.reset(apvts.getRawParameterValue("am1_hz")->load());
    smAm2.reset(apvts.getRawParameterValue("am2_hz")->load());
    smBridge.reset(apvts.getRawParameterValue("bridge_hz")->load());
    smDiff.reset(apvts.getRawParameterValue("diff_hz")->load());
    smDepthDiff.reset(apvts.getRawParameterValue("depth_diff")->load());
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool GoldenIntegrationSynthAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;
    return true;
}
#endif

void GoldenIntegrationSynthAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi)
{
    juce::ignoreUnused(midi);
    juce::ScopedNoDenormals d;

    auto mode = (int) apvts.getRawParameterValue("mode")->load();
    // Smooth rate parameters
    smAm1.setTarget(apvts.getRawParameterValue("am1_hz")->load());
    smAm2.setTarget(apvts.getRawParameterValue("am2_hz")->load());
    smBridge.setTarget(apvts.getRawParameterValue("bridge_hz")->load());
    smDiff.setTarget(apvts.getRawParameterValue("diff_hz")->load());

    // Update smoothing targets
    smDepth1.setTarget(apvts.getRawParameterValue("depth1")->load());
    smDepth2.setTarget(apvts.getRawParameterValue("depth2")->load());
    smDepthB.setTarget(apvts.getRawParameterValue("depth_bridge")->load());
    smSide.setTarget(apvts.getRawParameterValue("side_gain")->load());

    // Rebuild partial set if parameters changed
    int pn = (int) apvts.getRawParameterValue("partials_n")->load();
    uint32_t ps = (uint32_t) apvts.getRawParameterValue("partials_seed")->load();
    float lo = apvts.getRawParameterValue("carrier_lo_hz")->load();
    float hi = apvts.getRawParameterValue("carrier_hi_hz")->load();
    float tilt = apvts.getRawParameterValue("tone_tilt_db_per_oct")->load();
    if (pn != requestedPartials || ps != partialSeed || lo != currentLo || hi != currentHi || tilt != currentTilt) {
        requestedPartials = pn; partialSeed = ps; currentLo = lo; currentHi = hi; currentTilt = tilt;
        partials.prepare(currentSR, requestedPartials, partialSeed, currentLo, currentHi, currentTilt);
    }

    auto* l = buffer.getWritePointer(0);
    auto* r = buffer.getWritePointer(1);

    for (int i = 0; i < buffer.getNumSamples(); ++i) {
        float am1 = smAm1.getNext();
        float am2 = smAm2.getNext();
        float brHz = smBridge.getNext();
        float dHz = smDiff.getNext();
        float d1 = smDepth1.getNext();
        float d2 = smDepth2.getNext();
        float dB = smDepthB.getNext();
        float dD = smDepthDiff.getNext();
        float env;
        // Both modes share the same structure; Mode A may ignore bridge depth/diff if 0
        float appliedDiff = dHz > 0.0f ? dHz : std::abs(am2 - am1);
        env = modulators.envA(am1, d1, am2, d2, brHz, dB, appliedDiff, dD);
        float base = partials.next();
        float L = base * env;
        float R = base * env;

        // Mid/Side stereo shaping
        float sideG = smSide.getNext();
        float M = 0.5f * (L + R);
        float S = 0.5f * (L - R);
        S *= sideG;
        L = M + S;
        R = M - S;

        l[i] = limiter.process(L);
        r[i] = limiter.process(R);
    }
    juce::ignoreUnused(mode);
}

void GoldenIntegrationSynthAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    juce::MemoryOutputStream mos(destData, false);
    state.writeToStream(mos);
}

void GoldenIntegrationSynthAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    juce::ValueTree state = juce::ValueTree::readFromData(data, size_t(sizeInBytes));
    if (state.isValid()) {
        apvts.replaceState(state);
    }
}

juce::AudioProcessorValueTreeState::ParameterLayout GoldenIntegrationSynthAudioProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;
    params.push_back(std::make_unique<juce::AudioParameterChoice>("mode", "Mode", juce::StringArray{"A_rational", "B_golden_bridge"}, 0));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("am1_hz", "AM1 Hz", juce::NormalisableRange<float>(0.0f, 60.0f, 0.01f), 8.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("am2_hz", "AM2 Hz", juce::NormalisableRange<float>(0.0f, 60.0f, 0.01f), 16.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("bridge_hz", "Bridge Hz", juce::NormalisableRange<float>(0.0f, 80.0f, 0.01f), 0.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("depth1", "Depth1", juce::NormalisableRange<float>(0.0f, 0.6f, 0.001f), 0.30f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("depth2", "Depth2", juce::NormalisableRange<float>(0.0f, 0.6f, 0.001f), 0.30f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("depth_bridge", "Depth Bridge", juce::NormalisableRange<float>(0.0f, 0.6f, 0.001f), 0.20f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("depth_diff", "Depth Diff", juce::NormalisableRange<float>(0.0f, 0.6f, 0.001f), 0.00f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("diff_hz", "Diff Hz", juce::NormalisableRange<float>(0.0f, 60.0f, 0.01f), 0.0f));
    params.push_back(std::make_unique<juce::AudioParameterInt>("partials_n", "Partials N", 1, 24, 8));
    params.push_back(std::make_unique<juce::AudioParameterInt>("partials_seed", "Partials Seed", 0, 1<<20, 1234));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("side_gain", "Side Gain", juce::NormalisableRange<float>(0.0f, 1.0f, 0.001f), 0.25f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("carrier_lo_hz", "Carrier Lo Hz", juce::NormalisableRange<float>(50.0f, 4000.0f, 0.1f), 200.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("carrier_hi_hz", "Carrier Hi Hz", juce::NormalisableRange<float>(300.0f, 8000.0f, 0.1f), 1200.0f));
    params.push_back(std::make_unique<juce::AudioParameterFloat>("tone_tilt_db_per_oct", "Tone Tilt dB/oct", juce::NormalisableRange<float>(-12.0f, 12.0f, 0.1f), -3.0f));
    return { params.begin(), params.end() };
}

juce::AudioProcessorEditor* GoldenIntegrationSynthAudioProcessor::createEditor()
{
    return new GoldenIntegrationSynthAudioProcessorEditor(*this);
}

void GoldenIntegrationSynthAudioProcessor::applyPreset(const juce::String& id)
{
    auto setF = [this](const juce::String& pid, float v){ if (auto* p = apvts.getParameter(pid)) p->setValueNotifyingHost(p->getNormalisableRange().convertTo0to1(v)); };
    auto setI = [this](const juce::String& pid, int v){ if (auto* p = apvts.getParameter(pid)) p->setValueNotifyingHost(p->getNormalisableRange().convertTo0to1((float)v)); };
    auto setC = [this](const juce::String& pid, int choiceIndex){ if (auto* p = apvts.getParameter(pid)) p->setValueNotifyingHost(p->getNormalisableRange().convertTo0to1((float)choiceIndex)); };

    if (id == "A1_8_16") {
        setC("mode", 0);
        setF("am1_hz", 8.0f);
        setF("am2_hz", 16.0f);
        setF("bridge_hz", 0.0f);
        setF("depth1", 0.30f);
        setF("depth2", 0.30f);
        setF("depth_bridge", 0.0f);
        setI("partials_n", 8);
        setI("partials_seed", 1234);
        setF("side_gain", 0.25f);
    } else if (id == "A2_6_9") {
        setC("mode", 0);
        setF("am1_hz", 6.0f);
        setF("am2_hz", 9.0f);
        setF("bridge_hz", 0.0f);
        setF("depth1", 0.25f);
        setF("depth2", 0.25f);
        setF("depth_bridge", 0.0f);
        setI("partials_n", 8);
        setI("partials_seed", 5678);
        setF("side_gain", 0.25f);
    } else if (id == "B0_phi_triplet_exact_6_11_bridge_18") {
        setC("mode", 1);
        // Exact φ triplet: φ^4, φ^5, φ^6 (modulators = low, mid; bridge = high)
        const double phi = (1.0 + std::sqrt(5.0)) * 0.5;
        const double mid = 11.090169943749475; // φ^5
        const double low = mid / phi;           // φ^4
        const double high = low + mid;          // φ^6
        setF("am1_hz", (float) low);
        setF("am2_hz", (float) mid);
        setF("bridge_hz", (float) high);
        setF("depth1", 0.25f);
        setF("depth2", 0.25f);
        setF("depth_bridge", 0.20f);
        setI("partials_n", 8);
        setI("partials_seed", 9012);
        setF("side_gain", 0.25f);
    } else if (id == "B1_phi_triplet_exact_11_18_bridge_29") {
        setC("mode", 1);
        // Exact φ triplet: φ^5, φ^6, φ^7 (modulators = mid, high; bridge = next)
        const double phi = (1.0 + std::sqrt(5.0)) * 0.5;
        const double mid = 11.090169943749475;  // φ^5
        const double high = mid * phi;          // φ^6
        const double next = mid + high;         // φ^7
        setF("am1_hz", (float) mid);
        setF("am2_hz", (float) high);
        setF("bridge_hz", (float) next);
        setF("depth1", 0.25f);
        setF("depth2", 0.25f);
        setF("depth_bridge", 0.20f);
        setI("partials_n", 8);
        setI("partials_seed", 3456);
        setF("side_gain", 0.25f);
    } else if (id == "B-1_phi_triplet_exact_4_7_bridge_11") {
        setC("mode", 1);
        // Lower φ triplet: φ^3, φ^4, φ^5
        const double phi = (1.0 + std::sqrt(5.0)) * 0.5;
        const double low = 4.23606797749979;     // φ^3
        const double mid = low * phi;            // φ^4
        const double next = low + mid;           // φ^5
        setF("am1_hz", (float) low);
        setF("am2_hz", (float) mid);
        setF("bridge_hz", (float) next);
        setF("depth1", 0.25f);
        setF("depth2", 0.25f);
        setF("depth_bridge", 0.20f);
        setI("partials_n", 8);
        setI("partials_seed", 2222);
        setF("side_gain", 0.25f);
    } else if (id == "B2_phi_triplet_exact_18_29_bridge_47") {
        setC("mode", 1);
        // Upper φ triplet: φ^6, φ^7, φ^8
        const double phi = (1.0 + std::sqrt(5.0)) * 0.5;
        const double mid = 17.94427190999916;    // φ^6
        const double high = mid * phi;           // φ^7
        const double next = mid + high;          // φ^8
        setF("am1_hz", (float) mid);
        setF("am2_hz", (float) high);
        setF("bridge_hz", (float) next);
        setF("depth1", 0.25f);
        setF("depth2", 0.25f);
        setF("depth_bridge", 0.20f);
        setI("partials_n", 8);
        setI("partials_seed", 7777);
        setF("side_gain", 0.25f);
    }
}

// This is the factory function JUCE expects for plugins
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new GoldenIntegrationSynthAudioProcessor();
}
