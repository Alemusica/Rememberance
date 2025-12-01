#pragma once

#include <JuceHeader.h>
#include "DSP/Core.h"

class GoldenIntegrationSynthAudioProcessor  : public juce::AudioProcessor
{
public:
    GoldenIntegrationSynthAudioProcessor();
    ~GoldenIntegrationSynthAudioProcessor() override = default;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
   #ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
   #endif
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    //==============================================================================
    const juce::String getName() const override { return JucePlugin_Name; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    //==============================================================================
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int index) override {}
    const juce::String getProgramName (int index) override { return {}; }
    void changeProgramName (int index, const juce::String& newName) override {}

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    juce::AudioProcessorValueTreeState apvts { *this, nullptr, "PARAMS", createParameterLayout() };

    // Apply curated presets aligned with the cited literature
    void applyPreset(const juce::String& presetId);

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    golden::Partials partials;
    golden::Modulators modulators;
    golden::Limiter limiter;
    golden::Smoothed smDepth1, smDepth2, smDepthB, smDepthDiff, smSide;
    golden::Smoothed smAm1, smAm2, smBridge, smDiff;

    int requestedPartials { 8 };
    uint32_t partialSeed { 1234 };
    double currentSR { 48000.0 };

    std::unique_ptr<juce::FileLogger> fileLogger;

    float currentLo { 200.0f }, currentHi { 1200.0f }, currentTilt { -3.0f };

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (GoldenIntegrationSynthAudioProcessor)
};
