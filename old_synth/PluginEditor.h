#pragma once

#include <JuceHeader.h>

class GoldenIntegrationSynthAudioProcessor;

class GoldenIntegrationSynthAudioProcessorEditor  : public juce::AudioProcessorEditor
{
public:
    explicit GoldenIntegrationSynthAudioProcessorEditor (GoldenIntegrationSynthAudioProcessor& p);
    ~GoldenIntegrationSynthAudioProcessorEditor() override = default;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    GoldenIntegrationSynthAudioProcessor& processor;
    juce::ComboBox presetBox;
    juce::TextButton applyButton { "Apply Preset" };
    juce::Label titleLabel;

    // New parameter controls (minimal)
    juce::Slider loSlider, hiSlider, tiltSlider, depthDiffSlider, diffHzSlider, sideSlider;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> loAtt, hiAtt, tiltAtt, depthDiffAtt, diffHzAtt, sideAtt;
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (GoldenIntegrationSynthAudioProcessorEditor)
};
