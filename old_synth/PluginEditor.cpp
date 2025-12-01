#include "PluginProcessor.h"
#include "PluginEditor.h"

GoldenIntegrationSynthAudioProcessorEditor::GoldenIntegrationSynthAudioProcessorEditor (GoldenIntegrationSynthAudioProcessor& p)
    : AudioProcessorEditor (&p), processor (p)
{
    setSize (460, 180);

    titleLabel.setText("Golden Integration Synth", juce::dontSendNotification);
    titleLabel.setJustificationType(juce::Justification::centred);
    titleLabel.setColour(juce::Label::textColourId, juce::Colours::gold);
    titleLabel.setFont(juce::Font(18.0f, juce::Font::bold));
    addAndMakeVisible(titleLabel);

    presetBox.addItem("A1 — 8 & 16 Hz (A)", 1);
    presetBox.addItem("A2 — 6 & 9 Hz (A)", 2);
    // ASCII-only labels to avoid glyph issues in some hosts
    presetBox.addItem("B-1 — phi^3..phi^5 (bridge phi^5)", 3);
    presetBox.addItem("B0  — phi^4..phi^6 (bridge phi^6)", 4);
    presetBox.addItem("B1  — phi^5..phi^7 (bridge phi^7)", 5);
    presetBox.addItem("B2  — phi^6..phi^8 (bridge phi^8)", 6);
    presetBox.setSelectedId(1, juce::dontSendNotification);
    addAndMakeVisible(presetBox);

    applyButton.onClick = [this]{
        switch (presetBox.getSelectedId()) {
            case 1: processor.applyPreset("A1_8_16"); break;
            case 2: processor.applyPreset("A2_6_9"); break;
            case 3: processor.applyPreset("B-1_phi_triplet_exact_4_7_bridge_11"); break;
            case 4: processor.applyPreset("B0_phi_triplet_exact_6_11_bridge_18"); break;
            case 5: processor.applyPreset("B1_phi_triplet_exact_11_18_bridge_29"); break;
            case 6: processor.applyPreset("B2_phi_triplet_exact_18_29_bridge_47"); break;
            default: break;
        }
    };
    addAndMakeVisible(applyButton);

    auto initSlider = [&](juce::Slider& s, const juce::String& suffix){
        s.setSliderStyle(juce::Slider::LinearBar);
        s.setTextBoxStyle(juce::Slider::TextBoxRight, false, 80, 20);
        s.setTextValueSuffix(suffix);
    };

    initSlider(loSlider, " Hz");
    initSlider(hiSlider, " Hz");
    initSlider(tiltSlider, " dB/oct");
    initSlider(depthDiffSlider, "");
    initSlider(diffHzSlider, " Hz");
    initSlider(sideSlider, "");

    addAndMakeVisible(loSlider);
    addAndMakeVisible(hiSlider);
    addAndMakeVisible(tiltSlider);
    addAndMakeVisible(depthDiffSlider);
    addAndMakeVisible(diffHzSlider);
    addAndMakeVisible(sideSlider);

    loAtt.reset(new juce::AudioProcessorValueTreeState::SliderAttachment(processor.apvts, "carrier_lo_hz", loSlider));
    hiAtt.reset(new juce::AudioProcessorValueTreeState::SliderAttachment(processor.apvts, "carrier_hi_hz", hiSlider));
    tiltAtt.reset(new juce::AudioProcessorValueTreeState::SliderAttachment(processor.apvts, "tone_tilt_db_per_oct", tiltSlider));
    depthDiffAtt.reset(new juce::AudioProcessorValueTreeState::SliderAttachment(processor.apvts, "depth_diff", depthDiffSlider));
    diffHzAtt.reset(new juce::AudioProcessorValueTreeState::SliderAttachment(processor.apvts, "diff_hz", diffHzSlider));
    sideAtt.reset(new juce::AudioProcessorValueTreeState::SliderAttachment(processor.apvts, "side_gain", sideSlider));
}

void GoldenIntegrationSynthAudioProcessorEditor::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colours::black);
}

void GoldenIntegrationSynthAudioProcessorEditor::resized()
{
    auto r = getLocalBounds().reduced(12);
    titleLabel.setBounds(r.removeFromTop(28));
    auto row = r.removeFromTop(28);
    presetBox.setBounds(row.removeFromLeft(300));
    row.removeFromLeft(8);
    applyButton.setBounds(row.removeFromLeft(120));

    r.removeFromTop(8);
    auto c1 = r.removeFromTop(24);
    loSlider.setBounds(c1.removeFromLeft(220));
    c1.removeFromLeft(8);
    hiSlider.setBounds(c1.removeFromLeft(220));

    r.removeFromTop(6);
    auto c2 = r.removeFromTop(24);
    tiltSlider.setBounds(c2.removeFromLeft(220));
    c2.removeFromLeft(8);
    sideSlider.setBounds(c2.removeFromLeft(220));

    r.removeFromTop(6);
    auto c3 = r.removeFromTop(24);
    depthDiffSlider.setBounds(c3.removeFromLeft(220));
    c3.removeFromLeft(8);
    diffHzSlider.setBounds(c3.removeFromLeft(220));
}
