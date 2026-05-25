#ifndef __PY_VOICE_INPUT_H_
#define __PY_VOICE_INPUT_H_

#include <glib.h>
#include <pulse/pulseaudio.h>
#include <pulse/simple.h>
#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include "FeatureExtractor.h"

struct OrtEnv;
struct OrtSession;
struct OrtValue;
struct OrtApi;

namespace PY {

class VoiceInput {
public:
    VoiceInput();
    ~VoiceInput();

    gboolean handleKeyEvent(guint keyval, guint keycode, guint modifiers);
    gboolean isRecording() const { return m_recording.load(); }
    std::string getLastResult();

private:
    bool initOnnxRuntime();
    void shutdownOnnxRuntime();
    void startRecording();
    void stopRecording();
    std::string transcribe(const std::vector<int16_t>& samples);
    void recordThread();

    static void streamReadCb(pa_stream* s, size_t nbytes, void* userdata);

    static constexpr int SAMPLE_RATE = 16000;

    std::atomic<bool> m_recording;
    std::atomic<bool> m_stop_requested;
    std::string m_last_result;

    std::chrono::steady_clock::time_point m_last_ctrl_press;
    bool m_ctrl_held_after_double;
    bool m_has_other_key_since_last_ctrl;

    std::thread m_record_thread;
    std::mutex m_result_mutex;

    void* m_dl_handle;
    const OrtApi* m_api;
    OrtEnv* m_env;
    OrtSession* m_session;
    bool m_model_loaded;

    std::vector<int16_t> m_record_buffer;
    std::mutex m_buffer_mutex;

    CMVNStats m_cmvn;
    std::vector<std::string> m_tokens;

    std::string m_model_path;
    std::string m_cmvn_path;
    std::string m_punc_model_path;
    std::vector<int> m_punc_tokens;
};

}

#endif
