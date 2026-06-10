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
    ~VoiceInput();

    gboolean handleKeyEvent(guint keyval, guint keycode, guint modifiers);
    gboolean isRecording() const { return m_recording.load(); }
    std::string getLastResult();

    static VoiceInput & instance(void) { return *m_instance; }
    static void init(void);
    static void finalize(void);

private:
    VoiceInput();
    bool initOnnxRuntime();
    void shutdownOnnxRuntime();
    void startRecording();
    void stopRecording();
    std::string transcribe(const std::vector<int16_t>& samples,
                           std::vector<bool>& space_before);
    void recordThread();

    static void streamReadCb(pa_stream* s, size_t nbytes, void* userdata);

    static constexpr int SAMPLE_RATE = 16000;

    std::atomic<bool> m_recording;
    std::atomic<bool> m_stop_requested;
    std::string m_last_result;

    std::chrono::steady_clock::time_point m_ctrl_press_time;
    bool m_ctrl_pending = false;

    std::thread m_record_thread;
    std::mutex m_result_mutex;

    void* m_dl_handle;
    const OrtApi* m_api;
    OrtEnv* m_env;

    /* Paraformer model */
    OrtSession* m_session;
    bool m_model_loaded;
    std::vector<std::string> m_tokens;

    /* Punctuation model */
    OrtSession* m_punc_session;
    bool m_punc_model_loaded;
    std::vector<std::string> m_punc_tokens_str;

    std::vector<int16_t> m_record_buffer;
    std::mutex m_buffer_mutex;

    CMVNStats m_cmvn;

    std::string m_model_path;
    std::string m_cmvn_path;
    std::string m_punc_model_path;

    std::vector<std::string> punctuate(const std::vector<int>& token_ids);

    static std::unique_ptr<VoiceInput> m_instance;
};

}

#endif
