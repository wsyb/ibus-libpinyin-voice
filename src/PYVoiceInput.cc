#include "PYVoiceInput.h"
#include <ibus.h>
#include <pulse/error.h>
#include <onnxruntime_c_api.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <glib/gstdio.h>

using namespace PY;

static std::string g_log_path = "/tmp/vocotype-voice.log";

#include <cmath>

static void playTone(int freq, int duration_ms) {
    pa_sample_spec ss;
    ss.format = PA_SAMPLE_S16LE;
    ss.channels = 1;
    ss.rate = 44100;
    int error;
    pa_simple* s = pa_simple_new(NULL, "ibus-voice-beep",
                                  PA_STREAM_PLAYBACK, NULL, "beep",
                                  &ss, NULL, NULL, &error);
    if (!s) return;
    int n = ss.rate * duration_ms / 1000;
    for (int i = 0; i < n; i++) {
        double t = (double)i / ss.rate;
        double env = 1.0;
        int fade = n / 8;
        if (i < fade) env = (double)i / fade;
        else if (i > n - fade) env = (double)(n - i) / fade;
        int16_t sample = (int16_t)(32767 * 0.5 * env * sin(2.0 * M_PI * freq * t));
        pa_simple_write(s, &sample, sizeof(sample), &error);
    }
    pa_simple_drain(s, &error);
    pa_simple_free(s);
}

static std::thread playToneAsync(int freq, int duration_ms) {
    return std::thread([freq, duration_ms]() {
        playTone(freq, duration_ms);
    });
}

static void playBeep(const char*) {
    playTone(880, 80);
}

static void playBeepDone(const char*) {
    std::thread t = playToneAsync(660, 120);
    t.detach();
}

static void vlog(const char* fmt, ...) {
    FILE* f = fopen(g_log_path.c_str(), "a");
    if (!f) return;
    va_list args;
    va_start(args, fmt);
    vfprintf(f, fmt, args);
    va_end(args);
    fprintf(f, "\n");
    fclose(f);
}

#define CHECK_ORT(expr) do { \
    OrtStatus* _s = (expr); \
    if (_s) { \
        const char* _msg = m_api->GetErrorMessage(_s); \
        vlog("VoiceInput: ORT error at %s:%d: %s", __FILE__, __LINE__, _msg ? _msg : "unknown"); \
        m_api->ReleaseStatus(_s); \
        return ""; \
    } \
} while(0)

static std::string getModelDir() {
    const char* home = g_get_home_dir();
    std::string base = std::string(home) +
        "/.cache/modelscope/hub/models/iic/"
        "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx";
    if (g_file_test(base.c_str(), G_FILE_TEST_IS_DIR))
        return base;
    return "";
}

static std::string findFileInDir(const std::string& dir,
                                  const std::vector<std::string>& candidates) {
    for (const auto& name : candidates) {
        std::string path = dir + "/" + name;
        if (g_file_test(path.c_str(), G_FILE_TEST_EXISTS))
            return path;
    }
    return "";
}

VoiceInput::VoiceInput()
    : m_recording(false),
      m_stop_requested(false),
      m_ctrl_held_after_double(false),
      m_has_other_key_since_last_ctrl(false),
      m_dl_handle(nullptr),
      m_api(nullptr),
      m_env(nullptr),
      m_session(nullptr),
      m_model_loaded(false)
{
    vlog("VoiceInput: initializing");

    std::string model_dir = getModelDir();
    if (model_dir.empty()) {
        vlog("VoiceInput: model directory not found");
        return;
    }

    m_model_path = findFileInDir(model_dir, {"model_quant.onnx", "model.int8.onnx"});
    m_cmvn_path = model_dir + "/am.mvn";

    if (m_model_path.empty()) {
        vlog("VoiceInput: model file not found");
        return;
    }

    if (!loadCMVN(m_cmvn_path, m_cmvn)) {
        vlog("VoiceInput: failed to load CMVN from %s", m_cmvn_path.c_str());
        return;
    }
    vlog("VoiceInput: CMVN loaded, dim=%d", (int)m_cmvn.means.size());

    std::string tokens_path = findFileInDir(model_dir, {"tokens.json", "tokens.txt"});
    if (!tokens_path.empty()) {
        if (tokens_path.find(".json") != std::string::npos) {
            std::ifstream f(tokens_path);
            if (f.is_open()) {
                std::string content((std::istreambuf_iterator<char>(f)),
                                     std::istreambuf_iterator<char>());
                size_t pos = 0;
                while ((pos = content.find('"', pos)) != std::string::npos) {
                    size_t end = content.find('"', pos + 1);
                    if (end == std::string::npos) break;
                    m_tokens.push_back(content.substr(pos + 1, end - pos - 1));
                    pos = end + 1;
                }
            }
        }
    }
    vlog("VoiceInput: tokens loaded, count=%d", (int)m_tokens.size());

    initOnnxRuntime();
}

bool VoiceInput::initOnnxRuntime() {
    m_dl_handle = dlopen("libonnxruntime.so.1.23", RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    if (!m_dl_handle) {
        vlog("VoiceInput: dlopen failed: %s", dlerror());
        return false;
    }
    vlog("VoiceInput: dlopen libonnxruntime OK");

    typedef const OrtApiBase*(ORT_API_CALL* GetApiBaseFn)(void);
    GetApiBaseFn get_api_base = (GetApiBaseFn)dlsym(m_dl_handle, "OrtGetApiBase");
    if (!get_api_base) {
        vlog("VoiceInput: dlsym OrtGetApiBase failed: %s", dlerror());
        dlclose(m_dl_handle);
        m_dl_handle = nullptr;
        return false;
    }

    const OrtApiBase* api_base = get_api_base();
    if (!api_base) {
        vlog("VoiceInput: OrtGetApiBase returned null");
        dlclose(m_dl_handle);
        m_dl_handle = nullptr;
        return false;
    }

    m_api = api_base->GetApi(ORT_API_VERSION);
    if (!m_api) {
        vlog("VoiceInput: GetApi(%d) returned null", ORT_API_VERSION);
        dlclose(m_dl_handle);
        m_dl_handle = nullptr;
        return false;
    }
    vlog("VoiceInput: OrtApi obtained, version=%d", ORT_API_VERSION);

    OrtStatus* st = nullptr;
    st = m_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ibus-voice", &m_env);
    if (st) {
        vlog("VoiceInput: CreateEnv failed: %s", m_api->GetErrorMessage(st));
        m_api->ReleaseStatus(st);
        m_api = nullptr;
        dlclose(m_dl_handle);
        m_dl_handle = nullptr;
        return false;
    }
    vlog("VoiceInput: Env created");

    OrtSessionOptions* opts = nullptr;
    st = m_api->CreateSessionOptions(&opts);
    if (st) {
        vlog("VoiceInput: CreateSessionOptions failed: %s", m_api->GetErrorMessage(st));
        m_api->ReleaseStatus(st);
        m_api->ReleaseEnv(m_env);
        m_env = nullptr;
        m_api = nullptr;
        dlclose(m_dl_handle);
        m_dl_handle = nullptr;
        return false;
    }

    m_api->SetIntraOpNumThreads(opts, 0);
    m_api->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_EXTENDED);

    st = m_api->CreateSession(m_env, m_model_path.c_str(), opts, &m_session);
    m_api->ReleaseSessionOptions(opts);
    if (st) {
        vlog("VoiceInput: CreateSession failed: %s", m_api->GetErrorMessage(st));
        m_api->ReleaseStatus(st);
        m_api->ReleaseEnv(m_env);
        m_env = nullptr;
        m_api = nullptr;
        dlclose(m_dl_handle);
        m_dl_handle = nullptr;
        return false;
    }

    m_model_loaded = true;
    vlog("VoiceInput: ONNX model loaded from %s", m_model_path.c_str());
    return true;
}

void VoiceInput::shutdownOnnxRuntime() {
    if (m_session) {
        m_api->ReleaseSession(m_session);
        m_session = nullptr;
    }
    if (m_env) {
        m_api->ReleaseEnv(m_env);
        m_env = nullptr;
    }
    if (m_dl_handle) {
        dlclose(m_dl_handle);
        m_dl_handle = nullptr;
    }
    m_api = nullptr;
}

VoiceInput::~VoiceInput() {
    if (m_recording.load()) {
        m_stop_requested.store(true);
        if (m_record_thread.joinable())
            m_record_thread.join();
    }
    shutdownOnnxRuntime();
}

std::string VoiceInput::getLastResult() {
    std::lock_guard<std::mutex> lock(m_result_mutex);
    return m_last_result;
}

gboolean VoiceInput::handleKeyEvent(guint keyval, guint keycode, guint modifiers) {
    bool pressed = !(modifiers & IBUS_RELEASE_MASK);

    /* Any non-Ctrl key pressed between two Ctrl presses means this is a
     * keyboard shortcut (e.g. Ctrl+C, Ctrl+V), not a pure double-Ctrl.
     * Track it to prevent accidental recording trigger. */
    if (keyval != IBUS_KEY_Control_L && keyval != IBUS_KEY_Control_R) {
        if (pressed)
            m_has_other_key_since_last_ctrl = true;
        return FALSE;
    }

    if (pressed) {
        auto now = std::chrono::steady_clock::now();
        auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - m_last_ctrl_press).count();

        if (dt < 400 && dt >= 0) {
            if (m_has_other_key_since_last_ctrl) {
                /* Another key was pressed between the two Ctrl presses;
                 * this is a shortcut combination, not a pure double-Ctrl.
                 * Reset and ignore. */
                m_last_ctrl_press = now;
                m_has_other_key_since_last_ctrl = false;
                return FALSE;
            }
            m_ctrl_held_after_double = true;
            startRecording();
            m_last_ctrl_press = std::chrono::steady_clock::time_point();
            return TRUE;
        }
        m_last_ctrl_press = now;
        m_has_other_key_since_last_ctrl = false;
        return FALSE;
    } else {
        if (m_ctrl_held_after_double) {
            m_ctrl_held_after_double = false;
            playBeepDone("/usr/share/sounds/freedesktop/stereo/complete.oga");
            stopRecording();
            return TRUE;
        }
        return FALSE;
    }
}

void VoiceInput::startRecording() {
    if (m_recording.load() || !m_model_loaded) return;

    m_recording.store(true);
    m_stop_requested.store(false);
    m_record_buffer.clear();

    {
        std::lock_guard<std::mutex> lock(m_result_mutex);
        m_last_result.clear();
    }

    vlog("VoiceInput: recording started");
    playBeep("/usr/share/sounds/freedesktop/stereo/bell.oga");
    m_record_thread = std::thread(&VoiceInput::recordThread, this);
}

void VoiceInput::stopRecording() {
    if (!m_recording.load()) return;
    auto t_start = std::chrono::steady_clock::now();
    m_stop_requested.store(true);
    if (m_record_thread.joinable())
        m_record_thread.join();
    auto t_after_join = std::chrono::steady_clock::now();
    auto join_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_join - t_start).count();
    vlog("VoiceInput: recording stopped, %d samples, join=%lldms", (int)m_record_buffer.size(), (long long)join_ms);

    m_recording.store(false);

    if (!m_record_buffer.empty()) {
        auto samples = m_record_buffer;
        m_record_buffer.clear();
        auto t_before_transcribe = std::chrono::steady_clock::now();
        std::string result = transcribe(samples);
        auto t_done = std::chrono::steady_clock::now();
        auto transcribe_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_done - t_before_transcribe).count();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_done - t_start).count();
        vlog("VoiceInput: transcribe=%lldms, total=%lldms", (long long)transcribe_ms, (long long)total_ms);
        vlog("VoiceInput: result='%s'", result.c_str());
        if (!result.empty() && result.back() != '。' && result.back() != '，') {
            result += "，";
        }
        {
            std::lock_guard<std::mutex> lock(m_result_mutex);
            m_last_result = result;
        }
    }
}

void VoiceInput::streamReadCb(pa_stream* s, size_t nbytes, void* userdata) {
    VoiceInput* self = (VoiceInput*)userdata;
    const void* data;
    if (pa_stream_peek(s, &data, &nbytes) < 0 || !data) return;
    int nsamples = nbytes / sizeof(int16_t);
    std::lock_guard<std::mutex> lock(self->m_buffer_mutex);
    self->m_record_buffer.insert(self->m_record_buffer.end(),
        (const int16_t*)data, (const int16_t*)data + nsamples);
    pa_stream_drop(s);
}

void VoiceInput::recordThread() {
    pa_mainloop* ml = pa_mainloop_new();
    if (!ml) {
        vlog("VoiceInput: pa_mainloop_new failed");
        m_recording.store(false);
        return;
    }
    pa_mainloop_api* mlapi = pa_mainloop_get_api(ml);

    pa_context* ctx = pa_context_new(mlapi, "ibus-voice-rec");
    pa_context_connect(ctx, NULL, PA_CONTEXT_NOFLAGS, NULL);
    while (pa_context_get_state(ctx) != PA_CONTEXT_READY) {
        if (pa_context_get_state(ctx) == PA_CONTEXT_FAILED) {
            vlog("VoiceInput: pa context failed");
            pa_context_unref(ctx);
            pa_mainloop_free(ml);
            m_recording.store(false);
            return;
        }
        pa_mainloop_iterate(ml, 1, NULL);
    }

    pa_sample_spec ss;
    ss.format = PA_SAMPLE_S16LE;
    ss.channels = 1;
    ss.rate = SAMPLE_RATE;

    pa_stream* stream = pa_stream_new(ctx, "voice record", &ss, NULL);
    if (!stream) {
        vlog("VoiceInput: pa_stream_new failed");
        pa_context_disconnect(ctx);
        pa_context_unref(ctx);
        pa_mainloop_free(ml);
        m_recording.store(false);
        return;
    }

    pa_stream_set_read_callback(stream, streamReadCb, this);

    pa_buffer_attr bufattr;
    bufattr.fragsize = 320;
    bufattr.maxlength = (uint32_t)-1;
    bufattr.minreq = (uint32_t)-1;
    bufattr.prebuf = (uint32_t)-1;
    bufattr.tlength = (uint32_t)-1;
    pa_stream_connect_record(stream, NULL, &bufattr, PA_STREAM_NOFLAGS);

    while (!m_stop_requested.load()) {
        pa_mainloop_iterate(ml, 0, NULL);
    }

    pa_stream_set_read_callback(stream, NULL, NULL);
    const void* data;
    size_t nbytes;
    while (pa_stream_readable_size(stream) > 0 && pa_stream_peek(stream, &data, &nbytes) == 0 && data) {
        int nsamples = nbytes / sizeof(int16_t);
        std::lock_guard<std::mutex> lock(m_buffer_mutex);
        m_record_buffer.insert(m_record_buffer.end(),
            (const int16_t*)data, (const int16_t*)data + nsamples);
        pa_stream_drop(stream);
    }

    pa_stream_disconnect(stream);
    pa_stream_unref(stream);
    pa_context_disconnect(ctx);
    pa_context_unref(ctx);
    pa_mainloop_free(ml);
}

std::string VoiceInput::transcribe(const std::vector<int16_t>& samples) {
    if (!m_model_loaded || !m_api || samples.empty())
        return "";

    std::vector<float> features;
    int num_frames;
    if (!extractFeatures(samples, m_cmvn, features, num_frames) || num_frames == 0) {
        vlog("VoiceInput: feature extraction failed");
        return "";
    }

    vlog("VoiceInput: features extracted, frames=%d, dim=%d",
         num_frames, (int)(features.size() / num_frames));

    auto t0 = std::chrono::steady_clock::now();

    int feat_dim = (int)m_cmvn.means.size();
    int64_t input_shape[3] = {1, num_frames, feat_dim};
    int64_t length_shape[1] = {1};
    int32_t speech_length = num_frames;

    OrtMemoryInfo* mem_info = nullptr;
    CHECK_ORT(m_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));

    OrtValue* speech_tensor = nullptr;
    CHECK_ORT(m_api->CreateTensorWithDataAsOrtValue(
        mem_info, features.data(), features.size() * sizeof(float),
        input_shape, 3,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &speech_tensor));

    OrtValue* length_tensor = nullptr;
    CHECK_ORT(m_api->CreateTensorWithDataAsOrtValue(
        mem_info, &speech_length, sizeof(int32_t),
        length_shape, 1,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &length_tensor));

    m_api->ReleaseMemoryInfo(mem_info);

    const char* input_names[] = {"speech", "speech_lengths"};
    const char* output_names[] = {"logits", "token_num"};
    OrtValue* inputs[] = {speech_tensor, length_tensor};
    OrtValue* outputs[2] = {nullptr, nullptr};

    vlog("VoiceInput: calling Session::Run...");
    OrtStatus* st = m_api->Run(m_session, nullptr,
        input_names, inputs, 2,
        output_names, 2, outputs);
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    vlog("VoiceInput: Session::Run took %lldms", (long long)ms);
    m_api->ReleaseValue(speech_tensor);
    m_api->ReleaseValue(length_tensor);

    if (st) {
        vlog("VoiceInput: Run failed: %s", m_api->GetErrorMessage(st));
        m_api->ReleaseStatus(st);
        return "";
    }

    vlog("VoiceInput: Session::Run returned OK");

    OrtTensorTypeAndShapeInfo* type_info = nullptr;
    CHECK_ORT(m_api->GetTensorTypeAndShape(outputs[0], &type_info));
    int64_t dims[4];
    size_t dims_count = 4;
    CHECK_ORT(m_api->GetDimensions(type_info, dims, dims_count));
    m_api->ReleaseTensorTypeAndShapeInfo(type_info);

    int seq_len = (int)dims[1];
    int vocab = (int)dims[2];

    float* logit_data = nullptr;
    CHECK_ORT(m_api->GetTensorMutableData(outputs[0], (void**)&logit_data));

    std::vector<int> token_ids;
    for (int t = 0; t < seq_len; t++) {
        int best_id = 0;
        float best_score = logit_data[t * vocab];
        for (int v = 1; v < vocab; v++) {
            if (logit_data[t * vocab + v] > best_score) {
                best_score = logit_data[t * vocab + v];
                best_id = v;
            }
        }
        if (best_id == 0 || best_id == 2) continue;
        if (!token_ids.empty() && best_id == token_ids.back()) continue;
        token_ids.push_back(best_id);
    }

    std::string result;
    for (int id : token_ids) {
        if (id > 0 && id < (int)m_tokens.size()) {
            std::string tok = m_tokens[id];
            if (tok == "<s>" || tok == "</s>" || tok == "<blank>" || tok == "<unk>")
                continue;
            if (tok.size() >= 2 && tok[tok.size() - 2] == '@' && tok[tok.size() - 1] == '@') {
                tok = tok.substr(0, tok.size() - 2);
            }
            result += tok;
        }
    }

    vlog("VoiceInput: decoded %d tokens, result='%s'", (int)token_ids.size(), result.c_str());

    if (outputs[0]) m_api->ReleaseValue(outputs[0]);
    if (outputs[1]) m_api->ReleaseValue(outputs[1]);

    return result;
}
