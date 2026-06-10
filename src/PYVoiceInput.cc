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

std::unique_ptr<VoiceInput> VoiceInput::m_instance;

void VoiceInput::init(void) {
    if (!m_instance) {
        m_instance.reset(new VoiceInput());
    }
}

void VoiceInput::finalize(void) {
    if (m_instance) {
        m_instance.reset();
    }
}

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
    playTone(880, 50);
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

/* Parse a JSON array of strings (tokens.json) into a vector.
 * Handles simple format: ["token1","token2",...] without escape handling. */
static std::vector<std::string> parseJsonStringArray(const std::string& path) {
    std::vector<std::string> result;
    std::ifstream f(path);
    if (!f.is_open()) return result;
    std::string content((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
    size_t pos = 0;
    while ((pos = content.find('"', pos)) != std::string::npos) {
        size_t end = content.find('"', pos + 1);
        if (end == std::string::npos) break;
        result.push_back(content.substr(pos + 1, end - pos - 1));
        pos = end + 1;
    }
    return result;
}

static std::string getPuncDir() {
    const char* home = g_get_home_dir();
    std::string base = std::string(home) +
        "/.cache/modelscope/hub/models/iic/"
        "punc_ct-transformer_zh-cn-common-vocab272727-onnx";
    if (g_file_test(base.c_str(), G_FILE_TEST_IS_DIR))
        return base;
    return "";
}

/* Decode UTF-8 string into a vector of Unicode codepoints */
static std::vector<uint32_t> decodeUtf8(const std::string& s) {
    std::vector<uint32_t> codepoints;
    size_t i = 0;
    while (i < s.size()) {
        uint32_t cp = 0;
        unsigned char c = (unsigned char)s[i];
        int bytes = 0;
        if (c < 0x80) { cp = c; bytes = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; bytes = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; bytes = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; bytes = 4; }
        else { i++; continue; }
        for (int j = 1; j < bytes && i + j < s.size(); j++) {
            cp = (cp << 6) | ((unsigned char)s[i + j] & 0x3F);
        }
        codepoints.push_back(cp);
        i += bytes;
    }
    return codepoints;
}

/* Convert a Unicode codepoint to its UTF-8 string */
static std::string codepointToUtf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xF0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
    return s;
}

VoiceInput::VoiceInput()
    : m_recording(false),
      m_stop_requested(false),
      m_dl_handle(nullptr),
      m_api(nullptr),
      m_env(nullptr),
      m_session(nullptr),
      m_model_loaded(false),
      m_punc_session(nullptr),
      m_punc_model_loaded(false)
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
    if (!tokens_path.empty())
        m_tokens = parseJsonStringArray(tokens_path);
    vlog("VoiceInput: tokens loaded, count=%d", (int)m_tokens.size());

    /* Load punctuation model */
    std::string punc_dir = getPuncDir();
    if (!punc_dir.empty()) {
        m_punc_model_path = findFileInDir(punc_dir, {"model_quant.onnx", "model.onnx"});
        if (!m_punc_model_path.empty()) {
            std::string punc_tokens_path = findFileInDir(punc_dir, {"tokens.json", "tokens.txt"});
            if (!punc_tokens_path.empty())
                m_punc_tokens_str = parseJsonStringArray(punc_tokens_path);
            vlog("VoiceInput: punctuation tokens loaded, count=%d", (int)m_punc_tokens_str.size());
        }
    }

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

    /* Create punctuation session */
    if (!m_punc_model_path.empty()) {
        OrtSessionOptions* punc_opts = nullptr;
        st = m_api->CreateSessionOptions(&punc_opts);
        if (!st) {
            m_api->SetInterOpNumThreads(punc_opts, 4);
            m_api->SetIntraOpNumThreads(punc_opts, 4);
            st = m_api->CreateSession(m_env, m_punc_model_path.c_str(), punc_opts, &m_punc_session);
            if (st) {
                vlog("VoiceInput: punctuation session failed: %s", m_api->GetErrorMessage(st));
                m_api->ReleaseStatus(st);
                m_punc_session = nullptr;
            } else {
                m_punc_model_loaded = true;
                vlog("VoiceInput: punctuation model loaded from %s", m_punc_model_path.c_str());
            }
            m_api->ReleaseSessionOptions(punc_opts);
        }
    }

    return true;
}

void VoiceInput::shutdownOnnxRuntime() {
    if (m_session) {
        m_api->ReleaseSession(m_session);
        m_session = nullptr;
    }
    if (m_punc_session) {
        m_api->ReleaseSession(m_punc_session);
        m_punc_session = nullptr;
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

    /* Only respond to Right Ctrl key */
    if (keyval != IBUS_KEY_Control_R)
        return FALSE;

    if (pressed && !m_recording.load()) {
        playBeep("/usr/share/sounds/freedesktop/stereo/complete.oga");
        startRecording();
        return TRUE;
    }

    if (!pressed && m_recording.load()) {
        stopRecording();
        return TRUE;
    }

    return FALSE;
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
    m_record_thread = std::thread(&VoiceInput::recordThread, this);
}

void VoiceInput::stopRecording() {
    if (!m_recording.load()) return;
    auto t_start = std::chrono::steady_clock::now();
    m_stop_requested.store(true);
    if (m_record_thread.joinable())
        m_record_thread.join();
    m_stop_requested.store(false);
    auto t_after_join = std::chrono::steady_clock::now();
    auto join_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_after_join - t_start).count();
    vlog("VoiceInput: recording stopped, %d samples, join=%lldms", (int)m_record_buffer.size(), (long long)join_ms);

    m_recording.store(false);

    /* Skip transcription if audio is too short (less than 1 second).
     * Short recordings are mostly silence/noise and cause hallucination
     * in the ASR model (e.g. generating "对的对的" from empty audio). */
    constexpr size_t MIN_SAMPLES = 16000;  // 1 second at 16kHz
    if (!m_record_buffer.empty() && m_record_buffer.size() < MIN_SAMPLES) {
        vlog("VoiceInput: audio too short (%d samples, need %zu), skipping",
             (int)m_record_buffer.size(), MIN_SAMPLES);
        m_record_buffer.clear();
        return;
    }

    if (!m_record_buffer.empty()) {
        auto samples = m_record_buffer;
        m_record_buffer.clear();
        auto t_before_transcribe = std::chrono::steady_clock::now();

        std::vector<bool> space_before;
        std::string result = transcribe(samples, space_before);

        auto t_done = std::chrono::steady_clock::now();
        auto transcribe_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_done - t_before_transcribe).count();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_done - t_start).count();
        vlog("VoiceInput: transcribe=%lldms, total=%lldms", (long long)transcribe_ms, (long long)total_ms);
        vlog("VoiceInput: result='%s'", result.c_str());

        /* Apply punctuation model if available and text has Chinese.
         * Only send Chinese characters to the model — English letters
         * confuse it and cause spurious punctuation inside words. */
        if (m_punc_model_loaded && !result.empty()) {
            std::vector<uint32_t> codepoints = decodeUtf8(result);

            /* Extract Chinese characters and their original positions */
            std::vector<int> cn_ids;
            std::vector<size_t> cn_positions;  /* index into codepoints[] */
            for (size_t i = 0; i < codepoints.size(); i++) {
                uint32_t cp = codepoints[i];
                if ((cp >= 0x4E00 && cp <= 0x9FFF) ||
                    (cp >= 0x3400 && cp <= 0x4DBF) ||
                    (cp >= 0xF900 && cp <= 0xFAFF)) {
                    std::string s = codepointToUtf8(cp);
                    auto it = std::find(m_punc_tokens_str.begin(), m_punc_tokens_str.end(), s);
                    cn_ids.push_back(it != m_punc_tokens_str.end()
                                     ? (int)(it - m_punc_tokens_str.begin()) : 0);
                    cn_positions.push_back(i);
                }
            }

            if (!cn_ids.empty()) {
                std::vector<std::string> punc_result = punctuate(cn_ids);
                if (!punc_result.empty()) {
                    /* Build a map: codepoint index → punctuation string */
                    std::vector<std::string> punc_at(codepoints.size());
                    for (size_t j = 0; j < punc_result.size() && j < cn_positions.size(); j++) {
                        if (!punc_result[j].empty())
                            punc_at[cn_positions[j]] = punc_result[j];
                    }

                    std::string punctuated;
                    for (size_t i = 0; i < codepoints.size(); i++) {
                        if (i < space_before.size() && space_before[i])
                            punctuated += ' ';
                        punctuated += codepointToUtf8(codepoints[i]);
                        punctuated += punc_at[i];
                    }
                    result = punctuated;
                    space_before.clear();
                    vlog("VoiceInput: after punctuate: '%s'", result.c_str());
                }
            }
        }

        /* Insert word boundary spaces if punctuation model wasn't applied */
        if (!space_before.empty()) {
            std::vector<uint32_t> codepoints = decodeUtf8(result);
            std::string with_spaces;
            for (size_t i = 0; i < codepoints.size(); i++) {
                if (i < space_before.size() && space_before[i])
                    with_spaces += ' ';
                with_spaces += codepointToUtf8(codepoints[i]);
            }
            result = with_spaces;
            space_before.clear();
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


#define CHECK_ORT_PUNC(expr) do { \
    OrtStatus* _s = (expr); \
    if (_s) { \
        m_api->ReleaseStatus(_s); \
        return {}; \
    } \
} while(0)

std::vector<std::string> VoiceInput::punctuate(const std::vector<int>& token_ids) {
    if (!m_punc_model_loaded || !m_api || token_ids.empty())
        return {};

    int seq_len = (int)token_ids.size();

    int64_t input_shape[2] = {1, seq_len};
    int64_t length_shape[1] = {1};
    int32_t text_length = seq_len;

    OrtMemoryInfo* mem_info = nullptr;
    CHECK_ORT_PUNC(m_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));

    OrtValue* input_tensor = nullptr;
    CHECK_ORT_PUNC(m_api->CreateTensorWithDataAsOrtValue(
        mem_info, (void*)token_ids.data(), token_ids.size() * sizeof(int32_t),
        input_shape, 2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &input_tensor));

    OrtValue* length_tensor = nullptr;
    CHECK_ORT_PUNC(m_api->CreateTensorWithDataAsOrtValue(
        mem_info, &text_length, sizeof(int32_t),
        length_shape, 1,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &length_tensor));

    m_api->ReleaseMemoryInfo(mem_info);

    const char* input_names[] = {"inputs", "text_lengths"};
    const char* output_names[] = {"logits"};
    OrtValue* inputs[] = {input_tensor, length_tensor};
    OrtValue* outputs[1] = {nullptr};

    OrtStatus* st = m_api->Run(m_punc_session, nullptr,
        input_names, inputs, 2,
        output_names, 1, outputs);
    m_api->ReleaseValue(input_tensor);
    m_api->ReleaseValue(length_tensor);

    if (st) {
        vlog("VoiceInput: punctuation Run failed: %s", m_api->GetErrorMessage(st));
        m_api->ReleaseStatus(st);
        return {};
    }

    OrtTensorTypeAndShapeInfo* type_info = nullptr;
    CHECK_ORT_PUNC(m_api->GetTensorTypeAndShape(outputs[0], &type_info));
    int64_t dims[3];
    size_t dims_count = 3;
    CHECK_ORT_PUNC(m_api->GetDimensions(type_info, dims, dims_count));
    m_api->ReleaseTensorTypeAndShapeInfo(type_info);

    int out_len = (int)dims[1];
    int punc_vocab = (int)dims[2];

    float* logit_data = nullptr;
    CHECK_ORT_PUNC(m_api->GetTensorMutableData(outputs[0], (void**)&logit_data));

    const char* punc_marks[] = {"", "", "，", "。", "？", "、"};

    std::vector<std::string> result_per_pos;
    for (int t = 0; t < out_len && t < seq_len; t++) {
        int best_punc = 0;
        float best_score = logit_data[t * punc_vocab];
        for (int v = 1; v < punc_vocab; v++) {
            if (logit_data[t * punc_vocab + v] > best_score) {
                best_score = logit_data[t * punc_vocab + v];
                best_punc = v;
            }
        }
        if (best_punc >= 2 && best_punc <= 5) {
            result_per_pos.push_back(punc_marks[best_punc]);
        } else {
            result_per_pos.push_back("");
        }
    }

    if (outputs[0]) m_api->ReleaseValue(outputs[0]);

    return result_per_pos;
}


std::string VoiceInput::transcribe(const std::vector<int16_t>& samples,
                                    std::vector<bool>& space_before) {
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
    space_before.clear();  /* output parameter: per-character space-before flag */
    bool prev_continues = false;
    bool prev_is_cjk = false;
    for (int id : token_ids) {
        if (id > 0 && id < (int)m_tokens.size()) {
            std::string tok = m_tokens[id];
            if (tok == "<s>" || tok == "</s>" || tok == "<blank>" || tok == "<unk>")
                continue;

            bool is_continuation = (tok.size() >= 2 &&
                tok[tok.size() - 2] == '@' && tok[tok.size() - 1] == '@');
            if (is_continuation)
                tok = tok.substr(0, tok.size() - 2);

            bool is_cjk = (tok.size() >= 3 &&
                (unsigned char)tok[0] >= 0xE4 && (unsigned char)tok[0] <= 0xE9);
            /* Single ASCII punctuation/symbol character (e.g. . , & @ ' -) */
            bool is_punct = (tok.size() == 1 && !is_cjk &&
                !std::isalnum((unsigned char)tok[0]));

            /* Mark word boundary: need space before this token if:
             * - not a continuation of previous token
             * - neither side is CJK
             * - current token is not punctuation */
            bool need_space = !result.empty() && !prev_continues &&
                              !prev_is_cjk && !is_cjk && !is_punct;

            /* Encode token as UTF-8 codepoints, mark first char with space */
            std::vector<uint32_t> cps = decodeUtf8(tok);
            for (size_t i = 0; i < cps.size(); i++) {
                space_before.push_back(need_space && i == 0);
            }
            result += tok;

            prev_continues = is_continuation;
            prev_is_cjk = is_cjk;
        }
    }

    /* space_before is returned via output parameter for post-punctuation insertion */

    vlog("VoiceInput: decoded %d tokens, result='%s'", (int)token_ids.size(), result.c_str());

    if (outputs[0]) m_api->ReleaseValue(outputs[0]);
    if (outputs[1]) m_api->ReleaseValue(outputs[1]);

    return result;
}
