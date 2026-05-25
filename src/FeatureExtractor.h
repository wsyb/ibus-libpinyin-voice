#ifndef __FEATURE_EXTRACTOR_H_
#define __FEATURE_EXTRACTOR_H_

#include <vector>
#include <string>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include "kaldi-native-fbank/csrc/online-feature.h"

namespace PY {

struct CMVNStats {
    std::vector<float> means;
    std::vector<float> vars;
    bool valid = false;
};

static bool loadCMVN(const std::string& path, CMVNStats& stats) {
    stats.valid = false;
    std::ifstream f(path);
    if (!f.is_open()) return false;

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(f, line)) {
        lines.push_back(line);
    }

    for (size_t i = 0; i < lines.size(); i++) {
        std::istringstream iss(lines[i]);
        std::string tag;
        if (!(iss >> tag)) continue;

        if (tag == "<AddShift>" && i + 1 < lines.size()) {
            std::istringstream next(lines[i + 1]);
            std::string lt, bracket; float lr;
            if (next >> lt >> lr >> bracket) {
                float val;
                while (next >> val) {
                    stats.means.push_back(val);
                }
            }
        } else if (tag == "<Rescale>" && i + 1 < lines.size()) {
            std::istringstream next(lines[i + 1]);
            std::string lt, bracket; float lr;
            if (next >> lt >> lr >> bracket) {
                float val;
                while (next >> val) {
                    stats.vars.push_back(val);
                }
            }
        }
    }

    stats.valid = !stats.means.empty() && !stats.vars.empty();
    return stats.valid;
}

static void applyLFR(const float* input, int T, int dim,
                     int lfr_m, int lfr_n,
                     std::vector<float>& output, int& out_frames) {
    int T_lfr = (int)std::ceil((float)T / lfr_n);
    int left_pad = (lfr_m - 1) / 2;
    int out_dim = dim * lfr_m;
    out_frames = T_lfr;

    std::vector<float> padded;
    padded.reserve((T + left_pad) * dim);
    for (int i = 0; i < left_pad; i++)
        padded.insert(padded.end(), input, input + dim);
    padded.insert(padded.end(), input, input + T * dim);

    int pad_T = T + left_pad;
    output.resize(T_lfr * out_dim, 0.0f);

    for (int i = 0; i < T_lfr; i++) {
        int start = i * lfr_n;
        if (lfr_m <= pad_T - start) {
            for (int j = 0; j < lfr_m; j++) {
                memcpy(&output[i * out_dim + j * dim],
                       &padded[(start + j) * dim],
                       dim * sizeof(float));
            }
        } else {
            int remaining = pad_T - start;
            for (int j = 0; j < remaining; j++) {
                memcpy(&output[i * out_dim + j * dim],
                       &padded[(start + j) * dim],
                       dim * sizeof(float));
            }
            for (int j = remaining; j < lfr_m; j++) {
                memcpy(&output[i * out_dim + j * dim],
                       &padded[(pad_T - 1) * dim],
                       dim * sizeof(float));
            }
        }
    }
}

static bool extractFeatures(const std::vector<int16_t>& samples,
                            const CMVNStats& cmvn,
                            std::vector<float>& features, int& num_frames) {
    if (samples.empty()) return false;

    const float sample_rate = 16000.0f;
    const int n_mels = 80;
    const int lfr_m = 7;
    const int lfr_n = 6;

    knf::FbankOptions opts;
    opts.frame_opts.samp_freq = sample_rate;
    opts.frame_opts.dither = 1.0f;
    opts.frame_opts.window_type = "hamming";
    opts.frame_opts.snip_edges = true;
    opts.frame_opts.frame_shift_ms = 10.0f;
    opts.frame_opts.frame_length_ms = 25.0f;
    opts.mel_opts.num_bins = n_mels;
    opts.energy_floor = 0.0f;
    opts.use_log_fbank = true;

    knf::OnlineFbank fbank(opts);

    std::vector<float> waveform(samples.size());
    for (size_t i = 0; i < samples.size(); i++)
        waveform[i] = (float)samples[i];
    fbank.AcceptWaveform(sample_rate, waveform.data(), waveform.size());
    fbank.InputFinished();

    int nframes = fbank.NumFramesReady();
    if (nframes == 0) return false;

    std::vector<float> fbank_feat(nframes * n_mels);
    for (int i = 0; i < nframes; i++) {
        const float* frame = fbank.GetFrame(i);
        if (!frame) return false;
        memcpy(&fbank_feat[i * n_mels], frame, n_mels * sizeof(float));
    }

    std::vector<float> lfr_feat;
    int lfr_frames;
    applyLFR(fbank_feat.data(), nframes, n_mels, lfr_m, lfr_n, lfr_feat, lfr_frames);
    if (lfr_frames == 0) return false;

    if (cmvn.valid) {
        int dim = (int)cmvn.means.size();
        for (int i = 0; i < lfr_frames; i++) {
            for (int j = 0; j < dim; j++) {
                lfr_feat[i * dim + j] += cmvn.means[j];
                lfr_feat[i * dim + j] *= cmvn.vars[j];
            }
        }
    }

    features = std::move(lfr_feat);
    num_frames = lfr_frames;
    return true;
}

}

#endif
