#include "llama.h"
#ifdef _WIN32
#include <windows.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <sstream>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

struct ActivationEdit {
    int neuron;
    float delta = 0;
    int prefill = 0;
    int decode = 0;
};

struct Calibration {
    decltype(&ggml_backend_tensor_get) read_tensor;
    decltype(&ggml_backend_tensor_set) write_tensor;
    std::map<int, float> medians;
    std::map<int, ActivationEdit> edits;
    bool active = true;
    bool edit_failed = false;
};

bool capture(ggml_tensor *tensor, bool ask, void *user) {
    auto &state = *static_cast<Calibration *>(user);
    if (!state.active && std::strncmp(tensor->name, "ffn_swiglu-", 11) == 0) {
        auto found = state.edits.find(std::atoi(tensor->name + 11));
        if (found == state.edits.end()) return false;
        if (ask) return true;
        auto &edit = found->second;
        const auto width = tensor->ne[0];
        const auto positions = tensor->ne[1];
        if (tensor->type != GGML_TYPE_F32 || tensor->nb[1] != width * sizeof(float) ||
            tensor->ne[2] != 1 || tensor->ne[3] != 1 || edit.neuron < 0 || edit.neuron >= width) {
            state.edit_failed = true;
            return false;
        }
        if (positions > 1) ++edit.prefill;
        else ++edit.decode;
        if (edit.delta != 0) {
            std::vector<float> values(width * positions);
            const auto bytes = values.size() * sizeof(float);
            state.read_tensor(tensor, values.data(), 0, bytes);
            for (int64_t pos = 0; pos < positions; ++pos) values[pos * width + edit.neuron] += edit.delta;
            state.write_tensor(tensor, values.data(), 0, bytes);
        }
        return true;
    }
    if (!state.active || std::strncmp(tensor->name, "l_out-", 6) != 0) return false;
    auto found = state.medians.find(std::atoi(tensor->name + 6));
    if (found == state.medians.end()) return false;
    if (ask) return true;
    const auto width = tensor->ne[0];
    const auto count = tensor->ne[1];
    if (tensor->type != GGML_TYPE_F32 || count < 2 || tensor->nb[1] != width * sizeof(float)) return false;
    std::vector<float> values(width * count);
    state.read_tensor(tensor, values.data(), 0, values.size() * sizeof(float));
    std::vector<float> norms;
    for (int64_t pos = 1; pos < count; ++pos) {
        double sum = 0;
        for (int64_t i = 0; i < width; ++i) {
            double value = values[pos * width + i];
            sum += value * value;
        }
        norms.push_back(static_cast<float>(std::sqrt(sum)));
    }
    auto midpoint = norms.begin() + (norms.size() - 1) / 2;
    std::nth_element(norms.begin(), midpoint, norms.end());
    found->second = *midpoint;
    return true;
}

struct Target {
    int layer;
    int neuron;
    float sign;
    float column_norm;
    std::vector<float> direction;
};

std::string json_string(const std::string &value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char ch : value) {
        if (ch == '"' || ch == '\\') out << '\\' << ch;
        else if (ch < 32) out << "\\u00" << std::hex << std::setw(2) << std::setfill('0') << int(ch);
        else out << ch;
    }
    out << '"';
    return out.str();
}
}

int main(int argc, char **argv) try {
    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << "neuron-study 2 / llama.cpp 6c01da6278ab6c69ed67d9124806b50d2b299ba7\n";
        return 0;
    }
    const char *replay_file = argc == 7 ? argv[--argc] : nullptr;
    if (argc != 5 && argc != 6) throw std::runtime_error("Usage: neuron-study-engine MODEL DIRECTIONS PROMPTS METHOD [DOSES [HISTORY]]");
    const std::string method = argv[4];
    const bool chat = method == "chat" && argc == 6;
    if (!chat && method != "activation" && method != "residual") throw std::runtime_error("Invalid method");
    const bool direct = chat || method == "activation";
    ggml_backend_load_all();
    llama_backend_init();
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 999;
    auto *model = llama_model_load_from_file(argv[1], model_params);
    if (!model) throw std::runtime_error("Model load failed");
    const auto width = llama_model_n_embd(model);
    const auto n_layers = llama_model_n_layer(model);
    auto *vocab = llama_model_get_vocab(model);
    const auto vocab_size = llama_vocab_n_tokens(vocab);

    Calibration calibration{ggml_backend_tensor_get, ggml_backend_tensor_set, {}, {}};
    std::vector<Target> targets;
    std::ifstream direction_file(argv[2], std::ios::binary);
    int layer, neuron;
    float sign, column_norm;
    while (direction_file.read(reinterpret_cast<char *>(&layer), sizeof(layer))) {
        if (layer < 1 || layer >= n_layers || calibration.medians.contains(layer))
            throw std::runtime_error("Invalid target layer");
        if (!direction_file.read(reinterpret_cast<char *>(&neuron), sizeof(neuron)) ||
            !direction_file.read(reinterpret_cast<char *>(&sign), sizeof(sign)) ||
            !direction_file.read(reinterpret_cast<char *>(&column_norm), sizeof(column_norm)))
            throw std::runtime_error("Missing neuron data");
        std::vector<float> direction(width);
        if (!direction_file.read(reinterpret_cast<char *>(direction.data()), width * sizeof(float)))
            throw std::runtime_error("Incomplete direction");
        targets.push_back({layer, neuron, sign, column_norm, std::move(direction)});
        if (direct) calibration.edits[layer] = {neuron};
        calibration.medians[layer] = 0.0f;
    }
    if (targets.empty()) throw std::runtime_error("No control vectors provided");

    const char *limit = std::getenv("QWEN_RUN_TOKEN_CAP");
    const int max_tokens = limit ? std::stoi(limit) : 96;
    if (max_tokens < 1 || max_tokens > 4096) throw std::runtime_error("Token cap outside 1..4096");
    const bool trajectory = !chat && std::getenv("QWEN_RUN_TRAJECTORY") != nullptr;
    std::map<std::string, std::vector<llama_token>> histories;
    auto context_params = llama_context_default_params();
    context_params.n_ctx = (trajectory || chat) ? 32768 : std::max(512, max_tokens + 512);
    context_params.n_batch = 256;
    context_params.n_ubatch = 256;
    context_params.cb_eval = capture;
    context_params.cb_eval_user_data = &calibration;
    auto *ctx = llama_init_from_model(model, context_params);
    if (!ctx) throw std::runtime_error("Context creation failed");

    auto tokenize = [&](const std::string &text, bool special) {
        std::vector<llama_token> ids(text.size() + 4);
        auto size = llama_tokenize(vocab, text.c_str(), int(text.size()), ids.data(), int(ids.size()), false, special);
        if (size <= 1) throw std::runtime_error("Tokenizer did not return a valid sequence");
        ids.resize(size);
        return ids;
    };
    // A fork starts from exact assistant IDs, with each user turn tokenized separately.
    std::vector<llama_token> prefix;
    if (replay_file) {
        if (!trajectory) throw std::runtime_error("Replay requires trajectory mode");
        std::ifstream saved(replay_file);
        if (!saved) throw std::runtime_error("Cannot open history file");
        std::string user, ids;
        while (std::getline(saved, user)) {
            if (!std::getline(saved, ids)) throw std::runtime_error("Missing history token line");
            auto turn = tokenize("<|im_start|>user\n" + user +
                                 "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", true);
            prefix.insert(prefix.end(), turn.begin(), turn.end());
            std::istringstream tokens(ids);
            llama_token id;
            while (tokens >> id) {
                if (id < 0 || id >= vocab_size) throw std::runtime_error("Invalid history token");
                prefix.push_back(id);
            }
            if (!tokens.eof()) throw std::runtime_error("Malformed history token");
            auto end = tokenize("<|im_end|>\n", true);
            prefix.insert(prefix.end(), end.begin(), end.end());
        }
    }
    auto decode = [&](std::vector<llama_token> &ids) {
        if (llama_decode(ctx, llama_batch_get_one(ids.data(), int(ids.size()))) != 0 || calibration.edit_failed)
            throw std::runtime_error("llama_decode failed");
        float *logits = llama_get_logits_ith(ctx, -1);
        if (!logits) throw std::runtime_error("Missing logits");
        return logits;
    };

    std::ifstream prompt_file(argv[3]);
    std::string reference;
    std::getline(prompt_file, reference);
    auto calibration_ids = tokenize(reference, false);
    decode(calibration_ids);
    calibration.active = false;
    for (auto [index, median] : calibration.medians) {
        if (!std::isfinite(median) || median <= 0) throw std::runtime_error("No calibration residual was captured");
        std::cout << "CALIBRATION " << index << " " << median << " " << calibration_ids.size() << std::endl;
    }

    const auto penalty_env = std::getenv("QWEN_RUN_REPEAT_PENALTY");
    const float penalty = penalty_env ? std::stof(penalty_env) : 1.0f;
    // Start fresh each turn, since input already contains this branch's complete history.
    auto greedy_sampler = [&](const std::vector<llama_token> &input) {
        std::vector<llama_token> seen;
        std::vector<bool> marked(vocab_size);
        if (penalty != 1.0f) {
            for (auto id : input) {
                if (!marked[id]) { marked[id] = true; seen.push_back(id); }
            }
        }
        return [&, seen = std::move(seen), marked = std::move(marked)](float *logits) mutable {
            for (auto id : seen)
                logits[id] = logits[id] < 0 ? logits[id] * penalty : logits[id] / penalty;
            auto token = llama_token(std::max_element(logits, logits + vocab_size) - logits);
            if (penalty != 1.0f && !marked[token]) {
                marked[token] = true;
                seen.push_back(token);
            }
            return token;
        };
    };

    if (chat) {
        const float dose = std::stof(argv[5]);
        if (!std::isfinite(dose)) throw std::runtime_error("Invalid dose");
        for (const auto &target : targets)
            calibration.edits.at(target.layer).delta = dose * target.sign * calibration.medians.at(target.layer) / target.column_norm;
        std::string history, message;
#ifdef _WIN32
        SetConsoleCP(CP_UTF8);
        SetConsoleOutputCP(CP_UTF8);
#endif
        std::cout << "Steered chat ready. /clear resets history. /bye exits.\n";
        while (std::cout << "\nYou> " << std::flush && std::getline(std::cin, message)) {
            if (message == "/bye" || message == "/exit") break;
            if (message == "/clear") { history.clear(); continue; }
            if (message.empty()) continue;
            auto turn = history + "<|im_start|>user\n" + message +
                        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
            auto ids = tokenize(turn, true);
            const unsigned reply_limit = limit ? max_tokens : 256;
            if (ids.size() + reply_limit > context_params.n_ctx) {
                std::cout << "Conversation exceeds the context window. Use /clear or shorten the message.\n";
                continue;
            }
            llama_memory_clear(llama_get_memory(ctx), true);
            float *logits = nullptr;
            for (size_t start = 0; start < ids.size(); start += context_params.n_batch) {
                auto end = std::min(ids.size(), start + context_params.n_batch);
                std::vector<llama_token> chunk(ids.begin() + start, ids.begin() + end);
                logits = decode(chunk);
            }
            std::cout << "Qwen> " << std::flush;
            std::string answer;
            auto sample = greedy_sampler(ids);
            for (unsigned i = 0; i < reply_limit; ++i) {
                auto token = sample(logits);
                if (llama_vocab_is_eog(vocab, token)) break;
                char piece[1024];
                int n = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, false);
                if (n < 0) throw std::runtime_error("Token exceeds output buffer");
                answer.append(piece, n);
                std::cout.write(piece, n).flush();
                std::vector<llama_token> next{token};
                logits = decode(next);
            }
            history = turn + answer + "<|im_end|>\n";
            std::cout << '\n';
        }
        llama_free(ctx);
        llama_model_free(model);
        return 0;
    }


    using Case = std::pair<std::string, std::vector<float>>;
    std::vector<Case> cases = {{"baseline", {}}, {"zero_dose", std::vector<float>(targets.size(), 0)}};
    if (argc == 5 && targets.size() == 2) {
        cases.push_back({"layer_13_only", {1, 0}});
        cases.push_back({"layer_14_only", {0, 1}});
    }
    std::istringstream dose_list(argc == 6 ? argv[5] : "50,75,100");
    std::string field;
    while (std::getline(dose_list, field, ',')) {
        int percent = std::stoi(field);
        if (percent < 1 || percent > 200) throw std::runtime_error("Invalid dose percent");
        std::ostringstream label;
        label << "dose_" << std::setfill('0') << std::setw(3) << percent;
        cases.push_back({label.str(), std::vector<float>(targets.size(), percent / 100.0f)});
    }
    cases.push_back({"restored", {}});
    int prompt_number = 0;
    std::string prompt;
    while (std::getline(prompt_file, prompt)) {
        if (prompt.empty()) continue;
        ++prompt_number;
        std::vector<float> baseline_logits;
        std::vector<llama_token> baseline_tokens;
        for (const auto &[label, scales] : cases) {
            std::vector<float> control((n_layers - 1) * width, 0);
            int first = n_layers, last = 0;
            for (auto &[index, edit] : calibration.edits) edit = {edit.neuron};
            for (size_t i = 0; i < scales.size(); ++i) {
                const auto &target = targets[i];
                first = std::min(first, target.layer);
                last = std::max(last, target.layer);
                const auto gain = scales[i] * target.sign * calibration.medians.at(target.layer);
                if (direct) calibration.edits.at(target.layer).delta = gain / target.column_norm;
                else {
                    auto dst = control.begin() + (target.layer - 1) * width;
                    std::transform(target.direction.begin(), target.direction.end(), dst,
                                   [gain](float coordinate) { return gain * coordinate; });
                }
            }
            if (!direct && llama_set_adapter_cvec(ctx, scales.empty() ? nullptr : control.data(),
                                      control.size(), width, first, last) != 0)
                throw std::runtime_error("Setting the control vector failed");
            llama_memory_clear(llama_get_memory(ctx), true);
            auto input = tokenize("<|im_start|>user\n" + prompt +
                                  "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", true);
            if (replay_file && prompt_number == 1) histories[label] = prefix;
            if (trajectory) input.insert(input.begin(), histories[label].begin(), histories[label].end());
            if (input.size() + max_tokens > context_params.n_ctx)
                throw std::runtime_error("Trajectory exceeds context capacity; history was not truncated");
            int prefill_calls = 0, singleton_calls = 0;
            float *logits = nullptr;
            for (size_t pos = 0; pos < input.size(); pos += context_params.n_batch) {
                auto stop = std::min(input.size(), pos + context_params.n_batch);
                std::vector<llama_token> chunk(input.begin()+pos, input.begin()+stop);
                chunk.size() == 1 ? ++singleton_calls : ++prefill_calls;
                logits = decode(chunk);
            }
            if (label == "baseline") baseline_logits.assign(logits, logits + vocab_size);
            float delta = 0;
            for (int token = 0; token < vocab_size; ++token) {
                delta = std::max(delta, std::abs(logits[token] - baseline_logits[token]));
            }

            std::string text;
            std::vector<llama_token> output;
            auto sample = greedy_sampler(input);
            bool stopped = false;
            for (int i = 0; i < max_tokens; ++i) {
                auto token = sample(logits);
                output.push_back(token);
                if (llama_vocab_is_eog(vocab, token)) {
                    stopped = true;
                    break;
                }
                char piece[256];
                int length = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, false);
                if (length < 0) throw std::runtime_error("A decoded token exceeds the output buffer");
                text.append(piece, length);
                if (i + 1 < max_tokens) {
                    std::vector<llama_token> next{token};
                    logits = decode(next);
                }
            }
            if (trajectory) {
                auto &history = histories[label];
                history = input;
                history.insert(history.end(), output.begin(), output.end() - int(stopped));
                auto separator = tokenize("<|im_end|>\n", true);
                history.insert(history.end(), separator.begin(), separator.end());
            }
            if (label == "baseline") baseline_tokens = output;
            else if (label == "zero_dose" || label == "restored") {
                if (output != baseline_tokens || delta > 0.0001f)
                    throw std::runtime_error("Baseline and control differ in tokens or logits");
            } else if (delta <= 0.0001f) {
                throw std::runtime_error("Nonzero dose left baseline logits unchanged");
            }
            for (auto [index, edit] : calibration.edits) {
                if (edit.prefill != prefill_calls || edit.decode != singleton_calls + int(output.size()) - 1)
                    throw std::runtime_error("Activation hook did not run at each forward pass");
            }
            std::cout << "RESULT {\"prompt_number\":" << prompt_number << ",\"condition\":" << json_string(label)
                      << ",\"input_tokens\":" << input.size() << ",\"first_step_max_logit_delta\":" << delta
                      << ",\"finish_reason\":" << json_string(stopped ? "stop" : "length")
                      << ",\"token_ids\":[";
            for (size_t i = 0; i < output.size(); ++i) std::cout << (i ? "," : "") << output[i];
            std::cout << "],\"text\":" << json_string(text) << ",\"hook_calls\":{";
            bool first_hook = true;
            for (auto [index, edit] : calibration.edits) {
                std::cout << (first_hook ? "" : ",") << '"' << index << "\":{\"prefill\":" << edit.prefill
                          << ",\"decode\":" << edit.decode << ",\"delta\":" << edit.delta << "}";
                first_hook = false;
            }
            std::cout << "}}" << std::endl;
        }
    }
    llama_free(ctx);
    llama_model_free(model);
    std::cout << "COMPLETE" << std::endl;
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    return 1;
}
