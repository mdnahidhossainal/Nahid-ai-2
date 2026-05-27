
#include <jni.h>
#include <string>
#include <vector>
#include <mutex>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>

#include "llama.h"

static std::mutex g_mutex;
static llama_model * g_model = nullptr;
static llama_context * g_ctx = nullptr;
static const llama_vocab * g_vocab = nullptr;
static std::string g_model_path;
static bool g_backend_inited = false;

static std::string bool_ok(bool v) { return v ? "OK ✅" : "NO ❌"; }

static void backend_init_once() {
    if (!g_backend_inited) {
        llama_backend_init();
        g_backend_inited = true;
    }
}

static std::string jstr(JNIEnv * env, jstring s) {
    if (!s) return "";
    const char * c = env->GetStringUTFChars(s, nullptr);
    std::string out = c ? c : "";
    if (c) env->ReleaseStringUTFChars(s, c);
    return out;
}

static jstring ret(JNIEnv * env, const std::string & s) {
    return env->NewStringUTF(s.c_str());
}

static void free_session_locked() {
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    g_vocab = nullptr;
    g_model_path.clear();
}

static bool is_eog_token(llama_token tok) {
    if (!g_vocab) return false;
    return llama_vocab_is_eog(g_vocab, tok);
}

static std::string token_to_piece(llama_token tok) {
    if (!g_vocab) return "";
    char buf[256];
    int n = llama_token_to_piece(g_vocab, tok, buf, sizeof(buf), 0, true);
    if (n < 0) {
        std::vector<char> big((size_t)(-n) + 8);
        n = llama_token_to_piece(g_vocab, tok, big.data(), (int)big.size(), 0, true);
        if (n > 0) return std::string(big.data(), (size_t)n);
        return "";
    }
    if (n > 0) return std::string(buf, (size_t)n);
    return "";
}

static std::string init_session_locked(const std::string & model_path) {
    std::ostringstream log;
    log << "STAGE6M_PERSISTENT_SESSION_INIT\n";
    log << "MODEL_PATH=" << model_path << "\n";

    backend_init_once();

    if (g_model && g_ctx && g_model_path == model_path) {
        log << "SESSION_ALREADY_READY ✅\n";
        return log.str();
    }

    free_session_locked();

    llama_model_params mparams = llama_model_default_params();
    g_model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!g_model) {
        log << "MODEL_LOAD_FAILED ❌\n";
        return log.str();
    }
    log << "MODEL_LOAD_OK ✅\n";

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 1024;
    cparams.n_batch = 1024;
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;

    g_ctx = llama_init_from_model(g_model, cparams);
    if (!g_ctx) {
        log << "CONTEXT_CREATE_FAILED ❌\n";
        free_session_locked();
        return log.str();
    }
    g_vocab = llama_model_get_vocab(g_model);
    g_model_path = model_path;

    log << "LLAMA_CONTEXT_CREATE_OK ✅\n";
    log << "VOCAB_READY=" << bool_ok(g_vocab != nullptr) << "\n";
    log << "STAGE6M_PERSISTENT_SESSION_READY ✅\n";
    return log.str();
}

static std::vector<llama_token> tokenize_locked(const std::string & prompt, bool add_bos=true, bool parse_special=true) {
    std::vector<llama_token> result;
    if (!g_vocab) return result;
    int n = llama_tokenize(g_vocab, prompt.c_str(), (int)prompt.size(), nullptr, 0, add_bos, parse_special);
    if (n < 0) n = -n;
    if (n <= 0) return result;
    result.resize((size_t)n);
    int got = llama_tokenize(g_vocab, prompt.c_str(), (int)prompt.size(), result.data(), n, add_bos, parse_special);
    if (got < 0) got = -got;
    if (got > 0 && got <= n) result.resize((size_t)got);
    return result;
}

static llama_token sample_greedy_locked() {
    const float * logits = llama_get_logits_ith(g_ctx, -1);
    if (!logits || !g_vocab) return 0;
    int n_vocab = llama_vocab_n_tokens(g_vocab);
    int best_id = 0;
    float best = -INFINITY;
    for (int i = 0; i < n_vocab; ++i) {
        if (is_eog_token((llama_token)i)) continue;
        float v = logits[i];
        if (v > best) {
            best = v;
            best_id = i;
        }
    }
    return (llama_token)best_id;
}

static std::string generate_locked(const std::string & prompt, int max_tokens) {
    std::ostringstream log;
    log << "STAGE6M_PERSISTENT_JUDGE_ROUTE\n";
    if (!g_model || !g_ctx || !g_vocab) {
        log << "SESSION_NOT_READY ❌\n";
        return log.str();
    }

#if defined(LLAMA_API)
    // no-op: marker only
#endif

    // Clear KV/memory so each route request is independent while model stays loaded.
#if defined(__cplusplus)
    llama_memory_clear(llama_get_memory(g_ctx), true);
#endif

    auto toks = tokenize_locked(prompt, true, true);
    log << "PROMPT_TOKENS_ORIGINAL=" << toks.size() << "\n";
    if (toks.empty()) {
        log << "TOKENIZE_FAILED ❌\n";
        return log.str();
    }

    // Stage 6M-L fix:
    // The previous persistent build used n_batch=256 and n_ctx=512.
    // The real Judge prompt is often ~300-400 tokens. Passing a batch larger
    // than n_batch can hang or take a very long time on Android. Keep the model
    // loaded, but make the single prompt decode safe and bounded.
    const int max_prompt_tokens = 896; // leaves room for generated route tokens inside n_ctx=1024
    if ((int)toks.size() > max_prompt_tokens) {
        log << "PROMPT_TRUNCATED_FOR_SAFE_DECODE_FROM=" << toks.size() << " TO=" << max_prompt_tokens << "\n";
        toks.erase(toks.begin(), toks.end() - max_prompt_tokens);
    }
    log << "PROMPT_TOKENS_USED=" << toks.size() << "\n";

    llama_batch batch = llama_batch_get_one(toks.data(), (int)toks.size());
    int rc = llama_decode(g_ctx, batch);
    log << "PROMPT_DECODE_RC=" << rc << "\n";
    if (rc != 0) {
        log << "PROMPT_DECODE_FAILED ❌\n";
        return log.str();
    }

    std::string text;
    int steps = std::max(1, std::min(max_tokens, 128));
    for (int i = 0; i < steps; ++i) {
        llama_token tok = sample_greedy_locked();
        if (tok == 0 || is_eog_token(tok)) {
            log << "EOG_AT_STEP=" << i << "\n";
            break;
        }
        std::string piece = token_to_piece(tok);
        text += piece;

        llama_batch b = llama_batch_get_one(&tok, 1);
        int drc = llama_decode(g_ctx, b);
        if (drc != 0) {
            log << "DECODE_TOKEN_FAILED_AT=" << i << " RC=" << drc << "\n";
            break;
        }

        // Route JSON should finish quickly; stop once a JSON object looks closed.
        if (text.find('}') != std::string::npos) break;
    }

    log << "GENERATED_TEXT=" << text << "\n";
    log << "STAGE6M_PERSISTENT_GENERATION_OK ✅\n";
    return log.str();
}


static int first_token_for_candidate_locked(const std::string & text) {
    auto toks = tokenize_locked(text, false, true);
    if (!toks.empty()) return (int)toks[0];
    return -1;
}

static std::string score_route_locked(const std::string & prompt) {
    std::ostringstream log;
    log << "STAGE6M_Q_NATIVE_ROUTE_WORD_SCORER\n";
    if (!g_model || !g_ctx || !g_vocab) {
        log << "SESSION_NOT_READY ❌\n";
        return log.str();
    }

    // No free text generation here. The Judge scores route WORD choices: CHAT / SCREEN / CONFIRM.
    llama_memory_clear(llama_get_memory(g_ctx), true);

    auto toks = tokenize_locked(prompt, true, true);
    log << "PROMPT_TOKENS_ORIGINAL=" << toks.size() << "\n";
    if (toks.empty()) {
        log << "TOKENIZE_FAILED ❌\n";
        return log.str();
    }
    const int max_prompt_tokens = 896;
    if ((int)toks.size() > max_prompt_tokens) {
        log << "PROMPT_TRUNCATED_FOR_SAFE_DECODE_FROM=" << toks.size() << " TO=" << max_prompt_tokens << "\n";
        toks.erase(toks.begin(), toks.end() - max_prompt_tokens);
    }
    log << "PROMPT_TOKENS_USED=" << toks.size() << "\n";

    llama_batch batch = llama_batch_get_one(toks.data(), (int)toks.size());
    int rc = llama_decode(g_ctx, batch);
    log << "PROMPT_DECODE_RC=" << rc << "\n";
    if (rc != 0) {
        log << "PROMPT_DECODE_FAILED ❌\n";
        return log.str();
    }

    const float * logits = llama_get_logits_ith(g_ctx, -1);
    if (!logits) {
        log << "LOGITS_NULL ❌\n";
        return log.str();
    }

    struct Cand { const char * word; const char * route; const char * tokenText1; const char * tokenText2; int token; float score; };
    Cand cands[3] = {
        {"CHAT", "GEMINI_CONVERSATION", " CHAT", " chat", -1, -INFINITY},
        {"SCREEN", "SCREEN_LOCATOR", " SCREEN", " screen", -1, -INFINITY},
        {"CONFIRM", "NEED_CONFIRMATION", " CONFIRM", " confirm", -1, -INFINITY}
    };

    for (auto & c : cands) {
        c.token = first_token_for_candidate_locked(c.tokenText1);
        if (c.token < 0) c.token = first_token_for_candidate_locked(c.tokenText2);
        if (c.token >= 0) c.score = logits[c.token];
        log << "CANDIDATE_" << c.word << "_ROUTE=" << c.route << "\n";
        log << "CANDIDATE_" << c.word << "_TOKEN=" << c.token << "\n";
        log << "CANDIDATE_" << c.word << "_SCORE=" << c.score << "\n";
    }

    int best = 0;
    for (int i = 1; i < 3; ++i) {
        if (cands[i].score > cands[best].score) best = i;
    }

    log << "WINNER_WORD=" << cands[best].word << "\n";
    log << "ROUTE=" << cands[best].route << "\n";
    log << "STAGE6M_Q_NATIVE_ROUTE_WORD_SCORER_OK ✅\n";
    return log.str();
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nahidai_assistant_core_judge_QwenJudgeNativeBridge_initJudgeSession(
        JNIEnv * env, jclass, jstring modelPath) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return ret(env, init_session_locked(jstr(env, modelPath)));
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_nahidai_assistant_core_judge_QwenJudgeNativeBridge_isJudgeReady(
        JNIEnv *, jclass) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return (g_model && g_ctx && g_vocab) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nahidai_assistant_core_judge_QwenJudgeNativeBridge_judgeRoutePersistent(
        JNIEnv * env, jclass, jstring prompt, jint maxTokens) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return ret(env, generate_locked(jstr(env, prompt), (int)maxTokens));
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nahidai_assistant_core_judge_QwenJudgeNativeBridge_scoreRoutePersistent(
        JNIEnv * env, jclass, jstring prompt) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return ret(env, score_route_locked(jstr(env, prompt)));
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nahidai_assistant_core_judge_QwenJudgeNativeBridge_releaseJudgeSession(
        JNIEnv * env, jclass) {
    std::lock_guard<std::mutex> lock(g_mutex);
    free_session_locked();
    return ret(env, "STAGE6M_PERSISTENT_SESSION_RELEASED ✅");
}

// Backward-compatible Stage 6D/6E/6F/6G JNI symbols used by the current Android app.
extern "C" JNIEXPORT jstring JNICALL
Java_com_nahidai_assistant_core_judge_QwenJudgeNativeBridge_nativeProbeStage6D(
        JNIEnv * env, jclass) {
    std::ostringstream log;
    log << "STAGE6D_NATIVE_PROBE\n";
    backend_init_once();
    log << "llama_backend_init: OK\n";
    log << "STAGE6D_NATIVE_PROBE_OK ✅\n";
    return ret(env, log.str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nahidai_assistant_core_judge_QwenJudgeNativeBridge_nativeProbeStage6E(
        JNIEnv * env, jclass, jstring modelPath) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return ret(env, init_session_locked(jstr(env, modelPath)));
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nahidai_assistant_core_judge_QwenJudgeNativeBridge_nativeProbeStage6F(
        JNIEnv * env, jclass, jstring modelPath, jstring prompt) {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::string init = init_session_locked(jstr(env, modelPath));
    std::ostringstream log;
    log << "STAGE6F_MODEL_LOAD_PROMPT_EVAL_PROBE\n" << init;
    if (g_ctx && g_vocab) {
        llama_memory_clear(llama_get_memory(g_ctx), true);
        auto toks = tokenize_locked(jstr(env, prompt), true, true);
        log << "TOKENIZE_RESULT=" << toks.size() << "\n";
        if (!toks.empty()) {
            llama_batch batch = llama_batch_get_one(toks.data(), (int)toks.size());
            int rc = llama_decode(g_ctx, batch);
            log << "LLAMA_DECODE_RC=" << rc << "\n";
            if (rc == 0) log << "STAGE6F_MODEL_EVAL_OK ✅\n";
        }
    }
    return ret(env, log.str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nahidai_assistant_core_judge_QwenJudgeNativeBridge_nativeProbeStage6G(
        JNIEnv * env, jclass, jstring modelPath, jstring prompt, jint maxTokens) {
    std::lock_guard<std::mutex> lock(g_mutex);
    std::string init = init_session_locked(jstr(env, modelPath));
    std::ostringstream log;
    log << "STAGE6G_FIRST_JSON_GENERATION_PROBE\n" << init;
    log << generate_locked(jstr(env, prompt), (int)maxTokens);
    return ret(env, log.str());
}
