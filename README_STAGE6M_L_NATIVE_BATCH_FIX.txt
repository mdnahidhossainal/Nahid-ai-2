Stage 6M-L: Native persistent Qwen judge prompt-decode hang fix

Problem found by Stage 6M-K probe:
- Native libraries load OK.
- initJudgeSession/model load/context create OK in ~2 seconds.
- isJudgeReady=true.
- Hang/timeout happens at judgeRoutePersistent(prompt, maxTokens), inside prompt eval/token generation.

Likely cause fixed here:
- Previous persistent native context used n_ctx=512 and n_batch=256.
- Real judge prompt can be ~300-400 tokens.
- A prompt batch larger than n_batch can hang/take too long on Android.

Changes:
- n_ctx: 1024
- n_batch: 1024
- Prompt decode bounded to <=896 tokens.
- libc++_shared.so packaging workflow preserved.

After GitHub Actions success, copy all files from artifact arm64-v8a into:
/storage/emulated/0/Download/NahidAI/app/src/main/jniLibs/arm64-v8a/

Required files:
libc++_shared.so
libggml.so
libggml-base.so
libggml-cpu.so
libllama.so
libnahid_qwen_judge.so
