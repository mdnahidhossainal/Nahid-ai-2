Stage 6M-N: Native Route Scorer

Purpose:
- Stop asking Qwen to generate route text/JSON.
- Qwen Judge should not write answers or explanations.
- Native code decodes the compact judge prompt, then compares next-token scores for A/B/C:
  A = GEMINI_CONVERSATION
  B = SCREEN_LOCATOR
  C = NEED_CONFIRMATION
- The winning score becomes the route.

New JNI method:
- scoreRoutePersistent(prompt)

After GitHub Actions success, copy all files from artifact arm64-v8a into:
/storage/emulated/0/Download/NahidAI/app/src/main/jniLibs/arm64-v8a/

Required files:
libc++_shared.so
libggml.so
libggml-base.so
libggml-cpu.so
libllama.so
libnahid_qwen_judge.so
