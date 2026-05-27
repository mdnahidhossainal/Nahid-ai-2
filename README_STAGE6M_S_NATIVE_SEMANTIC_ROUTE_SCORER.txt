Stage 6M-S native semantic route scorer

Purpose:
- Stop judging by fixed route word bias (CHAT/SCREEN/CONFIRM).
- Do not ask Qwen to write an answer, label, JSON, or explanation.
- Native scorer asks semantic YES/NO gates:
  1) Does the request require the CURRENT visible phone screen/UI/screenshot?
  2) If no, is it a sensitive phone action requiring confirmation?
- Then it returns only ROUTE=SCREEN_LOCATOR / NEED_CONFIRMATION / GEMINI_CONVERSATION.

After GitHub Actions success, download artifact:
libnahid_qwen_judge_stage6M_S_semantic_route_scorer_arm64_v8a

Copy all files from artifact arm64-v8a into:
/storage/emulated/0/Download/NahidAI/app/src/main/jniLibs/arm64-v8a/
