# Stage 6M-B — Persistent Qwen Judge native build

এই patch শুধু `/storage/emulated/0/Download/get hub` folder-এর জন্য।

কাজ:
- নতুন `libnahid_qwen_judge.so` build করবে।
- পুরোনো Stage 6D/6E/6F/6G JNI method রাখবে।
- নতুন persistent session JNI method যোগ করবে:
  - `initJudgeSession(modelPath)`
  - `isJudgeReady()`
  - `judgeRoutePersistent(prompt, maxTokens)`
  - `releaseJudgeSession()`

কেন দরকার:
আগের `.so` প্রতি request-এ model load করত। নতুন `.so` model/context memory-তে ধরে রাখার চেষ্টা করবে, যাতে AI CALL session-এর ভিতরে Judge route দ্রুত হয়।

Install:
1. এই ZIP extract করুন।
2. `2_PUT_THESE_INSIDE_get_hub_FOLDER` folder-এর ভিতরের files:
   `/storage/emulated/0/Download/get hub`
   folder-এর ভিতরে copy/extract করুন।
3. Replace/Overwrite দিন।
4. Termux থেকে push করুন:

```bash
cd "/storage/emulated/0/Download/get hub"
git config --global --add safe.directory "/storage/emulated/0/Download/get hub"
git add -A
git commit -m "Stage 6M persistent Qwen judge native build"
git push
```

5. GitHub Actions শেষ হলে artifact download করুন:
   `libnahid_qwen_judge_stage6M_persistent_arm64_v8a.zip`

6. artifact-এর `arm64-v8a` folder-এর files copy করুন:
   `/storage/emulated/0/Download/NahidAI/app/src/main/jniLibs/arm64-v8a/`

7. এরপর Android app build করবেন।

Important:
- এই stage শুধু native `.so` build করে।
- নতুন `.so` বসানোর পরে Java bridge patch লাগবে, যাতে app নতুন persistent methods call করে।
