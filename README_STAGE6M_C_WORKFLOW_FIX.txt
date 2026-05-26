Stage 6M-C workflow fix

This patch replaces only the GitHub Actions workflow.
It stops building llama.cpp tools/server/examples and builds only the required shared library target.

Install:
1) Extract this ZIP.
2) Copy 2_PUT_THESE_INSIDE_get_hub_FOLDER contents into:
   /storage/emulated/0/Download/get hub
3) Replace/overwrite .github/workflows/build-qwen-persistent-judge-so.yml
4) Push:
   cd "/storage/emulated/0/Download/get hub"
   git add -A
   git commit -m "Fix Stage 6M workflow build only llama shared libs"
   git push -u origin main --force

Expected: GitHub Actions should pass the llama.cpp shared library step and continue to Build libnahid_qwen_judge.so.
