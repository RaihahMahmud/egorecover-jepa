# EgoRecover-JEPA — Portable Codex Handoff

This ZIP is a **small code/context transfer**, not a 120 GB dataset export. It contains no original videos and no rclone OAuth token. Keep the repository **private** if possible. See `CODEX_START_PROMPT.md` for the first Codex message and `AGENTS.md` for persistent instructions.

1. Extract ZIP and upload all files/folders to a **private GitHub repository** using the GitHub website (do not upload ZIP as a single opaque file if Codex needs to read it as a repository). GitHub upload limits may require splitting into small batches.
2. Open Codex (`https://chatgpt.com/codex` or your supported Codex client) and connect/select the private repository. Confirm permissions for that repo.
3. Copy the content of `CODEX_START_PROMPT.md` into a new Codex task.
4. First request **audit and plan only**. Approve code edits after reviewing the plan. Test a Kaggle inventory and a three-video pilot before a large download.
5. Never put your Kaggle `RCLONE_TOKEN`, raw `rclone.conf`, Google Drive private access material, or raw MP4s into this repo or chat.

`AGENTS.md` is read automatically in supported Codex repository workflows. Original notebook outputs have been stripped from the ZIP to reduce leak risk; original unchanged uploads remain with the user. The epoch metrics CSV is provided separately.
