# 🔄 File Replacement Rules

(Paste this to ensure safe file updates)

---

# 🔄 **FILE REPLACEMENT RULES**

When updating or creating files, strictly follow these rules:

1.  **Full Replacement**: Always provide the **COMPLETE** file content. Do not use diffs or "rest of file" placeholders unless explicitly authorized for massive files.
2.  **Verification**: Before writing, verify the target path exists. Create parent directories if needed.
3.  **Safety**:
    *   **NEVER** overwrite `continuous_contract.json` (Real Data).
    *   **NEVER** overwrite `newprint.md` (Source Dump) until explicitly told to delete it.
    *   **NEVER** modify files outside the `fracfire/` directory.
4.  **Backup**: For critical configuration files, consider creating a backup (e.g., `config.yaml.bak`) before overwriting.
5.  **Atomic Writes**: If possible, write to a temporary file and rename it to ensure atomicity (though standard tool usage usually handles this).

---
