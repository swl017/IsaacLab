Update the iris_ma6 session progress tracking files. This is the END-OF-SESSION protocol from iris_ma6/CLAUDE.md.

## Steps

1. **Read current state**:
   - `source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/doc/active/progress.txt`
   - `source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/doc/active/feature_list.json`
   - `source/isaaclab_tasks/isaaclab_tasks/direct/iris_ma6/ARCHITECTURE.md`

2. **Review what was done this session**:
   - Check git diff (staged + unstaged) for modified files
   - Summarize changes made, bugs fixed, features added

3. **Append to progress.txt**:
   - Use today's date and a short title
   - "Done:" section listing concrete changes (files modified, tests passing, etc.)
   - "Next:" section listing follow-up tasks
   - Keep entries concise — one line per item

4. **Update feature_list.json** if any feature status changed:
   - Update `status` field ("done", "in_progress", "planned")
   - Update `notes` with date-stamped summary of changes
   - Update `last_updated` date

5. **Update ARCHITECTURE.md** if module dependencies changed

6. **Update module CONTEXT.md** if any module's interface (inputs, outputs, dependencies) changed

Only update files where changes are warranted. Do not create entries for trivial changes.
