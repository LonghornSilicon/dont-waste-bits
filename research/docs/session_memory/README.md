# Session memory (backup from Brev instance)

These are the persistent memory files Claude Code wrote during the 2026-04-20/21 research sessions on Brev. The originals live at:

```
~/.claude/projects/-home-shadeform-dont-waste-bits/memory/
```

When Claude Code runs locally on your machine, it looks for memory files at that path. To restore context when you start a fresh session locally:

```bash
# On your local machine, after cloning the repo:
mkdir -p ~/.claude/projects/-home-shadeform-dont-waste-bits/memory
cp research/docs/session_memory/*.md \
   ~/.claude/projects/-home-shadeform-dont-waste-bits/memory/
```

Then when you start Claude Code in the project directory, it will auto-load these as memory.

## File inventory

- **MEMORY.md** — the index (loaded automatically into every Claude session's context).
- **project_phase7_result.md** — findings from the Phase 7 ablation study (routing-is-noise, p4 sweep).
- **project_fpga_layout_decision.md** — decision matrix tying Phase 7j/7k/7m/rotation outcomes to 130nm silicon options (B / A / E).
- **project_turboquant_revisit_plan.md** — plan for the offline-rotation investigation (R2 tested and null, R3 requires runtime Hadamard).
- **reference_paper_artifacts.md** — paths to paper tex, figure scripts, experiment directories.

## Re-compilation checklist (to resume where we left off)

1. Clone the repo.
2. Restore memory files above.
3. Open Claude Code in the repo directory. Prompt something like: *"continue the paper-split work; we just finished Paper A, next up is Paper C"*.
4. Claude will see `MEMORY.md` in context and have the decision matrix + phase-7 findings + paper-split plan ready to use.
