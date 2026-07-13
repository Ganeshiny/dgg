# Strict Code Modification Protocol

Whenever Antigravity (or any subagent) modifies code, it MUST follow this strict verification protocol to ensure no logic is broken, no lines are lost, and no syntax errors are introduced. 

## Pre-Modification Baseline Protocol (Step 0 — runs BEFORE any edit)

Before touching any file, you MUST establish a frozen baseline.

- Run `python -m py_compile <target_file.py>` on the UNMODIFIED file first. If it already fails, STOP and report the pre-existing error before making any changes. Never conflate a pre-existing bug with one you introduced.
- Run `git stash list` and note any stashed state that could interfere.
- Identify every other file that imports or calls the function/class you are about to modify. Run `grep -rn "<function_or_class_name>" .` and record all call sites. You are responsible for verifying those call sites still work after your edit.

## 1. Contextual Diff Review (Check 1)
After using `replace_file_content` or `multi_replace_file_content`, you MUST run `git diff` on the terminal to manually review the exact lines that were added and removed. 
- Look explicitly for "dangling" lines left outside of functions.
- Verify that entire function blocks were not accidentally wiped out by a greedy regex/string match.
- Ensure the indentation of the injected code perfectly matches the surrounding context.

## 2. Syntax Compilation (Check 2)
Before claiming a task is complete, you MUST run a strict compiler check on the modified files to catch `IndentationError` or `SyntaxError`.
- For Python files: Run `python -m py_compile <modified_file.py>` in the terminal.
- Do not proceed until this command returns `0` (success) with no output.

## 3. Logical Tracing (Check 3)
Manually trace the logic of the modified code across the file.
- **Variables**: Ensure all variables used in the new block are defined in the current scope.
- **Signatures**: Check that the function parameters match what the caller is providing (e.g. if a function expects `datasets`, ensure the caller passes `datasets`).
- **Dependencies**: Ensure any newly required imports (like `os`, `numpy`, `torch`) are present at the top of the file.

## Check 4: Runtime Import Verification

After py_compile succeeds (Check 2), syntax correctness does not guarantee the module loads. Run:

```bash
python -c "import <module_name>"
```

A clean import catches:
- Circular imports introduced by new import statements
- Missing `__init__.py` entries
- Module-level code that raises on import (e.g. a constant computed from a missing file)

Do not proceed until this exits with code 0 and no output.

## Check 5: Cross-File Call Site Verification

For every call site identified in Step 0:
- Confirm the function signature you modified still matches what the caller passes. Pay explicit attention to: added/removed positional arguments, renamed keyword arguments, changed return types.
- If a call site passes a value by position and you added a parameter in the middle of the signature, this is a silent breakage that py_compile will not catch. Trace it manually.

## Check 6: Data Contract Assertions (mandatory for numpy/tensor code)

Any modification to data loading, splitting, preprocessing, or model forward passes MUST include an explicit shape/dtype assertion trace.

After the edit, manually verify:
- Array shapes at every stage of the modified pipeline match the expected contract (e.g. `train_labels.shape[1] == len(go_terms)`)
- No `.npy` file is being read from test-split data to compute training-time statistics (IC weights, label vocabularies, class frequencies). This is the vocabulary leakage pattern. If any statistic is computed on a file whose name contains `test`, flag it immediately and do not proceed.
- Label index alignment: if GO term order is derived from a sorted list, verify the sort is applied consistently across train, val, and test label matrices.

## Check 7: Failure Modes and Rollback

If ANY check (1–6) fails:
1. Run `git diff` immediately to isolate what you changed.
2. Run `git checkout -- <file>` to restore the file to its pre-edit state.
3. Report the exact failure output verbatim — do not paraphrase error messages.
4. Do not attempt a second edit until you have explained the root cause of the first failure and stated the precise correction you will make.

Never silently retry. Never tell the user "let me fix that" and immediately re-edit without the above sequence.

## Check 8: Execution Path Coverage (for logic changes)

When modifying a conditional branch, loop, or data-dependent path:
- Enumerate all distinct execution paths through the modified block (true/false branches, empty input, single element, normal batch).
- For each path, state explicitly whether it is exercised by the existing test suite or is untested.
- If a path is untested and the modification affects it, write a minimal inline assertion or test call that exercises it, run it, and confirm it passes before marking the task complete.

## Check 9: No Suppressed Exceptions

After any edit, scan the modified file for bare `except:`, `except Exception: pass`, and logging-only exception handlers that swallow errors silently. Any such pattern in code you have touched must be flagged and either fixed or explicitly justified in your report.

## Completion Gate

You are not permitted to tell the user the task is complete until you have explicitly stated, in your final message, which of the following checks were performed and the result of each:

- [ ] Step 0: Baseline compile passed / call sites identified (list them)
- [ ] Check 1: git diff reviewed, no dangling lines, no wiped blocks
- [ ] Check 2: py_compile passed
- [ ] Check 3: Logic traced — variables in scope, signatures match, imports present
- [ ] Check 4: Runtime import passed
- [ ] Check 5: All call sites verified against new signature
- [ ] Check 6: Data contracts and shape assertions verified / leakage audit passed
- [ ] Check 7: No rollback was needed (or: rollback performed, root cause stated)
- [ ] Check 8: All execution paths enumerated; untested paths identified
- [ ] Check 9: No suppressed exceptions in modified scope

If any checkbox is not ticked, state why it was not applicable. Do not omit it silently.
