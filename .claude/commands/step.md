Read PLAN.md and implement Step $ARGUMENTS.

Follow the detailed prompt for that step exactly.
After implementation:
1. Build: make -j$(nproc) — must succeed with zero errors
2. Run tests: ctest --output-on-failure — must pass
3. Check the **verification gate** at the bottom of the step in PLAN.md
4. If the gate fails, diagnose and fix before reporting done

Do NOT proceed to any other step. Only implement what Step $ARGUMENTS asks for.
If the step says to create stubs for future files, create stubs (not full implementations).
