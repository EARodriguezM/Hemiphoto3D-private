1. cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
2. ctest --output-on-failure
3. Report: total tests, passed, failed
4. For any failure: show the error, diagnose root cause, fix it
5. Re-run failed tests to confirm the fix
