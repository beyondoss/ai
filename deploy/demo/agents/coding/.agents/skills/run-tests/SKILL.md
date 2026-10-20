---
name: run-tests
description: Run this project's test suite and get a one-line verdict. Use whenever you need to know whether the app's tests pass.
---

# Running the tests

Run the script from the workspace root:

```sh
bash .agents/skills/run-tests/run-tests.sh
```

It runs every test in `app/`, prints the unittest output, and ends with one line:

`RUN-TESTS-SKILL: <N> run, <F> failed, <E> errors`

Report that line. Exit status 0 means every test passed.
