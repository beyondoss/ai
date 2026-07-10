# Third-Party Notices

`crates/agent-core` and `crates/agent` are a Rust reimplementation modeled on
[pi](https://github.com/badlogic/pi-mono) — the agent loop architecture, wire-dialect
behavior, tool set, prompt/skill conventions, and a number of literal strings (tool
descriptions, truncation markers, system-prompt fragments) are ported or adapted from
that project. No pi source files are vendored or compiled into this repository; this
notice exists to carry forward pi's own license terms per the MIT License's own
condition ("The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software").

The `web` tool's command surface (its `fetch`/`outline`/`locate`/`extract`/`table` modes, the
`title=a, href=a@href` row-extraction spec, and the `--where` filter grammar) is modeled on
[ax](https://github.com/yusukebe/ax), reimplemented in Rust. No ax source is vendored; ax's own MIT
terms are carried forward below on the same basis as pi's.

## pi (github.com/badlogic/pi-mono)

```
MIT License

Copyright (c) 2025 Mario Zechner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## ax (github.com/yusukebe/ax)

```
MIT License

Copyright (c) 2026 Yusuke Wada

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
