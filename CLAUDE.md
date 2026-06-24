# CLAUDE.md

## Building

This package depends on mlx-swift (Metal/C++). Plain `swift build` and Xcode-beta
fail here; build/test with the stable Xcode toolchain:

```bash
env -u TOOLCHAINS DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild test -scheme mlx-swift-rf-detr-Package -destination 'platform=macOS'
```

`swift package` commands (e.g. DocC) hit a swiftly PATH shim that selects an
incompatible toolchain — prepend the Xcode toolchain bin so `swift` is Xcode's:

```bash
XCTC=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin
env -u TOOLCHAINS PATH="$XCTC:$PATH" DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  ./Scripts/build_docs.sh
```

## Documentation

`MLXRFDETR` ships DocC-generated reference docs (see
`Sources/MLXRFDETR/Documentation.docc/` and `Scripts/build_docs.sh`).
**`///` doc comments on public/`open` symbols are published** to the static site
at https://mnmly.github.io/mlx-swift-rf-detr/ and (if `EMIT_LLMS_TXT=1` is used)
into `docs/llms.txt`.

When you add or modify a `public` or `open` declaration:

- Write a `///` doc comment. One-sentence summary, then a paragraph if the *why*
  is non-obvious. Skip restating what the signature already says.
- Document each parameter with `- Parameter name:` (use the **internal** name when
  there's an external label — DocC warns otherwise).
- Cross-reference related symbols with double-backtick links, e.g.
  `` ``RFDETR/load(directory:dtype:)`` ``. DocC link syntax is signature-sensitive:
  `foo(_:)` and `foo(_:_:)` are different.
- When you add a new top-level public symbol that belongs in the curated sidebar,
  add it under the appropriate `## Topics` group in
  `Sources/MLXRFDETR/Documentation.docc/MLXRFDETR.md`. Topics are organized by
  *user task*, not alphabetic order.

Verify before declaring documentation work done (using the build invocation above):

```bash
Scripts/build_docs.sh
```

Expect exit 0 and no new "doesn't exist at" or "external name used to document
parameter" warnings attributable to your changes.
