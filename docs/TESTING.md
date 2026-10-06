# Testing conventions

## A NEGATIVE test

Most tests in this repo assert that something GOOD happens. A **NEGATIVE** test instead pins the refusal:
it asserts that a specific wrong thing is refused, or that a specific guard mechanism's absence makes the
test fail. The discipline that earns the label is procedural, not a type annotation: before the test was
committed, its guard was reverted (or removed) and the test was watched fail. Only then was the guard put
back and the test committed passing. A NEGATIVE test with no fail-run behind it is just an assertion that
happens to be phrased as a refusal — the label is a claim about how the test was built, not about its
wording.

This matters most for a HERMETIC test (no control plane, no network, no real pack fetch, no real sleeps):
hermetic tests are cheap to keep green by accident, because a broken mock can make everything pass. Watching
the test fail first is what proves the mock is still wired to the thing it is supposed to catch.

### §3b.3 — real shapes over invented spellings

When a test fixture stands in for a wire shape or a presigned-URL shape that a real integration emits, the
fixture must be the REAL shape — copied from the actual emitter, line-cited in a comment — never a spelling
the test author invented that happens to look plausible. An invented spelling can drift from production
silently; a copied one breaks the moment either side moves, which is the point.
