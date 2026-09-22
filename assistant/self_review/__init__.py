"""
Persistent record-keeping of real problems Vaayu hits during normal use, and
a daily check-in that asks permission to write them up as a report - NOT an
autonomous code-rewriting system.

Why it stops at "write a report" instead of "fix it": Vaayu's brain is a
local model running on this laptop (qwen2.5:3b originally, qwen2.5:7b-instruct
as of the current config - see config/config.yaml's own comment for the
measured tradeoff of that switch). This isn't about one specific model size:
this whole project's test suite and bug history is the evidence for why a
local model this size can't be trusted to diagnose, fix, test, and ship
changes to its own source unsupervised - the 3B needed a retry-nudge just to
reliably call a tool instead of lying about the result (see
llm/ollama_client.py's grounding fix) and sometimes hallucinated a fake tool
call as plain text even after that nudge; a live side-by-side test of the 7B
model against this project's own real bug-case commands found its own new
failure (calling the wrong status tool before the right one, on a plain
"what's my battery" question) rather than a clean improvement. Pointing
either at its own codebase wouldn't produce self-improvement, and an
unattended process that edits and redeploys code controlling real actions
(sending messages, typing passwords) with no human review is a real risk
regardless of which model is driving it. So the loop here is: log real
issues as they happen (this package) -> ask permission at a configurable
daily time -> on yes, write a clear report -> the user brings that report to
an actual, human-supervised coding session (the same way every real fix in
this project has happened).
"""
