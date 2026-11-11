# Source-access correction after the component experiment freeze

The pasted teammate note described ved-testing as unpushed. `git ls-remote`
and GitHub branch/fork/pull-request checks exposed only main and harsh-testing.
We initially inferred that the teammate source was unavailable. That inference
was too strong: an exact GitHub commit lookup succeeded, followed by an exact-SHA
fetch from origin and a detached read-only worktree at:

`1f16b7ff59aae20c2a9adfbeb4943762d4b1f5b4`

The source and committed evaluation rows are now recovered and independently
reviewed. A hash-preserved copy is included under teammate-source/vlm-router.
This correction was communicated to the user immediately on discovery. The
frozen protocol is retained as written; this addendum records the later access
finding without retroactively altering the experiment design or source lock.

The remaining dependency is narrower: the fine-tuned adapter/merged weights
exist only in the teammate's private Modal volume. The source does not publish
a model endpoint, export, Hugging Face repository or W&B model artifact. No
teammate model weights or Modal credentials are available on this host.

The physical factorial remains rule/fault-based and cannot be relabeled as
trained-VLM driving. The recovered static results can be audited, but are not a
replacement for executing inference with the trained checkpoint. A new optional
v1 prompt-compatibility path is separate from the clean prompt; neither inherits
the reported picture accuracy without fresh evaluation on its actual workload.
