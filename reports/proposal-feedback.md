# Feedback response notes (team only — not in the PDF)

## 1. Spoken-English answer to the delay-margin question

Fresh sensors are fine — that is not what we are worrying about. While the LLM is thinking, the car is stuck with whatever skill it already picked (cruise, pass, return), so the fast controller keeps acting for that old plan even though it can still see the road. The base filter can still brake from current geometry, but if we do not leave a little extra room, we either brake late or keep fighting the shield because the locked skill still wants to make progress. So the pending-call margin is buffer for “we cannot change the plan yet,” not a hack for stale observations. That is the opposite of Periotto-style teleop delay, where the plant feedback itself is late; here state is fresh and the advice is late. We will turn the margin off and on in an ablation and only keep it if the numbers say it helps.

## 2. How the revision addresses each feedback bullet

1. **Why grow the safety margin while the LLM thinks?**  
   New Section 2 (“Why Grow Clearance While the LLM Thinks”) states that perception stays current; the delayed quantity is high-level skill choice; the RL skill stays locked to the old intent; the margin is time to revise the plan; Periotto is delayed plant feedback vs our delayed semantic/skill advice; and the margin is a hypothesis to ablate, not a guaranteed win.

2. **Elaborate how the setting differs from prior literature.**  
   Section 3 (“How This Differs from Prior Work”) is rewritten as short contrast bullets vs LeapAD / DriveVLM-Dual, FASIONAD++ / AdaDrive, CERT-Replan, Yu et al., and Periotto. AdaDrive is added to `references.bib`.

3. **Later parts sounded like LLM writing.**  
   Approach, Evaluation, and Scope are rewritten as short concrete sentences: what we train, what we freeze, what we compare, what we measure, and what we will not claim. Buzzwords and “we will comprehensively evaluate…” filler are cut.
