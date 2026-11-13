# What the final comparison found

We completed the promised comparison and report: 1,440 controlled physical simulator drives, then 36 integration drives with the actual trained model enabled in 18 of them. No reply was substituted for the trained model. The original source and previous studies remain preserved.

The integrated version finished 72/72 nominal conditions with no crashes. Unprotected RL finished 0/72; static CBF finished 24/72; unshielded timer routing finished 14/72. The full protection and replanning package clearly helps compared with those simpler systems.

It made 1.08 delayed requests per nominal drive versus 75.82 for 10 Hz polling: 98.6% fewer, with equal observed success. These controlled requests use a state rule, not actual VLM inference. Simple gated/benefit triggers match that count, and uncertainty wake-up uses fewer. The fancy deadline is not the strongest method here.

Blending reduced mean physical jerk RMS from 5.07 to 3.97 and mean peak from 93.25 to 75.74 m/s^3 compared with the matched no-blend version. Individual cases can get worse; this is an average result.

With 18 seconds of decision delay, extra stopping room raised completion from 8/24 to 24/24. The original inflation version also finished 24/24. Adding the pending margin reduced stress jerk RMS from 7.28 to 2.31 but made completion about 4 seconds slower. The margin-on arms had zero observed pending collisions; their necessity is unproved because protected margin-off arms were also safe.

In the actual camera-model subset, integrated, uncertainty-triggered and cheap local replanning each finished 6/6 worlds; matched timer routing finished 5/6. All four had zero crashes. The timer miss included a response longer than 45 seconds, so different CPU timing prevents a clean scheduler comparison.

There were 22 real predictions across 15 queried episodes, zero fallbacks, and 12 applied model lane-target changes. All twelve were overtakes; no model return-right target was applied before termination. The model is genuinely connected, but this small test does not show it beats a cheap rule that already knows traffic positions.

This meets the original course scope: combine existing robotics and machine-learning pieces, compare them fairly, and report what helps. It does not establish a new algorithm, official benchmark superiority, or real-vehicle safety.
