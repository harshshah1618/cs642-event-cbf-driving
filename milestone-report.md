# Safe Fast Control While a Slow Language Model Thinks
Milestone report | November 12, 2026
Akash Ravi, Harsh Shah, Ved Chadderwala

## Aim and current progress

Our project asks how a car can keep driving safely while a slower model decides what it should do next. We have built a working driving system in a simulator, connected the learned driver and the camera model, and completed comparisons with simpler systems. The main finding so far is that safety checks and changes of plan help a great deal. We have not yet shown that the learned driver or language model is needed for the tasks we tested.

## What we have built

The system has three parts. A planner looks at road images and suggests whether to stay in the right lane, pass on the left, or return right. A faster driver produces the steering and speed commands. It learned through practice in simulation, using a training method called PPO. A safety filter can reduce acceleration or brake when those commands would be unsafe. We also check the target lane before accepting a change and gradually mix the old and new commands to make the switch less sudden. Returning right uses the same driving skill as holding the right lane.

The main tests now run in MetaDrive, where the car and traffic move under simulated vehicle physics. The driver and safety checks continue working while a model reply is pending. A late reply is checked against the current scene before it can change the plan. We saved the settings, model inputs, replies and driving records so the comparisons can be checked and repeated. The software also passed 369 automated tests.

## What we have tested

We compared 15 versions in 1,440 simulated drives. These used three situations: a stopped car ahead, a stopped car followed by a vehicle cutting in, and denser traffic. We used eight traffic starting layouts and four waiting times. Each comparison starts from the same traffic conditions. The table covers the 0-1 second waits, giving 72 runs per version; the separate 18-second test checks unusually slow replies.

| Version | Finished | Crashes | Requests per drive |
|---|---:|---:|---:|
| Learned driver, no safety filter | 0/72 | 72 | 0 |
| Safety filter, no change of plan | 24/72 | 12 | 0 |
| Ask once per second, with protections | 72/72 | 0 | 13.82 |
| Ask up to ten times per second, with protections | 72/72 | 0 | 75.82 |
| Ask when needed, with protections | 72/72 | 0 | 1.08 |
| Simple local rules, with protections | 72/72 | 0 | 0* |

These comparisons use a traffic rule with a controlled waiting time, not the camera model. A finished run means no crash and at least 99% of the route. *The local replanner uses cheap rules rather than model requests. The protections include a braking backup and lane checks, so the gains cannot all be credited to one filter formula.

<!-- PAGE BREAK -->

## What the results tell us

Safety and progress are different problems. Braking can prevent a crash but leave the car waiting behind an obstacle. Changing the plan lets it continue. Our system asks for a new plan when the driving situation calls for one. It finished the same short-delay runs as asking up to ten times per second while making about 98.6% fewer requests. However, simpler rules for when to ask also worked well, and one used fewer requests than ours. Our rule for when to ask has not proved best.

We also found that a correct instruction can arrive after the car has stopped too close to turn. Keeping eight extra metres of stopping room while waiting, then letting the car use that room once a turn is accepted, helped in the 18-second tests: completion rose from 8/24 without the reserve to 24/24 with it. The original extra-margin version also finished 24/24, so our repair is not the only solution. Gradually changing commands reduced sudden changes in physical acceleration on average, although some individual cases became worse.

The extra margin while waiting has a tradeoff. Both margin-on and margin-off versions avoided collisions in the protected tests, so we cannot say the margin was needed for safety. At long delays it made the driving smoother but added about four seconds to average completion time. This is a choice between progress and comfort, rather than a clear win in every measure.

## Check with the actual camera model

We connected our teammate's trained camera model and ran 36 additional comparison drives. Eighteen allowed model calls; 15 actually requested an answer, producing 22 real replies with no substitute answers. On six traffic setups, our model-based version finished six, asking on a timer finished five, and the simple local rule also finished six. All three avoided crashes. Model response times differed between runs, and the timer's missed route included a reply that arrived too late. This small test does not prove that our model or rule for when to ask is better. All model decisions applied during driving were passes on the left; a model-directed return was not demonstrated.

## What is still missing

Our roads have been straight and the task is limited. The driving skills were trained on empty roads. Earlier comparisons also found that a hand-written lane controller performed similarly to PPO. The rule-based planner receives exact traffic positions, which makes its decisions easier than interpreting images. Our current results therefore show a useful combined system, but do not establish that PPO or the language model is essential. They also do not show that we beat the published systems mentioned in the proposal. The control rate is measured in simulation time, and the CPU camera-model runs do not establish real-time performance on a vehicle.

## Where we think this is leading

The next step is to find a task where learning adds a measurable benefit. We are considering curved roads with changing vehicle response for the driver, and instructions such as passing a vehicle and returning right before a marked point for the planner. These harder tasks have not been built or tested yet. We will choose one fixed test setting, train for that task, compare against strong simple controllers, and keep separate cases for the final evaluation. The aim is to identify which parts improve safety, progress or smoothness enough to justify their cost, while keeping the working system and the results already collected.
