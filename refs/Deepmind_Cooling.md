# Part 02 — Real World RL Application Analysis
## DeepMind: Cooling Commercial Buildings with Reinforcement Learning

**Paper:** Controlling Commercial Cooling Systems Using Reinforcement Learning  
**Authors:** Luo et al., DeepMind & Google, 2022  
**Link:** https://arxiv.org/abs/2211.07357

---

## 1. Problem Description

Large commercial buildings need HVAC systems to keep the temperature comfortable
for the people inside. The main component we are looking at is the chiller plant,
which is basically a system that cools water and circulates it through the building
to remove heat.

The way these systems are normally controlled is through hardcoded rules written
by HVAC engineers, based on their experience and basic physics. The problem with
this approach is that the rules never change. They cannot adapt to things like
shifting weather conditions, how many people are in the building, or equipment
wearing down over time. This leads to a lot of wasted energy.

The goal of this project is to replace these static rules with a Reinforcement
Learning agent that learns to control the chiller plant in a smarter way, using
less energy while still keeping people comfortable and the equipment safe.

To understand why this matters at scale: space cooling alone accounts for around
10% of the world's total electricity demand, so even small efficiency gains in
this area can have a real impact on energy consumption globally.

---

## 2. Why Reinforcement Learning?

The authors make a strong case for why RL is a good fit for this problem:

- **Sequential decisions.** Adjusting a chiller setting now will affect the system
  for the next several minutes or hours, not just the current moment. RL is built
  for exactly this kind of problem where actions have long-term consequences.

- **Clear reward signal.** Energy consumption is measured every 5 minutes, which
  makes it easy to define what "doing well" means for the agent.

- **No need for a physics model.** Methods like Model Predictive Control require
  a detailed mathematical model of each building, which takes a lot of time and
  expertise to build. RL can learn directly from real sensor data instead.

- **Too complex for manual rules.** The right chiller settings depend on dozens
  of variables all interacting with each other, like outside temperature, building
  load, and how many chillers are running. It is very hard to write rules that
  handle all of these combinations well.

## 3. State Space

The state is everything the agent can observe about the current situation. In this
case, the agent reads from the building's sensors every 5 minutes. The raw data
contains 176 different sensor measurements, including things like water temperatures,
flow rates, and equipment status (whether certain machines are on or off).

The authors did not use all 176 sensors. Using feature engineering, where 
they worked closely with HVAC experts to figure out which sensors actually matter,
they narrowed it down to 50 relevant measurements. This was important because using
too many irrelevant inputs makes it harder for the model to learn what actually affects
energy consumption.

Some examples of what the state includes:
- Water temperatures at different points in the system
- Flow rates of the pumps
- Outside weather conditions (temperature, humidity)
- Current building load (how much cooling is needed)
- Equipment status (which chillers and pumps are running)

---

## 4. Action Space

The actions are the controls the agent has available to it at each timestep. The
agent outputs a 12-dimensional vector, meaning it is adjusting 12 different things
at once. Some of these are continuous (a specific temperature or flow rate value)
and some are discrete (turning a piece of equipment on or off).

The full list of what the agent can control includes:

- The temperature setpoint for each chiller (how cold the water should be)
- How many chillers to run at the same time
- The cooling tower temperature setpoint
- The flow rate and number of condenser water pumps
- The chilled water differential pressure
- The number of chilled water pumps
- Whether to use mechanical cooling or free cooling (a more passive mode used
  when outside temperatures are cold enough)

One important detail is that the agent does not have unlimited freedom here. There
are 59 constraints on the actions and 24 constraints on what the sensors are allowed
to read, all defined to protect the equipment and keep people comfortable. The
agent has to find the best action within all of these boundaries.

## 5. Reward Function

The reward function is straightforward. At every 5 minute timestep, the agent
receives a reward equal to the negative of the total energy consumed by the
chiller plant during that period. So the more energy used, the more negative
the reward. The agent's goal is to maximize its total reward over time, which
means minimizing energy consumption.

This is a clean and honest reward signal because it directly measures what we
actually care about. There is no need to approximate or engineer a proxy metric,
the energy meter tells you exactly how well the agent is doing.

The agent also has to satisfy a set of constraints alongside this reward. These
are not part of the reward itself but act as hard boundaries the agent cannot
cross, for example keeping the water temperature within a safe range for occupants
and equipment. This makes the problem a constrained optimization rather than a
simple reward maximization.

---

## 6. RL Model and Algorithm (BCOOLER)

The algorithm developed for this project is called BCOOLER, which stands for
BVE-based Constrained Optimization Learner with Ensemble Regularization. It was
built specifically for this problem because no off-the-shelf RL algorithm was
able to handle all the real-world constraints and data limitations they faced.

**How it works at a high level:**

At every 5 minute timestep, the agent goes through the following process:

1. It reads the current state from the 50 sensors
2. It generates around 100,000 possible actions by sampling different combinations
   of setpoints and equipment settings
3. It filters out any actions that are predicted to violate the safety constraints
4. It scores the remaining actions using the Q-function and picks the best one

**The Q-function:**

The Q-function is the core of the agent's decision making. It takes a (state, action)
pair as input and predicts how much energy will be consumed over the next 15 minutes
if that action is taken. The agent uses this to compare all the candidate actions and
pick the one predicted to use the least energy.

The Q-function is represented by a neural network. Specifically, it uses a
multi-tower multi-headed architecture, where different parts of the network are
responsible for predicting energy consumption and each of the 24 observation
constraints separately. This was important because different predictions needed
different sets of input features to work well.

**The ensemble:**

Instead of training a single neural network, the authors trained an ensemble of
10 neural networks with the same architecture but different random starting points.
This gives 10 different predictions for the same input. The key insight is that
if all 10 networks agree, the agent can be confident. If they disagree a lot, it
means the agent is in unfamiliar territory.

This disagreement (measured as standard deviation across the 10 predictions) is
used to control the exploration-exploitation tradeoff:

- During normal operation, the agent picks the action with the best predicted
  energy saving, but penalizes actions where the ensemble disagrees a lot. This
  makes the agent cautious in unfamiliar situations.
- 5% of the time, the agent deliberately explores by picking actions where the
  ensemble disagrees, in order to gather new data and improve the model.

**Offline training with daily updates:**

The agent is not trained in a simulator. It learns directly from real data collected
by the building's sensors. It starts by training on historical data from the old
rule-based controller, then updates its model every day using the new data it
collected while in control. This means the agent gets better over time as it
sees more of the building's behavior.

## 7. Results

The system was tested on two real commercial buildings over a period of 3 months
each, using an A/B testing setup where the RL agent and the old rule-based controller
alternated control day by day. This allowed a fair comparison under similar weather
and load conditions.

The results were:

- *Building 1 (university campus):* 9% energy savings compared to the baseline
- *Building 2 (mixed-use commercial building):* 13% energy savings compared to
  the baseline

Both buildings maintained the same comfort levels for occupants throughout the
experiment, meaning the agent did not sacrifice comfort to save energy.

A few interesting patterns came out of the results:

- The agent performed better in cooler weather and at lower building loads. When
  temperatures were high and the building needed maximum cooling, the equipment
  was running close to its limits and there was less room for the agent to be clever
  about its decisions.

- Performance improved over time as the agent collected more data and updated its
  model. This is expected behavior for an agent that retrains daily.

- The agent discovered non-obvious strategies that the rule-based controller never
  used. For example, it learned to set the condenser water temperature lower than
  the baseline in certain situations. This made the cooling towers work harder but
  allowed the chillers to run more efficiently, resulting in lower total energy use.

- The agent also learned to account for sensors that had drifted out of calibration,
  internally adjusting its behavior to compensate for the measurement errors.

It is worth noting that before the experiment started, the existing rule-based
controller was tuned and improved as part of making the facility "AI ready". This
means the baseline the agent was compared against was already better than average,
so the real improvement over a typical untuned system would likely be even higher
than 9-13%.

---

## 8. Challenges

One of the most valuable parts of this paper is how honest the authors are about
the difficulties they faced. Deploying RL in a real physical system is very
different from training an agent in a simulator.

The main challenges they encountered were:

- *Limited data.* The agent had no simulator to train in, so it could only learn
  from real building data collected at 5 minute intervals. Each day only produces
  around 300 data points, which is very little for training a neural network.

- *Noisy and drifting sensors.* Temperature sensors in real buildings drift over
  time and sometimes get recalibrated suddenly, causing large jumps in the data.
  The agent had to be robust to this kind of noise.

- *Complex safety constraints.* The facility managers knew what safe operation
  looked like in practice but had never had to express it mathematically before.
  Translating their knowledge into formal constraints that the agent could use
  took significant back and forth with the HVAC experts at Trane.

- *Real time decisions.* The agent had to make a decision within 1 minute every
  5 minutes. With 100,000 candidate actions to score through a neural network
  ensemble, this required careful engineering to make fast enough.

- *Non-stationary environment.* The building behaves differently across seasons,
  occupancy patterns change, and equipment performance degrades over time. The
  agent had to continuously adapt rather than relying on a fixed model.

---

## 9. Our Assessment

This project is a strong example of what it takes to apply RL successfully outside
of a controlled research environment. The core RL concepts are relatively standard,
the Q-function, ensemble networks, and exploration-exploitation tradeoff are all
well established ideas. What makes this work interesting is the engineering effort
required to make them work reliably on a real physical system with safety constraints,
noisy data, and no simulator.

The results are meaningful. A 9-13% reduction in energy consumption across two
different types of commercial buildings, while maintaining occupant comfort, shows
that the approach generalizes and is not just tuned to one specific building.

From an RL perspective, the most notable design choice is the action search approach.
Rather than using a policy network that directly outputs an action, the agent generates
thousands of candidate actions and scores them with the Q-function. This was necessary
because the complex safety constraints made it very hard to build a policy network
whose outputs would always be safe. It is a practical and honest tradeoff between
theoretical elegance and real-world reliability.

The biggest limitation of the paper is that the authors could not run proper ablation
studies, meaning they could not isolate and measure the contribution of each individual
design choice. This is a direct consequence of deploying on a live system where you
cannot run two versions of the agent at the same time. It is an honest admission but
it does make it harder to know which parts of BCOOLER actually matter most.
