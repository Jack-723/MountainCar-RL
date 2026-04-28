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

## 6. RL Model & Algorithm (BCOOLER)

## 7. Results

## 8. Challenges

## 9. Our Assessment
