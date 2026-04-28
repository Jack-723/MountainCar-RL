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

## 4. Action Space

## 5. Reward Function

## 6. RL Model & Algorithm (BCOOLER)

## 7. Results

## 8. Challenges

## 9. Our Assessment