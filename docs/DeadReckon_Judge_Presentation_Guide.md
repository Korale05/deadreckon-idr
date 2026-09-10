# DeadReckon — Simple Explanation Guide for Judges

This document explains our project in the simplest possible words, followed by the questions judges are most likely to ask, with easy answers ready to speak out loud.

---

## PART 1: What Problem Are We Solving?

When you drive through a tunnel, an underground parking lot, or between tall buildings, your phone loses GPS signal. Google Maps either freezes or guesses wrong.

Phones have two hidden sensors that work even without GPS:
- **Accelerometer** – feels when the car speeds up or slows down
- **Gyroscope** – feels when the car turns

Using just these two sensors, a phone can try to guess its own position without GPS. This is called **Dead Reckoning**.

**The Problem:** These sensors are cheap and slightly imperfect. Every tiny mistake adds up over time. After driving for just one hour with no GPS, a phone using only these sensors can think it is **11 kilometers away from where it actually is**. That is completely useless.

**Our Solution:** We built a small, fast AI that watches these sensors and constantly fixes the mistakes, 10 times every second — so the phone stays accurate even with zero GPS.

---

## PART 2: Where Did We Get Our Data From?

We used a public dataset called **IO-VNBD**, made from real cars driving in the UK, France, and Nigeria.

For every drive, two things were recorded at the same time:

| What | Recorded By | Purpose |
|---|---|---|
| Phone sensor data (speed guesses, shaking, turning) | The smartphone | This is what our AI looks at |
| The car's true, exact speed and direction | The car's own computer | This is the "correct answer" we compare against |

By comparing "what the phone guessed" against "what actually happened," we can teach the AI exactly how wrong the phone usually is — and by how much.

---

## PART 3: What Does the AI Actually Look At? (Input)

Every 0.1 seconds, the AI is shown:

1. **The last 2 seconds of sensor shaking and turning** — like a tiny replay clip of how the car has been moving.
2. **The phone's current speed guess** — what the basic physics calculation currently thinks the speed is.

That's it. Two simple pieces of information.

---

## PART 4: What Does the AI Give Back? (Output)

The AI gives back just **2 numbers**:

- How much the phone's speed guess is wrong, sideways
- How much the phone's speed guess is wrong, forward/backward

Then we simply **add** this correction to the phone's guess to get a much better speed estimate.

```
Better Speed = Phone's Rough Guess + AI's Correction
```

No complicated steps — just one simple addition.

---

## PART 5: How Is the AI Built? (In Simple Words)

Think of the AI like a small team of workers, each doing one simple job:

1. **Pattern Spotters** – look at the last 2 seconds of shaking and spot things like bumps, brakes, or turns.
2. **Stabilizers** – keep all the numbers in a normal, safe range so nothing goes wild.
3. **Shrinker** – compresses the information down so it's faster to process.
4. **Memory Workers** – remember how the mistake has been building up over the last 2 seconds, not just right now.
5. **Forgetful Trainer (Dropout)** – during training only, randomly hides some information on purpose, so the AI doesn't cheat by memorizing — this makes it work well on roads it has never seen.
6. **Final Decision Makers** – take everything the team learned and turn it into the final 2 correction numbers.

**Size:** The whole AI fits in under 250 KB — smaller than a single photo on your phone.
**Speed:** It finishes thinking in about 3 milliseconds — faster than the blink of an eye, 10 times every second.

---

## PART 6: How Was It Trained?

- We showed the AI thousands of short 2-second clips from real drives, along with the correct answer for each clip.
- We used a grading method called **Huber Loss**, which is fair — small mistakes are graded normally, but rare big shocks (like hitting a pothole) don't confuse the AI's learning too much.
- We made sure the AI was tested on a full drive it had **never seen before** during training — so we know it's not just memorizing, it's actually learning the pattern.

---

## PART 7: How Well Does It Work? (Real Numbers, Explained Simply)

We tested this on a real 1.4-hour, 38-kilometer drive the AI had never seen before.

| Method | How Far Off At The End | In Simple Words |
|---|---|---|
| **Phone sensors only, no AI, no GPS** | About 11 kilometers off | Basically useless — like ending up in the wrong city |
| **Phone sensors + our AI, still no GPS** | About 2.3 kilometers off (under 6% error) | Big improvement — stays roughly on the right road |
| **Full system with GPS turned on** | About 1 meter off | As accurate as parking in the correct parking spot |

**In short: our AI alone cuts the error by about 80%, even with zero GPS.**

---

## PART 8: How Does This Fit Together With GPS?

We use a second system called an **EKF** (a trusted method used in navigation for over 50 years, even used in space missions). Think of it as a referee that listens to three helpers and decides how much to trust each one:

1. **Basic physics guess** – always available, but drifts over time
2. **Our AI's corrected guess** – much more reliable, especially useful with no GPS
3. **GPS** – very accurate when available, but sometimes missing

When GPS disappears (tunnel), the referee leans on our AI instead. When GPS comes back, it smoothly switches back — without any sudden jump on the screen.

---

## PART 9: One-Sentence Summary

*"Our AI watches how a car shakes, brakes, and turns, and uses that to constantly fix a phone's built-in guess of its own speed — so that even with zero GPS, like inside a tunnel, the phone still knows roughly where it is, instead of getting lost by kilometers."*

---

# PART 10: Questions Judges Will Likely Ask (With Easy Answers)

### Q1: Why do we need this? Doesn't Google Maps already work?
**A:** Google Maps needs GPS or internet to work well. Inside tunnels, parking garages, or between tall buildings, GPS signals get blocked, and Google Maps either freezes or shows the wrong location. Our system works completely offline using only the phone's built-in motion sensors, and stays accurate even with zero GPS.

### Q2: Why use AI instead of just doing the physics math?
**A:** Physics math alone drifts very badly — 11 kilometers off after just one hour. Sensor mistakes aren't simple or predictable; they change depending on the road, the car, and even the temperature. AI is good at learning these messy, complicated patterns that plain math cannot capture.

### Q3: Why is your AI so small? Why not use a bigger, more powerful AI?
**A:** Our AI needs to run instantly, 10 times every second, directly on a phone — without draining the battery or slowing the phone down. A bigger AI would be slower and would use much more battery, for very little extra benefit. Our small AI is fast enough to run in about 3 milliseconds and small enough to fit in under 250 KB.

### Q4: How do you know the AI isn't just memorizing the test drive?
**A:** We made sure the exact drive we test on was **completely removed** from training. The AI never sees that drive until the final test. So its performance shows real learning, not memorization.

### Q5: What happens if someone opens the app for the first time inside a tunnel, with no GPS at all?
**A:** The app still works. It starts tracking movement relative to wherever the phone currently is. As soon as the phone gets even one GPS signal — like when exiting the tunnel — it instantly snaps the whole recorded path onto the correct real-world location.

### Q6: Does this drain the phone's battery?
**A:** No. When GPS signal is strong and reliable, our AI actually goes to sleep and uses almost no battery, since GPS alone is accurate enough at that point. The AI only wakes up and works hard when GPS gets weak or disappears.

### Q7: Does this need an internet connection?
**A:** No. Everything — the sensors, the AI, and the math that combines everything together — runs completely offline, directly on the phone.

### Q8: What real numbers can you show us to prove this works?
**A:** On a real 38-kilometer, 1.4-hour drive with zero GPS the whole time:
- Plain physics guessing was off by about 11 kilometers (29% error).
- Adding our AI cut that down to about 2.3 kilometers (under 6% error) — roughly an 80% improvement.
- With GPS turned on and combined with our AI, the final result was accurate to about 1 meter — roughly the size of a parking space.

---

*This guide is written in simple, everyday language so it can be explained confidently to any judge, regardless of their technical background.*
