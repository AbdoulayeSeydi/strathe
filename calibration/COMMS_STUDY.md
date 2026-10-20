# Test your fleet's coordination under acoustic limits: `strathe comms-study`

A fleet that coordinates in a perfect simulation can fall apart at sea, because everything one vehicle knows
about another comes through an acoustic modem. The modem is slow, delayed, shared and lossy. `strathe comms-study`
answers the question a team needs before a field trial:

> With our modem, our number of vehicles and our coordination strategy, how much latency, bandwidth loss and
> packet loss can the fleet take before coordination breaks?

It runs a formation task on Strathe's calibrated vehicle physics, with every vehicle knowing its teammates only
through a simulated shared acoustic channel. It changes one channel setting at a time and reports where
coordination holds and where it breaks, with 95% intervals.

```bash
python strathe.py comms-study --out my_study                       # the default study, about 10 minutes on 8 cores
python strathe.py comms-study --modem slow --fleet-size 6 --out s6  # your modem class and fleet size
python strathe.py comms-study --sweep latency loss --values loss=0,0.2,0.4,0.6 --out lat_loss
python strathe.py comms-study --controller my_formation:MyController --out mine   # your own strategy
python strathe.py comms-study --help                                # every option
```

The web page has the same thing as a form: **Test your fleet's coordination under acoustic limits**.

## 1. What is simulated

**The task.**
- **The leader:** vehicle 0 runs a survey at `--speed` (default 0.2 m/s). Once in each 60 s leg window it turns
  by ±`--turn` degrees (default 60), at a random time the others can't know in advance.
- **The followers:** every other vehicle holds its slot in a formation (`--formation wedge | line | column`,
  `--spacing`, or your own `--offsets`).
- **What a follower knows:** where the leader and the others are comes only from their acoustic beacons. So the
  formation holds exactly as well as the channel lets it.
- **Formation orientation:**
  - by default the formation keeps its compass orientation, so each slot moves at the leader's speed;
  - `--formation-orientation leader` makes it turn with the leader. Strathe refuses that when the turn would need
    a slot to move faster than the vehicle can.

**The vehicles.** All of them run the same calibrated 6-DOF physics: by default the shipped BlueROV2, or your own
calibration with `--params` and `--vehicle-spec`. Thrust is limited to ±30 N and torque to ±5 N·m.

**The channel (`acoustic_network.py`).** One shared medium, as at sea:

| Effect | How it's modelled | Set it with |
|---|---|---|
| **Travel time** | range ÷ 1500 m/s, fixed when the packet leaves | `--spacing` (and the formation) |
| **Time on the air** | preamble + 8 × bytes ÷ bit rate. A 32-byte beacon at 80 bit/s is on the air for 3.2 s plus the preamble | `--bit-rate`, `--packet-overhead`, `--beacon-bytes` |
| **Modem delay** | decoding time after the packet arrives | `--latency` |
| **Sharing the water** | `tdma`: each vehicle owns a slot in a repeating cycle, slot = time on the air + a guard for the longest travel time. `aloha`: send when a beacon is due (with random jitter), and overlapping packets at a receiver destroy each other | `--mac tdma` or `--mac aloha` |
| **Half duplex** | a modem can't hear while it transmits | always on |
| **Packet loss** | per receiver: independent, or in bursts (a two-state Gilbert-Elliott chain with that mean burst length) | `--loss`, `--burst` |
| **Range** | beyond 3000 m nothing arrives | `--max-range` |

**Modem classes** (`--modem`) give a starting point when you haven't measured yours. They are generic classes,
not specific products: check your modem's data sheet and override the numbers, or measure the link with
`strathe fit-link`.

| Class | Bit rate | Preamble | Decoding delay |
|---|---|---|---|
| `slow` | 80 bit/s | 1.0 s | 0.5 s |
| `medium` (default) | 1000 bit/s | 0.5 s | 0.3 s |
| `fast` | 5000 bit/s | 0.25 s | 0.2 s |
| `ideal` | unlimited | 0 | 0 |

**Bandwidth is time.** With TDMA every vehicle waits for its slot, so the cycle (vehicles × (time on the air +
guard)) is the real update interval once it is longer than the beacon interval. That's how more vehicles or a
slower modem make every teammate's information older. With random access, collisions climb fast as the channel gets
busy: textbook unslotted random access (ALOHA) carries at most about 18% of the channel's capacity.

**Between beacons**, each vehicle predicts its teammates (`--belief`):
- `last_beacon`: the teammate stays where it last said it was;
- `constant_velocity` (default): it keeps its reported velocity;
- `physics`: it is dead-reckoned with the vehicle's calibrated model and its last reported thrust. `--belief-params`
  gives it a different, e.g. uncalibrated, model;
- `oracle`: the truth, as an upper bound.

## 2. What the study changes

From the base setting (your flags), one setting at a time. Defaults are in brackets, from easy to hard:

| Setting | What changes | Default values |
|---|---|---|
| `latency` | modem decoding delay, s | 0, 1, 2, 5, 10, 20 |
| `bit_rate` | bit/s | unlimited, 5000, 1000, 300, 100, 50 |
| `loss` | independent packet loss | 0, 10, 20, 30, 50, 70% |
| `burst` | the same loss in bursts of this many packets (30% if the base has no loss) | independent, 2, 4, 8, 16 |
| `fleet_size` | vehicles sharing the channel | 2, 3, 4, 6, 8 |
| `beacon_interval` | seconds between each vehicle's beacons | 1, 2, 5, 10, 20 |
| `spacing` | formation spacing, m (not in the default sweep) | 5, 10, 20, 50, 100 |
| `mac` | how the channel is shared | tdma, aloha |
| `belief` | teammate prediction | oracle, physics, constant_velocity, last_beacon |
| `controller` | the coordination strategy (with your `--controller`) | leader_follower and yours |

- **Choose settings** with `--sweep latency loss ...` and **values** with `--values latency=0,3,6,12`
  (`inf` = unlimited bit rate).
- **Every value runs the same episodes:** same seeds, same leader turns, same starts. So differences come from the
  setting, not luck.
- **Episodes:** `--episodes` (default 30). Fewer runs faster and gives wider intervals.
- **A first look:** `--quick` runs every other default value.

## 3. What you get back

In `--out`:
- **`report.md`:**
  - one line per setting: where coordination holds (success at least 80%) and where it breaks, with the 95%
    interval;
  - then a table and a plot per setting.
- **`summary.csv`:** one row per value. It has:
  - success rate and its interval;
  - formation error (mean and interval, median);
  - closest approach and collisions;
  - belief error and beacon age;
  - delivery ratio, how busy the channel was, time on the air and the TDMA cycle.
- **`episodes.csv`:** one row per episode, including every packet's fate: `n_delivered`, `n_collision`,
  `n_half_duplex`, `n_lost`, `n_out_of_range`, `n_no_slot` (no TDMA slot before a newer beacon) and
  `n_superseded` (replaced while waiting for the modem).
- **`plots/`:** success and formation error against each setting.
- **`study.yaml`:** the exact command, base setting and seeds, to repeat it.

**Success** means:
- the formation error stays under `--tolerance` (default 3 m RMS; each follower's distance from where its slot
  really is, after 30 s);
- and no two vehicles come closer than 1.5 m.

## 4. Your own coordination strategy

Write a class with an `act` method and pass it as `module:Class`:

```python
# my_formation.py
import numpy as np

class MyController:
    def __init__(self, cfg, plan):        # optional: the study's CoordinationConfig, and the leader's plan
        self.cfg, self.plan = cfg, plan   # (only the leader may use the plan: the others can't know it)

    def act(self, i, own, teammates, t):
        """i: this vehicle (0 = the leader). own: its own state (.position, .velocity in world north-east-down,
        .R body->world, .nu body velocities). teammates: {id: what THIS vehicle believes about it, or None before
        its first beacon}, each with .position, .velocity, .heading (rad), .age_s (how old the newest beacon is).
        t: seconds. Returns 6 numbers: body-frame force (N) and torque (N*m): forward, starboard, down, roll,
        pitch, yaw."""
        ...
        return np.zeros(6)
```

```bash
python strathe.py comms-study --controller my_formation:MyController --sweep controller latency loss --out mine
```

- **Where your module is found:** the folder you run from. Use `--controller-path` for another one.
- **What `--sweep controller` does:** runs the built-in leader-follower and yours on the same episodes, side by side.
- **What your controller sees:** only beliefs, never the truth. The score is always the truth.

The built-in `leader_follower` (in `fleet_coordination.py`) is a reasonable baseline, not a tuned product:
- each follower steers to the believed leader position plus its slot, using the believed leader velocity as a
  feed-forward;
- it pushes away from any teammate believed closer than 4 m;
- it is a PID with thrust saturation.

## 5. Reading the results honestly

- **The channel model is generic.** It has the effects that break coordination (latency, airtime, collisions, half
  duplex, bursts) at textbook values. For an answer specific to your lab:
  - set your modem's bit rate, preamble and delay;
  - measure loss against range with `strathe fit-link`;
  - check the burst length in your logs.
- **Not modelled:**
  - multipath geometry, Doppler and the capture effect;
  - TDMA clock drift, carrier sensing, acknowledgements and retransmission, and relaying;
  - navigation error: vehicles know their own positions exactly here; `strathe fleet` has sensor noise for two
    vehicles.

  See LIMITS.md.
- **The thresholds are about this task.** "Breaks at 10 s of delay" is for this leader speed, these turns and this
  tolerance. A faster leader or a tighter tolerance breaks sooner. Set them to your mission's (`--speed`, `--turn`,
  `--tolerance`, `--spacing`).
- **Two-vehicle experiments with trained policies** stay in `strathe fleet` (CUSTOM_EXPERIMENT.md). `comms-study` is
  for N vehicles and your own strategies.
