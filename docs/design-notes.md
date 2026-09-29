# DPO in plain English

## The setup

You have a prompt x and two answers: one people preferred (y_w, the winner)
and one they didn't (y_l, the loser). You also have a reference model, usually
the model before tuning. DPO nudges the policy toward the winner and away from
the loser, without ever training a separate reward model.

## The trick

DPO defines an *implicit reward* for any answer:

    r(x, y) = beta * log( pi(y|x) / pi_ref(y|x) )

In words: how much more (or less) likely is this answer under the tuned model
than under the reference model, scaled by beta. If tuning made the answer more
likely, the reward is positive.

Then it assumes the Bradley-Terry model of human preference: the chance that a
person prefers y_w over y_l is sigmoid(r_w - r_l). Training minimizes the
negative log of that:

    loss = -log( sigmoid( beta * (logratio_w - logratio_l) ) )

where logratio_w = log pi(y_w) - log pi_ref(y_w), and same for the loser.

So the loss only cares about the *margin* between winner and loser implicit
rewards. Big positive margin, small loss. That's the whole algorithm.

## What beta does

Beta is the leash length. Small beta lets the policy drift far from the
reference model to chase the preference signal. Large beta keeps it close.

Too small: the model overfits the 10k pairs. It learns quirks of the dataset
instead of better answers. Watch the KL to the reference model explode.

Too large: nothing happens. The margin barely moves and the evals look flat.

The sweep {0.05, 0.1, 0.2} exists to watch both failure modes with your own
eyes. That observation is the actual lesson of this project.

## Why the reference model matters

The reference model anchors everything. The implicit reward is *relative* to
it. If you swap the reference model mid-project, old margins and new margins
are not comparable. Freeze it, record its exact id, and never touch it.

## Failure modes to watch for

**Length bias.** In most preference datasets the chosen answer is longer.
DPO can learn "longer is better" instead of "better is better". That's why the
generation eval reports length-controlled win rate.

**Both answers are bad.** DPO only pushes the margin. If the winner is also a
bad answer, the model learns to prefer a bad answer more strongly. Read random
pairs by hand during data prep so you know what you're actually teaching.

**Margin goes up, quality doesn't.** The held-out margin can improve while
generations get worse (reward hacking at tiny scale). That's why there are
three evals, not one. If margin accuracy rises but the blinded win rate is
flat, say so in RUN_NOTES.md. That's a real result.

**KL explosion.** Mean log(pi/pi_ref) climbing fast means the policy is
running away from the reference. Usually beta is too small or the learning
rate is too high. Stop the run. Don't hope it recovers.

## What DPO is not

DPO is not RL. There are no rollouts, no exploration, no reward model to
over-optimize against. It's supervised learning on preference pairs wearing a
clever loss function. Labs do plenty of actual RL (PPO, RLVR, online loops).
This project teaches you the preference-tuning half of post-training, honestly
scoped. The other half is a different project.
