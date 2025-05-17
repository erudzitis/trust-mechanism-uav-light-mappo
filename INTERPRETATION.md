Metric Interpretation Guide for MAPPO Learning

Value Loss
Good Learning: Starts high (0.5-2.0), decreases consistently, eventually converging to near zero (<0.01)
Poor Learning: Stays high, fluctuates wildly, or decreases but plateaus at a high value

Policy Loss
Good Learning: Maintains negative values (typically between -0.01 and -0.2), may fluctuate but shouldn't trend toward zero
Poor Learning: Becomes very negative (below -1.0) or approaches zero/positive values

Distribution Entropy
Good Learning: Starts high (3.0-5.0), gradually decreases but maintains moderate values (1.0-3.0) even late in training
Poor Learning: Drops too quickly (suggests premature convergence) or stays too high (suggests no learning)

Policy Ratio
Good Learning: Stays close to 1.0, typically between 0.8-1.2, with occasional excursions but always returning
Poor Learning: Consistently reaches the clip bounds (often 0.8 or 1.2) or wildly fluctuates

Actor Gradient Norm
Good Learning: May start high (2.0-6.0) and show some decline, but often maintains moderate values (1.0-3.0)
Poor Learning: Grows over time, becomes extremely small (approaching 0) too early, or shows extreme spikes

Critic Gradient Norm
Good Learning: Starts high (2.0-5.0) and consistently decreases, eventually reaching low values (<0.5)
Poor Learning: Fails to decrease, or decreases but then increases again