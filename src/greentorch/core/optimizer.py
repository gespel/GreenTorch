from termcolor import colored

class Optimizer:
    def __init__(self, logger):
        self.logger = logger


    def calc_optimize_frequency(self, gpu_id: str, curr_power: float) -> float:
        raise NotImplementedError


class SimpleDirectionalOptimizer(Optimizer):
    def __init__(self, logger):
        super().__init__(logger)
        self.epsilon_ema = 0
        self.epsilon_alpha = 0.3
        self.last_epsilon = None
        self.direction = -1
        self.epsilon_tolerance = 0.005
        self.step_mhz = 50


    def calc_optimize_frequency(self, curr_frequency: float, curr_key: float, curr_power: float) -> float:
        self.last_freq = curr_frequency
        if curr_frequency <= 0:
            self.logger.info("Could not read current GPU frequency; skipping adjustment.")
            return curr_frequency

        if curr_power <= 0:
            self.logger.info("Power is non-positive; keeping current frequency.")
            return curr_frequency

        epsilon_raw = curr_key / curr_power
        if self.epsilon_ema is None:
            epsilon = epsilon_raw
            self.epsilon_ema = epsilon_raw
        else:
            self.epsilon_ema = (self.epsilon_alpha * epsilon_raw) + ((1.0 - self.epsilon_alpha) * self.epsilon_ema)
            epsilon = self.epsilon_ema

        if self.last_epsilon is None:
            self.last_epsilon = epsilon
            self.logger.debug(
                f"Baseline ε initialized to {epsilon:0.4f} (raw {epsilon_raw:0.4f}) at {curr_frequency:0.4f} MHz."
            )
            next_freq = curr_frequency + (self.direction * self.step_mhz)
            return next_freq

        prev = self.last_epsilon

        if prev == 0:
            prev = 1e-12

        rel_change = (epsilon - prev) / abs(prev)

        if rel_change > self.epsilon_tolerance:
            self.logger.info(
                f"ε improved {prev:0.4f} -> {epsilon:0.4f} (raw {epsilon_raw:0.4f}, Δ={rel_change:+.2%}). "
                f"{colored('CONTINUE', 'green')} direction {self.direction:+d}"
            )
        elif rel_change < -self.epsilon_tolerance:
            self.direction *= -1
            self.logger.info(
                f"ε worsened {prev:0.4f} -> {epsilon:0.4f} (raw {epsilon_raw:0.4f}, Δ={rel_change:+.2%}). "
                f"{colored('REVERSE', 'red')} direction to {self.direction:+d}"
            )
        else:
            self.logger.info(
                f"ε change within deadband {prev:0.4f} -> {epsilon:0.4f} (raw {epsilon_raw:0.4f}, Δ={rel_change:+.2%}). "
                f"{colored('HOLD', 'yellow')} direction to {self.direction:+d}"
            )

        self.last_epsilon = epsilon
        next_freq = curr_frequency + (self.direction * self.step_mhz)
        return next_freq
