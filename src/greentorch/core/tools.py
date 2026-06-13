from matplotlib import pyplot as plt

def print_profiling(profiler_values: list):
    for gpu in profiler_values:
        keys = profiler_values[gpu]["key"]
        power = profiler_values[gpu]["power"]
        frequencies = profiler_values[gpu]["max_frequency"]

        plt.plot(frequencies, [keys[k] / power[k] for k in range(0, len(keys))])
        plt.xlabel("Frequency in Mhz")
        plt.ylabel("Key/Power Usage")
        plt.grid(True)
        plt.show()