from matplotlib import pyplot as plt
import statistics

def print_profiling(profiler_all_values: list):
    for gpu in profiler_all_values[0]:
        keys = []
        powers = []
        frequencies = []

        for measurement_index in range(0, len(profiler_all_values[0][gpu]["key"])):
            key_list = [profiler_all_values[i][gpu]["key"][measurement_index] for i in range(0, len(profiler_all_values))]
            power_list = [profiler_all_values[i][gpu]["power"][measurement_index] for i in range(0, len(profiler_all_values))]
            frequency_list = [profiler_all_values[i][gpu]["max_frequency"][measurement_index] for i in range(0, len(profiler_all_values))]
            key = statistics.fmean(key_list)
            power = statistics.fmean(power_list)
            frequency = statistics.fmean(frequency_list)
            keys.append(key)
            powers.append(power)
            frequencies.append(frequency)

        plt.plot(frequencies, [keys[k] / powers[k] for k in range(0, len(keys))])
        plt.xlabel("Frequency in Mhz")
        plt.ylabel("Efficiency (Iterations per Second/Power Usage)")
        plt.title("Frequency Efficiency for the AMD RX 6900 XT")
        plt.grid(True)
        plt.show()