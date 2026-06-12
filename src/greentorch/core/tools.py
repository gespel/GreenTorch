from matplotlib import pyplot as plt

def print_profiling(profiler_values: list):
    out = {}
    for measurement in profiler_values:
        for gpu_id in measurement:
            if gpu_id in out:
                out[gpu_id]["measurements"].append((measurement[gpu_id]["max_frequency"], measurement[gpu_id]["key"], measurement[gpu_id]["power"]))
            else:
                out[gpu_id] = {
                    "measurements": []
                }
    for gpu in out:
        keys = []
        power = []
        frequencies = []
        for measurement in out[gpu]["measurements"]:
            keys.append(measurement[1])
            power.append(measurement[2])
            frequencies.append(measurement[0])
        plt.plot(frequencies, [keys[k] / power[k] for k in range(0, len(keys))])
        plt.xlabel("Frequency in Mhz")
        plt.ylabel("Key/Power Usage")
        plt.grid(True)
        plt.show()