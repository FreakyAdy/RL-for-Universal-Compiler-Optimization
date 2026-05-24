# Hardware setup

## x86-64 (RAPL + perf)

1. Load `msr` module (if needed): `sudo modprobe msr`
2. Ensure RAPL readable: `sudo chmod +r /sys/class/powercap/intel-rapl:0/energy_uj`
3. Set `perf` paranoid: `sudo sysctl -w kernel.perf_event_paranoid=1`
4. For stable rewards during collection, pin frequency:
   ```bash
   sudo cpupower frequency-set -g powersave
   sudo cpupower frequency-set -d 2.0GHz -u 2.0GHz
   ```
5. Document turbo-on vs turbo-off in experiment logs.

## ARM64

Energy APIs vary by platform. `rl_uco/hardware/cpu_arm.py` uses:

- `perf` for cycles/time where available
- Optional `spe` or sysfs energy interfaces when present
- Falls back to time-only reward with `energy_j = 0` and warning

## NVIDIA CUDA

- Install driver + CUDA toolkit
- `pip install pynvml`
- Set `CUDA_VISIBLE_DEVICES` for collector workers

## Windows

Native RAPL access is limited. Use Docker/Linux VM for data collection; development and training can run on Windows with cached datasets.
