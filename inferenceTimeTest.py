from time import sleep
import torch



starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# MEASURE PERFORMANCE

starter.record()
sleep(10)
ender.record()

torch.cuda.synchronize()
curr_time = starter.elapsed_time(ender)

print(f'Sleep time: {curr_time}')
