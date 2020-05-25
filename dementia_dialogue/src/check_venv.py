import sys

print(sys.prefix)
print(sys.exec_prefix)

print(sys.base_prefix)
print(sys.base_exec_prefix)

import torch
print(torch.cuda.is_available())
