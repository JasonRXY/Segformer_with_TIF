import psutil

# 获取 CPU 核心数
cpu_count = psutil.cpu_count(logical=False)  # 物理核心数
logical_cpu_count = psutil.cpu_count(logical=True)  # 逻辑核心数，包括超线程

# 获取内存信息
memory_info = psutil.virtual_memory()

print(f"Physical CPU cores: {cpu_count}")
print(f"Logical CPU cores: {logical_cpu_count}")
print(f"Total memory: {memory_info.total / (1024 ** 3):.2f} GB")
print(f"Available memory: {memory_info.available / (1024 ** 3):.2f} GB")
