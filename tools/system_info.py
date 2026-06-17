import psutil


def get_system_info() -> str:
    try:
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        disk = psutil.disk_usage("C:\\")
        return (
            f"CPU: {cpu}%, "
            f"RAM: {used_gb:.1f}GB / {total_gb:.1f}GB ({mem.percent}% used), "
            f"Disk C: {disk.percent}% full"
        )
    except Exception as e:
        return f"Could not get system info: {e}"
