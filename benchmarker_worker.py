import time
import psutil
import multiprocessing
import numpy as np
import math
import gc
import platform
import array
import random 

from PySide6.QtCore import QObject, Signal, QTimer, QThread

# --- GLOBAL WMI SETUP (For Temperature) ---
# WMI is initialized globally. It will be None if wmi is not installed or service fails.
W = None
if platform.system() == "Windows":
    try:
        import wmi
        W = wmi.WMI()
    except ImportError:
        pass
    except Exception:
        pass


# --- DATA STRUCTURES AND HELPER FUNCTIONS ---
def simple_worker_wrapper(args):
    """Wrapper to run fixed-duration test inside a single process."""
    op_func, duration = args
    return run_standard_ops_for_duration(op_func, duration)
# --- Branch Prediction Arrays (Defined Once) ---
# NOTE: Arrays are now created locally inside run_branch_test to prevent memory copy issues.
def calculate_pi_to_n_digits(n):
    """Integer Test: Adjusted for fine timing granularity."""
    # Reduced range for faster single operation loop closure
    return sum(1/i**2 for i in range(1, 1000)) 

def run_matrix_multiplication(size):
    """Floating-Point Test."""
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    C = np.dot(A, B)
    return C.size

def run_standard_ops_for_duration(op_func, duration):
    """General function for running fixed-duration ops."""
    start_time = time.perf_counter()
    ops_count = 0
    while (time.perf_counter() - start_time) < duration:
        op_func()
        ops_count += 1
    return (time.perf_counter() - start_time), ops_count


# --- MEMORY & CACHE BENCHMARK FUNCTIONS ---

def run_memory_bandwidth_test(duration):
    """Measures sequential memory bandwidth (Score = MB/s)."""
    SIZE_MB = 256
    array_size_bytes = int(SIZE_MB * 1024 * 1024)
    num_elements = array_size_bytes // 8 
    A = np.random.rand(num_elements).astype(np.float64)
    B = np.empty_like(A)

    start_time = time.perf_counter()
    ops_count = 0 
    
    while (time.perf_counter() - start_time) < duration:
        np.copyto(B, A) 
        ops_count += 1
        
    total_time = time.perf_counter() - start_time
    total_gb_moved = (ops_count * array_size_bytes * 2) / (1024**3)
    bandwidth_gb_s = total_gb_moved / total_time if total_time > 0 else 0
    
    return total_time, int(bandwidth_gb_s * 1024) 

def run_cache_stride_test(stride_bytes=64):
    """Measures access latency for a single stride (Score = Latency in ns)."""
    SIZE_MB = 64 
    array_size_bytes = int(SIZE_MB * 1024 * 1024)
    data_array = array.array('i', (0 for _ in range(array_size_bytes // 4)))
    
    step = stride_bytes // 4
    length = len(data_array)
    num_iterations = 10 # Reduced iterations for safe, fast measurement
    
    if step == 0 or step >= length:
        return 0.001, 0.0

    start_time = time.perf_counter()
    
    for _ in range(num_iterations):
        for i in range(0, length, step):
            data_array[i] += 1
            
    total_time = time.perf_counter() - start_time
    total_accesses = num_iterations * (length // step)
    latency_ns = (total_time * 1e9) / total_accesses if total_accesses > 0 else 0
    
    return total_time, latency_ns


# --- BRANCH PREDICTION BENCHMARK FUNCTION ---

def run_branch_test(is_predictable):
    """Internal helper to run the core conditional loop, creating data locally."""
    SIZE = 1000000 
    
    if is_predictable:
        arr = np.arange(SIZE).astype(np.int32)
    else:
        arr = np.random.randint(0, 200, size=SIZE).astype(np.int32)
    
    total_sum = 0
    start_time = time.perf_counter()
    
    for x in arr:
        if x < 100:
            total_sum += 1
        else:
            total_sum -= 1
            
    return time.perf_counter() - start_time
    
def calculate_branch_penalty():
    """Measures the misprediction penalty ratio (Score = Ratio * 1000)."""
    num_runs = 5
    time_predictable = sum(run_branch_test(True) for _ in range(num_runs)) / num_runs
    time_unpredictable = sum(run_branch_test(False) for _ in range(num_runs)) / num_runs
    
    penalty_ratio = time_unpredictable / time_predictable if time_predictable > 0 else 0
    
    return time_unpredictable, penalty_ratio * 1000


# --- MULTIPROCESSING INITIALIZER ---
def worker_initializer():
    """Sets the process as a daemon so it terminates when the parent process does."""
    multiprocessing.current_process().daemon = True


# --- MASTER ROUTER FUNCTION ---

def worker_target_function(test_type, duration):
    """The main function that dispatches the job based on the test_type."""
    gc.disable()
    
    # --- Execute Benchmark Logic ---
    if test_type == 'integer':
        # 🔑 FINAL FIX: Wrap the function call in a lambda 
        # to ensure the function reference is cleanly passed and executed immediately, 
        # resolving the hang/stall issue for the pure Python loop.
        total_time, ops_count = run_standard_ops_for_duration(lambda: calculate_pi_to_n_digits(10000), duration)
    
    elif test_type == 'floating':
        # This already uses a lambda, so it's stable.
        total_time, ops_count = run_standard_ops_for_duration(lambda: run_matrix_multiplication(50), duration)
    elif test_type == 'memory':
        total_time, score_value = run_memory_bandwidth_test(duration=duration)
        ops_count = score_value # Score is now MB/s
        
    elif test_type == 'cache':
        total_time, score_value = run_cache_stride_test(stride_bytes=64) 
        ops_count = score_value # Score is now Latency (ns)
        
    elif test_type == 'branch':
        total_time, score_value = calculate_branch_penalty()
        ops_count = score_value # Score is now Penalty Ratio * 1000
    
    else:
        # Default or error case
        total_time = 0
        ops_count = 0
    
    gc.enable()
    return total_time, ops_count

# ------------------------------------------------------------------
# CRITICAL: SYSTEM MONITOR 
# ------------------------------------------------------------------
class SystemMonitor(QObject):
    """Collects real-time system data (CPU, Memory, Temp) and emits it to the GUI."""
    data_ready = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer = QTimer()
        self.timer.setInterval(200) 
        self.timer.timeout.connect(self._check_cpu)
        
        # Get the global WMI object if it was successfully initialized
        self.W = W 

    def start_monitoring(self):
        self.timer.start()

    def stop_monitoring(self):
        self.timer.stop()

    def _check_cpu(self):
        # 1. Get CPU & Memory usage (Reliable)
        cpu_usage = psutil.cpu_percent(interval=None) 
        mem_usage = psutil.virtual_memory().percent 
        
        # 2. Get Temperature using WMI (For Desktops)
        temp_c = 0.0
        
        if platform.system() == "Windows" and self.W is not None: 
            try:
                # Prioritize Win32_TemperatureProbe, Fallback to MSAcpi_ThermalZoneTemperature
                temp_data = self.W.query("SELECT CurrentReading FROM Win32_TemperatureProbe")
                if not temp_data:
                    temp_data = self.W.query("SELECT CurrentTemperature FROM MSAcpi_ThermalZoneTemperature")
                
                if temp_data:
                    temp_reading = temp_data[0].CurrentTemperature if hasattr(temp_data[0], 'CurrentTemperature') else temp_data[0].CurrentReading
                    temp_c = (temp_reading / 10.0) - 273.15
                    temp_c = round(temp_c, 1)

            except Exception:
                temp_c = 0.0
        
        # 3. Emit the final dictionary
        data = {
            'cpu': cpu_usage,
            'memory': mem_usage,
            'temperature': temp_c
        }
        self.data_ready.emit(data)

# ------------------------------------------------------------------
# BENCHMARK WORKER (Runs CPU-intensive tasks)
# ------------------------------------------------------------------
class BenchmarkWorker(QObject):
    """Runs the long-running, CPU-intensive benchmark in a separate thread."""
    result_ready = Signal(dict)
    
    def __init__(self, test_config, parent=None):
        super().__init__(parent)
        self.test_config = test_config
        self._is_running = False

    def run_benchmark(self):
        self._is_running = True
        config = self.test_config
        
        # 🔑 MODIFIED LOGIC: Run multi-core if the user selected more than 1 core.
        # The single-core restriction for 'integer' and 'floating' is REMOVED.
        if config['cores'] > 1:
            # Multi-threading/Multi-processing logic
            
            # We must use the daemon initializer to prevent orphaned processes
            pool = multiprocessing.Pool(
                processes=config['cores'],
                initializer=worker_initializer 
            )
            duration_per_process = config['duration']
            workload_iterable = [(config['test_type'], duration_per_process)] * config['cores']
            
            # Dispatch the workload to all worker processes
            results = pool.starmap(worker_target_function, workload_iterable) 
            
            pool.close()
            pool.join()
            
            # Aggregate results
            total_time = max([r[0] for r in results])
            total_ops = sum([r[1] for r in results])

        else:
            # Single-core execution path (if user set cores=1)
            total_time, total_ops = worker_target_function(
                config['test_type'], 
                config['duration']
            )

        final_score = total_ops / total_time if total_time > 0 else 0
        
        results = {
            'test_type': config['test_type'],
            'cores_used': config['cores'],
            'total_time_s': round(total_time, 3),
            'total_ops': total_ops,
            'score_ops_sec': round(final_score, 2),
            'system_info': psutil.cpu_freq().current,
        }
        
        self.result_ready.emit(results)
        self._is_running = False