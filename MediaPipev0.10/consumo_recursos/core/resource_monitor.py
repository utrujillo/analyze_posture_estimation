import platform
import psutil
import torch
import time
import subprocess
from typing import Dict, Optional, Any

class ResourceMonitor:
    """Clase mejorada para monitoreo de recursos con soporte completo para Apple Silicon"""
    
    def __init__(self):
        self.system = platform.system()
        self.process = psutil.Process()
        self.cpu_cores = psutil.cpu_count(logical=False)
        self.gpu_type = self._detect_gpu_type()
        
    def _detect_gpu_type(self) -> Optional[str]:
        """Detecta el tipo de GPU disponible"""
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        return None
    
    def get_system_info(self) -> Dict[str, Any]:
        """Obtiene información estática del sistema"""
        return {
            'system': platform.platform(),
            'cpu': self._get_cpu_info(),
            'ram': self._get_ram_info(),
            'gpu': self._get_gpu_info()
        }
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """Información detallada del CPU"""
        freq = psutil.cpu_freq()
        return {
            'model': platform.processor(),
            'cores': self.cpu_cores,
            'max_freq': f"{freq.max:.0f}MHz",
            'current_freq': f"{freq.current:.0f}MHz"
        }
    
    def _get_ram_info(self) -> Dict[str, str]:
        """Información de la memoria RAM"""
        mem = psutil.virtual_memory()
        return {
            'total': f"{mem.total / (1024**3):.1f}GB",
            'available': f"{mem.available / (1024**3):.1f}GB"
        }
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Información de la GPU"""
        if self.gpu_type == 'mps':
            return {
                'type': 'Apple Silicon GPU',
                'details': 'Metal Performance Shaders',
                'architecture': 'M1/M2 GPU'
            }
        elif self.gpu_type == 'cuda':
            return {
                'type': 'NVIDIA',
                'name': torch.cuda.get_device_name(0),
                'capability': torch.cuda.get_device_capability(0),
                'total_memory': f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB"
            }
        return {'type': 'No GPU disponible'}
    
    def measure(self) -> Dict[str, Any]:
        """Mide el uso actual de recursos con soporte para Apple Silicon"""
        # Sincronizar operaciones GPU antes de medir
        if self.gpu_type == 'cuda':
            torch.cuda.synchronize()
        elif self.gpu_type == 'mps':
            torch.mps.synchronize()
        
        return {
            'cpu': self._measure_cpu(),
            'ram': self._measure_ram(),
            'gpu': self._measure_gpu()
        }
    
    def _measure_cpu(self) -> Dict[str, Any]:
        """Mide uso del CPU"""
        return {
            'usage_percent': psutil.cpu_percent(interval=0.1),
            'per_core_usage': psutil.cpu_percent(interval=0.1, percpu=True)
        }
    
    def _measure_ram(self) -> Dict[str, float]:
        """Mide uso de RAM"""
        mem = self.process.memory_info()
        return {
            'rss_mb': mem.rss / (1024 * 1024),  # Resident Set Size
            'vms_mb': mem.vms / (1024 * 1024)   # Virtual Memory Size
        }
    
    def _measure_gpu(self) -> Optional[Dict[str, Any]]:
        """Mide uso de GPU con soporte mejorado para Apple Silicon"""
        if self.gpu_type == 'mps':
            return self._measure_mps_gpu()
        elif self.gpu_type == 'cuda':
            return self._measure_cuda_gpu()
        return None
    
    def _measure_mps_gpu(self) -> Dict[str, Any]:
        """Métricas específicas para GPU Apple Silicon"""
        torch.mps.empty_cache()
        
        gpu_data = {
            'memory_used_mb': torch.mps.current_allocated_memory() / (1024 * 1024),
            'memory_reserved_mb': torch.mps.driver_allocated_memory() / (1024 * 1024),
            'utilization_percent': self._get_mac_gpu_utilization(),
            'gpu_type': 'mps'
        }
        
        # Intento obtener información adicional del sistema
        try:
            gpu_data.update(self._get_mac_gpu_extra_info())
        except:
            pass
            
        return gpu_data
    
    def _measure_cuda_gpu(self) -> Dict[str, Any]:
        """Métricas para GPU NVIDIA"""
        torch.cuda.empty_cache()
        gpu_data = {
            'memory_used_mb': torch.cuda.memory_allocated() / (1024 * 1024),
            'memory_reserved_mb': torch.cuda.memory_reserved() / (1024 * 1024),
            'utilization_percent': 0,
            'gpu_type': 'cuda'
        }
        
        # Intento obtener información de NVML si está disponible
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_data['utilization_percent'] = util.gpu
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_data['memory_used_mb_nvml'] = mem_info.used / (1024 * 1024)
            pynvml.nvmlShutdown()
        except:
            pass
            
        return gpu_data
    
    def _get_mac_gpu_utilization(self) -> float:
        """Estimación de utilización de GPU en macOS"""
        try:
            # Método 1: Usando actividad de CPU como proxy
            cpu_usage = psutil.cpu_percent(interval=0.1)
            return min(cpu_usage / self.cpu_cores * 100, 100)
            
            # Método alternativo: Usando system_profiler (lento pero más preciso)
            # result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
            #                       capture_output=True, text=True)
            # output = result.stdout
            # if 'Utilization' in output:
            #     return float(output.split('Utilization')[1].split('%')[0].strip())
        except:
            return 0.0
    
    def _get_mac_gpu_extra_info(self) -> Dict[str, Any]:
        """Obtiene información adicional de GPU en macOS"""
        try:
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                  capture_output=True, text=True)
            output = result.stdout
            
            info = {}
            if 'Metal Support' in output:
                info['metal_support'] = output.split('Metal Support:')[1].split('\n')[0].strip()
            if 'VRAM' in output:
                vram = output.split('VRAM (Dynamic, Max):')[1].split('\n')[0].strip()
                info['vram_total'] = vram
            return info
        except:
            return {}