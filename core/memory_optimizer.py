#!/usr/bin/env python3
"""
Memory Optimizer - Advanced memory management for 3D pose processing

Priority: HIGH
Dependencies: torch, numpy, psutil
Test Coverage Required: 100%

This module implements intelligent memory management with garbage collection,
caching strategies, and memory-aware processing.
"""

import numpy as np
import torch
import gc
import psutil
import weakref
import threading
import time
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import sys
import logging
from functools import wraps
from collections import OrderedDict
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory optimization"""
    max_cache_size_gb: float = 2.0
    gc_threshold: float = 0.8  # Trigger GC at 80% memory usage
    enable_smart_caching: bool = True
    cache_ttl_seconds: float = 300.0  # 5 minutes
    memory_monitoring: bool = True
    auto_optimization: bool = True
    prefetch_enabled: bool = True
    compression_enabled: bool = False


class MemoryMonitor:
    """Real-time memory monitoring and alerts"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.monitoring_active = False
        self.monitor_thread = None
        self.callbacks = []
        
        # Memory tracking
        self.memory_history = []
        self.peak_usage = {'cpu': 0.0, 'gpu': 0.0}
        self.last_gc_time = time.time()
        
    def start_monitoring(self):
        """Start memory monitoring thread"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Memory monitoring loop"""
        while self.monitoring_active:
            try:
                # Get current memory usage
                cpu_usage = self.get_cpu_memory_usage()
                gpu_usage = self.get_gpu_memory_usage()
                
                # Update peak usage
                self.peak_usage['cpu'] = max(self.peak_usage['cpu'], cpu_usage)
                self.peak_usage['gpu'] = max(self.peak_usage['gpu'], gpu_usage)
                
                # Store history
                timestamp = time.time()
                self.memory_history.append({
                    'timestamp': timestamp,
                    'cpu_gb': cpu_usage,
                    'gpu_gb': gpu_usage
                })
                
                # Keep only recent history (last 5 minutes)
                cutoff_time = timestamp - 300
                self.memory_history = [
                    entry for entry in self.memory_history 
                    if entry['timestamp'] > cutoff_time
                ]
                
                # Check thresholds and trigger actions
                max_usage = max(cpu_usage, gpu_usage)
                if max_usage > self.config.gc_threshold:
                    self._trigger_memory_cleanup()
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(cpu_usage, gpu_usage)
                    except Exception as e:
                        logger.warning(f"Memory callback failed: {e}")
                
                time.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(5.0)  # Wait longer on error
    
    def get_cpu_memory_usage(self) -> float:
        """Get CPU memory usage in GB"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / (1024**3)
    
    def get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0
    
    def _trigger_memory_cleanup(self):
        """Trigger memory cleanup when threshold exceeded"""
        current_time = time.time()
        
        # Avoid too frequent GC calls
        if current_time - self.last_gc_time < 10.0:  # 10 seconds minimum
            return
        
        logger.info("Memory threshold exceeded, triggering cleanup")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        self.last_gc_time = current_time
    
    def add_callback(self, callback: Callable[[float, float], None]):
        """Add memory usage callback"""
        self.callbacks.append(callback)
    
    def get_statistics(self) -> Dict:
        """Get memory usage statistics"""
        if not self.memory_history:
            return {'status': 'no_data'}
        
        recent_history = self.memory_history[-60:]  # Last minute
        cpu_values = [entry['cpu_gb'] for entry in recent_history]
        gpu_values = [entry['gpu_gb'] for entry in recent_history]
        
        return {
            'current_cpu_gb': self.get_cpu_memory_usage(),
            'current_gpu_gb': self.get_gpu_memory_usage(),
            'peak_cpu_gb': self.peak_usage['cpu'],
            'peak_gpu_gb': self.peak_usage['gpu'],
            'avg_cpu_gb_1min': np.mean(cpu_values),
            'avg_gpu_gb_1min': np.mean(gpu_values),
            'monitoring_active': self.monitoring_active,
            'history_entries': len(self.memory_history)
        }


class SmartCache:
    """Intelligent caching with TTL and memory awareness"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.cache = OrderedDict()
        self.access_times = {}
        self.creation_times = {}
        self.memory_usage = 0.0  # Estimated cache memory usage in GB
        
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self._lock:
            if key in self.cache:
                # Update access time
                self.access_times[key] = time.time()
                
                # Check TTL
                if self._is_expired(key):
                    self._remove(key)
                    return None
                
                # Move to end (most recently used)
                value = self.cache[key]
                del self.cache[key]
                self.cache[key] = value
                
                return value
            
            return None
    
    def put(self, key: str, value: Any, size_hint: Optional[float] = None):
        """Put item in cache"""
        with self._lock:
            # Estimate size if not provided
            if size_hint is None:
                size_hint = self._estimate_size(value)
            
            # Remove existing item if present
            if key in self.cache:
                self._remove(key)
            
            # Check if we need to evict items
            while (self.memory_usage + size_hint > self.config.max_cache_size_gb and 
                   len(self.cache) > 0):
                self._evict_lru()
            
            # Add new item
            current_time = time.time()
            self.cache[key] = value
            self.access_times[key] = current_time
            self.creation_times[key] = current_time
            self.memory_usage += size_hint
    
    def _estimate_size(self, value: Any) -> float:
        """Estimate memory size of value in GB"""
        if isinstance(value, (torch.Tensor, np.ndarray)):
            if isinstance(value, torch.Tensor):
                return value.element_size() * value.numel() / (1024**3)
            else:
                return value.nbytes / (1024**3)
        elif isinstance(value, (list, tuple)):
            return sum(self._estimate_size(item) for item in value)
        elif isinstance(value, dict):
            return sum(self._estimate_size(v) for v in value.values())
        else:
            # Rough estimate for other objects
            return sys.getsizeof(value) / (1024**3)
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self.creation_times:
            return True
        
        age = time.time() - self.creation_times[key]
        return age > self.config.cache_ttl_seconds
    
    def _evict_lru(self):
        """Evict least recently used item"""
        if not self.cache:
            return
        
        # Find LRU item
        lru_key = min(self.access_times.keys(), 
                     key=lambda k: self.access_times[k])
        
        self._remove(lru_key)
    
    def _remove(self, key: str):
        """Remove item from cache"""
        if key in self.cache:
            value = self.cache[key]
            size = self._estimate_size(value)
            
            del self.cache[key]
            del self.access_times[key]
            del self.creation_times[key]
            
            self.memory_usage = max(0.0, self.memory_usage - size)
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.creation_times.clear()
            self.memory_usage = 0.0
    
    def cleanup_expired(self):
        """Remove expired entries"""
        with self._lock:
            expired_keys = [
                key for key in self.cache.keys()
                if self._is_expired(key)
            ]
            
            for key in expired_keys:
                self._remove(key)
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_statistics(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            return {
                'entries': len(self.cache),
                'memory_usage_gb': self.memory_usage,
                'max_size_gb': self.config.max_cache_size_gb,
                'utilization': self.memory_usage / self.config.max_cache_size_gb,
                'oldest_entry_age': (
                    time.time() - min(self.creation_times.values())
                    if self.creation_times else 0
                ),
                'ttl_seconds': self.config.cache_ttl_seconds
            }


class MemoryOptimizedProcessor:
    """Memory-optimized processor with intelligent management"""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        
        # Initialize components
        self.monitor = MemoryMonitor(self.config)
        self.cache = SmartCache(self.config)
        
        # Optimization state
        self.optimization_active = False
        self.processing_stats = {
            'frames_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_optimizations': 0
        }
        
        # Start monitoring if enabled
        if self.config.memory_monitoring:
            self.monitor.start_monitoring()
            self.monitor.add_callback(self._memory_callback)
        
        # Setup periodic cleanup
        self._setup_periodic_cleanup()
        
        logger.info("MemoryOptimizedProcessor initialized")
    
    def _memory_callback(self, cpu_usage: float, gpu_usage: float):
        """Callback for memory usage changes"""
        # Trigger optimization if memory usage is high
        if max(cpu_usage, gpu_usage) > self.config.gc_threshold:
            if not self.optimization_active:
                self._optimize_memory()
    
    def _setup_periodic_cleanup(self):
        """Setup periodic cleanup thread"""
        def cleanup_loop():
            while True:
                time.sleep(60.0)  # Run every minute
                
                try:
                    # Clean expired cache entries
                    self.cache.cleanup_expired()
                    
                    # Force GC occasionally
                    if time.time() % 300 < 60:  # Every 5 minutes
                        gc.collect()
                        
                except Exception as e:
                    logger.error(f"Periodic cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
    
    def _optimize_memory(self):
        """Perform memory optimization"""
        if self.optimization_active:
            return
        
        self.optimization_active = True
        
        try:
            logger.info("Starting memory optimization")
            
            # Clear least important cache entries
            self.cache.cleanup_expired()
            
            # If still high memory, evict more cache entries
            if self.monitor.get_cpu_memory_usage() > self.config.gc_threshold:
                initial_entries = len(self.cache.cache)
                
                # Evict up to 50% of cache
                target_size = max(1, len(self.cache.cache) // 2)
                while len(self.cache.cache) > target_size:
                    self.cache._evict_lru()
                
                evicted = initial_entries - len(self.cache.cache)
                logger.info(f"Evicted {evicted} cache entries for memory optimization")
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            self.processing_stats['memory_optimizations'] += 1
            
            logger.info("Memory optimization completed")
            
        finally:
            self.optimization_active = False
    
    def process_with_memory_optimization(self, 
                                       data: Any,
                                       processor_func: Callable,
                                       cache_key: Optional[str] = None) -> Any:
        """Process data with memory optimization"""
        
        # Try cache first if key provided
        if cache_key and self.config.enable_smart_caching:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.processing_stats['cache_hits'] += 1
                return cached_result
            
            self.processing_stats['cache_misses'] += 1
        
        # Pre-processing memory check
        if self.config.auto_optimization:
            current_usage = max(
                self.monitor.get_cpu_memory_usage(),
                self.monitor.get_gpu_memory_usage()
            )
            
            if current_usage > self.config.gc_threshold:
                self._optimize_memory()
        
        # Process data
        try:
            result = processor_func(data)
            
            # Cache result if key provided
            if cache_key and self.config.enable_smart_caching:
                self.cache.put(cache_key, result)
            
            self.processing_stats['frames_processed'] += 1
            
            return result
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("Out of memory error, attempting recovery")
                
                # Emergency memory cleanup
                self.cache.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Retry with smaller data or different approach
                logger.info("Retrying after memory cleanup")
                result = processor_func(data)
                
                self.processing_stats['frames_processed'] += 1
                return result
            else:
                raise
    
    def create_memory_efficient_batch(self, data_list: List[Any], 
                                    max_memory_gb: Optional[float] = None) -> List[List[Any]]:
        """Create memory-efficient batches"""
        if max_memory_gb is None:
            max_memory_gb = self.config.max_cache_size_gb / 2  # Conservative
        
        batches = []
        current_batch = []
        current_size = 0.0
        
        for data in data_list:
            # Estimate size
            if isinstance(data, (torch.Tensor, np.ndarray)):
                if isinstance(data, torch.Tensor):
                    item_size = data.element_size() * data.numel() / (1024**3)
                else:
                    item_size = data.nbytes / (1024**3)
            else:
                item_size = 0.01  # Small default
            
            # Check if adding this item exceeds limit
            if current_size + item_size > max_memory_gb and current_batch:
                batches.append(current_batch)
                current_batch = [data]
                current_size = item_size
            else:
                current_batch.append(data)
                current_size += item_size
        
        # Add remaining batch
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"Created {len(batches)} memory-efficient batches")
        return batches
    
    def get_optimization_statistics(self) -> Dict:
        """Get comprehensive optimization statistics"""
        stats = {
            'processing': self.processing_stats.copy(),
            'memory': self.monitor.get_statistics(),
            'cache': self.cache.get_statistics(),
            'config': {
                'max_cache_gb': self.config.max_cache_size_gb,
                'gc_threshold': self.config.gc_threshold,
                'auto_optimization': self.config.auto_optimization,
                'smart_caching': self.config.enable_smart_caching
            }
        }
        
        # Calculate derived metrics
        total_requests = stats['processing']['cache_hits'] + stats['processing']['cache_misses']
        if total_requests > 0:
            stats['cache_hit_rate'] = stats['processing']['cache_hits'] / total_requests
        else:
            stats['cache_hit_rate'] = 0.0
        
        return stats
    
    def shutdown(self):
        """Cleanup and shutdown optimizer"""
        logger.info("Shutting down memory optimizer")
        
        if self.config.memory_monitoring:
            self.monitor.stop_monitoring()
        
        self.cache.clear()
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def memory_optimized(cache_key: Optional[str] = None,
                    max_memory_gb: Optional[float] = None):
    """Decorator for memory-optimized processing"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create optimizer instance
            if not hasattr(wrapper, '_optimizer'):
                config = MemoryConfig()
                if max_memory_gb:
                    config.max_cache_size_gb = max_memory_gb
                wrapper._optimizer = MemoryOptimizedProcessor(config)
            
            # Generate cache key if needed
            if cache_key:
                key = f"{func.__name__}_{cache_key}_{hash(str(args) + str(kwargs))}"
            else:
                key = None
            
            # Process with optimization
            return wrapper._optimizer.process_with_memory_optimization(
                (args, kwargs),
                lambda data: func(*data[0], **data[1]),
                key
            )
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test memory optimization
    print("Testing memory optimization...")
    
    config = MemoryConfig(
        max_cache_size_gb=0.5,
        gc_threshold=0.7,
        enable_smart_caching=True
    )
    
    optimizer = MemoryOptimizedProcessor(config)
    
    # Test function
    def dummy_processor(data):
        # Simulate processing
        result = np.random.randn(1000, 1000).astype(np.float32)
        time.sleep(0.1)
        return result
    
    # Test processing with caching
    start_time = time.time()
    
    for i in range(10):
        cache_key = f"test_data_{i % 3}"  # Reuse some keys
        
        result = optimizer.process_with_memory_optimization(
            f"data_{i}",
            dummy_processor,
            cache_key
        )
        
        print(f"Processed item {i}, result shape: {result.shape}")
    
    total_time = time.time() - start_time
    
    # Get statistics
    stats = optimizer.get_optimization_statistics()
    
    print(f"\nOptimization test completed:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"  Cache entries: {stats['cache']['entries']}")
    print(f"  Memory usage: {stats['cache']['memory_usage_gb']:.3f} GB")
    print(f"  Memory optimizations: {stats['processing']['memory_optimizations']}")
    
    # Cleanup
    optimizer.shutdown()
    
    print("[PASS] Memory optimization test completed")