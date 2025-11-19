import os
import json
import torch
from typing import Dict, Any
import time

class CacheStatistics:
    """用于统计TaylorSeer缓存使用情况的类"""

    def __init__(self, cache_dir: str = "/home/Projects/Diff-cache/taylorseer_flux"):
        self.cache_dir = cache_dir
        self.stats = {
            'layer_cache_sizes': {},
            'cache_hits': 0,
            'cache_misses': 0,
            'taylor_calculations': 0,
            'full_calculations': 0,
            'toca_calculations': 0,
            'baseline_cache_size': 0,  # 全部计算时的cache大小
            'current_cache_size': 0,
            'taylor_cache_increase': 0,  # TaylorSeer特有的cache增加量
            'taylor_cache_size_mb': 0,
            'peak_cache_size': 0,
            'baseline_established': False,
            'max_order': None,
            'first_enhance': None,
            'fresh_threshold': None,
            'step_logs': []
        }
        self.start_time = time.time()
        self.baseline_locked = False  # 防止重复设置baseline
        self.config_reported = False

    def get_cache_size_mb(self, cache_dic: Dict) -> float:
        """计算cache字典的内存使用量(MB)"""
        total_size = 0
        try:
            # 递归计算字典大小
            def get_dict_size(d):
                size = 0
                for key, value in d.items():
                    if isinstance(value, dict):
                        size += get_dict_size(value)
                    elif isinstance(value, torch.Tensor):
                        size += value.element_size() * value.nelement()
                    else:
                        # 对于其他类型，估算大小
                        size += len(str(value).encode('utf-8'))
                return size

            total_size = get_dict_size(cache_dic)
            return total_size / (1024 * 1024)  # 转换为MB
        except Exception as e:
            print(f"计算cache大小时出错: {e}")
            return 0

    def update_calculation_stats(self, calc_type: str):
        """更新计算类型统计"""
        if calc_type == 'full':
            self.stats['full_calculations'] += 1
        elif calc_type == 'Taylor':
            self.stats['taylor_calculations'] += 1
        elif calc_type == 'ToCa':
            self.stats['toca_calculations'] += 1

    def establish_baseline(self, cache_dic: Dict):
        """建立baseline - 在全部计算且不使用Taylor的情况下测量基础cache大小"""
        if self.baseline_locked:
            return

        baseline_size = self.get_cache_size_mb(cache_dic)
        self.stats['baseline_cache_size'] = baseline_size
        self.stats['baseline_established'] = True
        self.baseline_locked = True
        print(f"✓ Baseline cache size established: {baseline_size:.2f} MB (full calculation without Taylor)")

    def update_cache_size(self, cache_dic: Dict, calc_type: str = None):
        """更新当前cache大小统计"""
        current_size = self.get_cache_size_mb(cache_dic)
        self.stats['current_cache_size'] = current_size
        self.stats['peak_cache_size'] = max(self.stats['peak_cache_size'], current_size)

        # 建立baseline：只在全部计算且不调用Taylor时建立
        if calc_type == 'full' and not self.stats['baseline_established']:
            self.establish_baseline(cache_dic)

        # 计算TaylorSeer特有的cache增加量
        if self.stats['baseline_established']:
            self.stats['taylor_cache_increase'] = current_size - self.stats['baseline_cache_size']
            self.stats['taylor_cache_size_mb'] = max(self.stats['taylor_cache_increase'], 0.0)

    def _record_config(self, cache_dic: Dict):
        if cache_dic is None:
            return
        updated = False
        if self.stats['max_order'] is None:
            self.stats['max_order'] = cache_dic.get('max_order')
            updated = True
        if self.stats['first_enhance'] is None:
            self.stats['first_enhance'] = cache_dic.get('first_enhance')
            updated = True
        if self.stats['fresh_threshold'] is None:
            self.stats['fresh_threshold'] = cache_dic.get('fresh_threshold')
            updated = True
        if updated and not self.config_reported and self.stats['max_order'] is not None:
            print(
                f"[TaylorSeer] 配置: max_order={self.stats['max_order']}, "
                f"first_enhance={self.stats['first_enhance']}, "
                f"fresh_threshold={self.stats['fresh_threshold']}"
            )
            self.config_reported = True

    def _record_step_log(self, current: Dict, calc_type: str):
        if current is None:
            return
        step = int(current.get('step', -1))
        entry = {
            'step': step,
            'type': calc_type,
            'stream': current.get('stream'),
            'layer': current.get('layer')
        }
        self.stats['step_logs'].append(entry)
        print(
            f"[TaylorSeer] Step {step:02d} -> {calc_type} "
            f"(stream={entry['stream']}, layer={entry['layer']})"
        )

    def print_statistics(self):
        """打印cache统计信息"""
        elapsed_time = time.time() - self.start_time

        print("=" * 60)
        print("TaylorSeer Cache Statistics")
        print("=" * 60)
        print(f"运行时间: {elapsed_time:.2f} 秒")

        if self.stats['baseline_established']:
            print(f"Baseline cache大小 (无Taylor): {self.stats['baseline_cache_size']:.2f} MB")
            print(f"当前cache大小: {self.stats['current_cache_size']:.2f} MB")
            print(f"TaylorSeer cache增加量: {self.stats['taylor_cache_increase']:.2f} MB")
            print(f"TaylorSeer cache总量: {self.stats['taylor_cache_size_mb']:.2f} MB")
            print(f"峰值cache大小: {self.stats['peak_cache_size']:.2f} MB")

            if self.stats['baseline_cache_size'] > 0:
                increase_ratio = self.stats['taylor_cache_increase'] / self.stats['baseline_cache_size'] * 100
                print(f"Cache增加比例: {increase_ratio:.1f}%")
        else:
            print(f"当前cache大小: {self.stats['current_cache_size']:.2f} MB")
            print("⚠️  Baseline未建立，无法计算TaylorSeer特有的cache增加量")

        if self.stats['max_order'] is not None:
            print()
            print("TaylorSeer参数:")
            print(f"  max_order: {self.stats['max_order']}")
            print(f"  first_enhance: {self.stats['first_enhance']}")
            print(f"  fresh_threshold: {self.stats['fresh_threshold']}")

        print()
        print("计算类型统计:")
        print(f"  完整计算 (full): {self.stats['full_calculations']} 次")
        print(f"  Taylor近似: {self.stats['taylor_calculations']} 次")
        print(f"  ToCa缓存: {self.stats['toca_calculations']} 次")

        total_calcs = (self.stats['full_calculations'] +
                      self.stats['taylor_calculations'] +
                      self.stats['toca_calculations'])

        if total_calcs > 0:
            taylor_ratio = self.stats['taylor_calculations'] / total_calcs * 100
            cache_ratio = (self.stats['taylor_calculations'] + self.stats['toca_calculations']) / total_calcs * 100
            print()
            print("性能分析:")
            print(f"  Taylor近似比例: {taylor_ratio:.1f}%")
            print(f"  缓存使用比例: {cache_ratio:.1f}%")
            print(f"  计算节省比例: {cache_ratio:.1f}% (近似估算)")

        if self.stats['step_logs']:
            print()
            print("逐步缓存模式:")
            for log in self.stats['step_logs']:
                print(
                    f"  Step {log['step']:>3}: {log['type']:<6} "
                    f"(stream={log.get('stream')}, layer={log.get('layer')})"
                )

        print("=" * 60)

    def save_statistics(self, filename: str = "cache_stats.json"):
        """保存统计信息到文件"""
        try:
            # 如果使用默认文件名，则添加max_order和fresh_threshold
            if filename == "cache_stats.json":
                max_order = self.stats.get("max_order", "unknown")
                fresh_threshold = self.stats.get("fresh_threshold", "unknown")
                filename = f"cache_stats_maxorder{max_order}_fresh{fresh_threshold}.json"

            dir_path = os.path.dirname(filename)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(self.stats, f, indent=2)
            print(f"统计信息已保存到 {filename}")
        except Exception as e:
            print(f"保存统计信息时出错: {e}")

# 全局统计对象
cache_stats = CacheStatistics()

def get_cache_stats():
    """获取全局cache统计对象"""
    return cache_stats

def update_cache_statistics(cache_dic: Dict, calc_type: str, current: Dict | None = None):
    """更新cache统计信息的便捷函数"""
    cache_stats._record_config(cache_dic)
    cache_stats._record_step_log(current, calc_type)
    cache_stats.update_cache_size(cache_dic, calc_type)
    cache_stats.update_calculation_stats(calc_type)

def print_cache_statistics():
    """打印cache统计信息的便捷函数"""
    cache_stats.print_statistics()
