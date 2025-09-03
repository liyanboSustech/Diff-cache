import torch
import math
from typing import Dict, Any, Optional, Tuple
from diffusers.models import FluxTransformer2DModel
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock
import pywt
import numpy as np

class AdaptiveFeatureCache:
    def __init__(self, max_order=4, similarity_threshold=0.8, wavelet='db4', wavelet_levels=1, wavelet_history_size=4):
        self.cache = {}
        self.layer_stability = {} 
        self.similarity_threshold = similarity_threshold
        
        # --- WPC & Taylor 参数 ---
        self.wavelet = wavelet
        self.wavelet_levels = wavelet_levels
        self.wavelet_history_size = wavelet_history_size
        self.max_order = max_order
        
        # 添加维度限制参数 - 进一步降低限制以确保安全
        self.max_dimension_per_chunk = 200  # 更保守的维度限制
        self.enable_chunked_processing = True  # 启用分块处理
        
        # 添加简化 wavelet 选项作为备选
        self.fallback_wavelets = ['haar', 'db1', 'db2']
        self.current_wavelet_idx = 0

    def cache_init(self, transformer: FluxTransformer2DModel):
        cache_dic = {}
        cache_dic['cache'] = {}
        cache_dic['history'] = {} 
        
        streams = ['double_stream', 'single_stream']
        num_layers_map = {
            'double_stream': transformer.config.num_layers,
            'single_stream': transformer.config.num_single_layers
        }

        for stream in streams:
            cache_dic['cache'][stream] = {}
            cache_dic['history'][stream] = {}
            for j in range(num_layers_map[stream]):
                cache_dic['cache'][stream][j] = {}
                cache_dic['history'][stream][j] = {}

        # --- 开关和配置 ---
        cache_dic['wavelet_cache'] = True
        cache_dic['taylor_cache'] = True
        
        # 简化版：我们只预测整个块的最终输出，所以模块名固定
        cache_dic['module_name'] = 'block_output'
        cache_dic['fresh_threshold'] = 3  # 每 N 步强制刷新

        current = {}
        current['activated_steps'] = [0]
        current['step'] = 0
        current['num_steps'] = 50 

        return cache_dic, current

    def select_strategy_for_layer(self, cache_dic, current):
        layer_idx = current['layer']
        stream = current['stream']
        
        if current['step'] in current['activated_steps']:
            return 'full'

        if len(current['activated_steps']) < self.wavelet_history_size:
            return 'full'
            
        stability = self.layer_stability.get(f"{stream}_{layer_idx}", 0.5)

        if cache_dic.get('wavelet_cache') and stability > 0.9:
            return 'Wavelet'
        elif cache_dic.get('taylor_cache') and stability > 0.7:
            return 'Taylor'
        else:
            # 对于不稳定的层，降级为全量计算以保证质量
            return 'full'

    def update_stability(self, cache_dic, current, feature):
        stream = current['stream']
        layer = current['layer']
        module = cache_dic['module_name']
        key = f"{stream}_{layer}"

        # 从Taylor缓存中获取上一个新鲜步的特征
        last_feature_data = cache_dic['cache'][stream][layer].get(module, {}).get(0)
        if last_feature_data is None:
            return
        
        # 确保特征维度匹配
        if last_feature_data.shape != feature.shape:
            return
            
        cos_sim = torch.nn.functional.cosine_similarity(
            feature.reshape(1, -1), 
            last_feature_data.reshape(1, -1)
        ).item()
        
        current_stability = self.layer_stability.get(key, 0.5)
        self.layer_stability[key] = 0.9 * current_stability + 0.1 * cos_sim

    def derivative_approximation(self, cache_dic, current, feature):
        stream, layer, module = current['stream'], current['layer'], cache_dic['module_name']
        diff_dist = current['activated_steps'][-1] - current['activated_steps'][-2] if len(current['activated_steps']) > 1 else 1

        updated_factors = {0: feature}
        for i in range(self.max_order):
            prev_factor = cache_dic['cache'][stream][layer].get(module, {}).get(i)
            if prev_factor is not None:
                updated_factors[i + 1] = (updated_factors[i] - prev_factor) / diff_dist
            else:
                break
        cache_dic['cache'][stream][layer][module] = updated_factors

    def taylor_formula(self, cache_dic, current):
        stream, layer, module = current['stream'], current['layer'], cache_dic['module_name']
        x = current['step'] - current['activated_steps'][-1]
        
        output = 0
        taylor_factors = cache_dic['cache'][stream][layer].get(module, {})
        for i in range(len(taylor_factors)):
            output += (1 / math.factorial(i)) * taylor_factors[i] * (x ** i)
        return output

    def update_feature_history(self, cache_dic, current, feature):
        stream, layer, module = current['stream'], current['layer'], cache_dic['module_name']
        if module not in cache_dic['history'][stream][layer]:
            cache_dic['history'][stream][layer][module] = []
        
        history = cache_dic['history'][stream][layer][module]
        feat_cpu = feature.detach().to(torch.float32).cpu()
        history.append(feat_cpu.contiguous().numpy())
        if len(history) > self.wavelet_history_size:
            history.pop(0)

    def _try_wavelet_with_fallback(self, signal_chunk, wavelet_name):
        """尝试使用指定 wavelet，如果失败则尝试备选 wavelet"""
        try:
            # 限制分解层数以避免维度问题
            max_levels = min(self.wavelet_levels, pywt.dwt_max_level(signal_chunk.shape[1], wavelet_name))
            if max_levels <= 0:
                return None
                
            coeffs = pywt.wavedec(signal_chunk, wavelet_name, level=max_levels, axis=1)
            
            # 预测系数
            cA = coeffs[0]
            if cA.shape[1] < 2:
                return None
                
            cA_pred = cA[:, -1] + (cA[:, -1] - cA[:, -2])
            
            pred_coeffs = [cA_pred.reshape(-1, 1)]
            for cD in coeffs[1:]:
                pred_coeffs.append(np.zeros_like(cD[:, :1]))
            
            # 重构信号
            reconstructed = pywt.waverec(pred_coeffs, wavelet_name, axis=1)
            return reconstructed[:, 0]
            
        except (ValueError, RuntimeError) as e:
            if "Maximum allowed dimension exceeded" in str(e) or "Invalid signal length" in str(e):
                return None
            else:
                raise e

    def _process_wavelet_chunk(self, signal_chunk):
        """处理单个 wavelet 块，带有多级备选方案"""
        # 首先尝试原始 wavelet
        result = self._try_wavelet_with_fallback(signal_chunk, self.wavelet)
        if result is not None:
            return result
            
        # 如果失败，尝试备选 wavelets
        for fallback_wavelet in self.fallback_wavelets:
            result = self._try_wavelet_with_fallback(signal_chunk, fallback_wavelet)
            if result is not None:
                return result
        
        # 如果所有 wavelet 都失败，使用简单预测
        return self._simple_predict_chunk(signal_chunk)

    def _simple_predict_chunk(self, signal_chunk):
        """简单预测方法作为备选方案"""
        # 使用线性外推
        if signal_chunk.shape[1] >= 2:
            last_val = signal_chunk[:, -1]
            second_last_val = signal_chunk[:, -2]
            predicted = 2 * last_val - second_last_val
            return predicted
        else:
            return signal_chunk[:, -1]

    def _adaptive_chunk_size(self, total_elements):
        """根据数据大小自适应调整块大小"""
        if total_elements > 10000:
            return min(self.max_dimension_per_chunk // 2, 100)
        elif total_elements > 5000:
            return min(self.max_dimension_per_chunk // 1.5, 150)
        else:
            return self.max_dimension_per_chunk

    def wavelet_predict(self, cache_dic, current):
        stream, layer, module = current['stream'], current['layer'], cache_dic['module_name']
        history = cache_dic['history'][stream][layer].get(module, [])
        
        if len(history) < self.wavelet_history_size:
            return None

        try:
            device = next(self.parameters()).device if isinstance(self, torch.nn.Module) else 'cuda'
        except:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            signal = np.stack(history, axis=-1)
            original_shape = signal.shape[:-1]
            signal_flat = signal.reshape(-1, self.wavelet_history_size)
            
            # 检查维度是否超出限制
            total_elements = signal_flat.shape[0]
            
            # 如果数据太大，直接使用简单预测
            if total_elements > self.max_dimension_per_chunk * 20:
                print(f"Data too large ({total_elements}), using simple prediction")
                predicted_flat = self._simple_predict_chunk(signal_flat)
            
            elif self.enable_chunked_processing and total_elements > self.max_dimension_per_chunk:
                # 分块处理，使用自适应块大小
                predicted_chunks = []
                chunk_size = self._adaptive_chunk_size(total_elements)
                
                for i in range(0, total_elements, chunk_size):
                    end_idx = min(i + chunk_size, total_elements)
                    chunk = signal_flat[i:end_idx]
                    
                    try:
                        predicted_chunk = self._process_wavelet_chunk(chunk)
                        predicted_chunks.append(predicted_chunk)
                    except Exception as e:
                        print(f"Warning: Wavelet chunk processing failed: {e}")
                        predicted_chunks.append(self._simple_predict_chunk(chunk))
                
                # 合并预测结果
                if predicted_chunks:
                    predicted_flat = np.concatenate(predicted_chunks, axis=0)
                else:
                    # 如果所有块都失败，使用最简单的预测
                    predicted_flat = signal_flat[:, -1]
            
            else:
                # 直接处理（维度较小的情况）
                try:
                    predicted_flat = self._process_wavelet_chunk(signal_flat)
                except Exception as e:
                    print(f"Warning: Direct wavelet processing failed: {e}")
                    predicted_flat = self._simple_predict_chunk(signal_flat)
            
            # 恢复原始形状并转换为张量
            predicted_feature = torch.from_numpy(predicted_flat).reshape(original_shape).to(device)
            
            # 数据类型和范围检查
            if predicted_feature.dtype != torch.float32:
                predicted_feature = predicted_feature.float()
                
            # 检查 NaN 和 Inf
            if torch.any(torch.isnan(predicted_feature)) or torch.any(torch.isinf(predicted_feature)):
                print("Warning: NaN or Inf detected in wavelet prediction, falling back to simple method")
                return None
                
            return predicted_feature
            
        except Exception as e:
            print(f"Warning: Wavelet prediction completely failed: {e}")
            return None

def afcache_flux_double_block_forward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None):
    joint_attention_kwargs = joint_attention_kwargs or {}
    cache_dic = joint_attention_kwargs['cache_dic']
    current = joint_attention_kwargs['current']
    
    afcache = self.afcache_instance # 从 self 获取实例
    
    strategy = afcache.select_strategy_for_layer(cache_dic, current)

    if strategy == 'full':
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)
        
        attention_outputs = self.attn(hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states, image_rotary_emb=image_rotary_emb)
        attn_output, context_attn_output = attention_outputs
        
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output
        
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        
        # 更新所有缓存
        afcache.update_stability(cache_dic, current, hidden_states)
        afcache.derivative_approximation(cache_dic, current, hidden_states)
        afcache.update_feature_history(cache_dic, current, hidden_states)

    elif strategy == 'Wavelet':
        try:
            pred_hs = afcache.wavelet_predict(cache_dic, current)
            if pred_hs is not None:
                hidden_states = pred_hs
                # 注意: encoder_hidden_states 在此简化模型中未被预测，实际可能需要单独预测
                # 为保持流程完整，我们假设它保持不变或使用简单策略
            else: # 降级为 Taylor
                hidden_states = afcache.taylor_formula(cache_dic, current)
        except Exception as e:
            print(f"Warning: Wavelet prediction failed, falling back to Taylor: {e}")
            try:
                hidden_states = afcache.taylor_formula(cache_dic, current)
            except Exception as e2:
                print(f"Warning: Taylor formula also failed: {e2}, using input as fallback")
                # 最后的备选方案：保持输入不变
                pass
            
    elif strategy == 'Taylor':
        try:
            hidden_states = afcache.taylor_formula(cache_dic, current)
        except Exception as e:
            print(f"Warning: Taylor formula failed: {e}, using input as fallback")
            # 保持输入不变作为最后备选
            pass

    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states

def afcache_flux_single_block_forward(self, hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None):
    joint_attention_kwargs = joint_attention_kwargs or {}
    cache_dic = joint_attention_kwargs['cache_dic']
    current = joint_attention_kwargs['current']
    
    afcache = self.afcache_instance # 从 self 获取实例

    strategy = afcache.select_strategy_for_layer(cache_dic, current)
    
    residual = hidden_states

    if strategy == 'full':
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        gate = gate.unsqueeze(1)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        attn_output = self.attn(hidden_states=norm_hidden_states, image_rotary_emb=image_rotary_emb)
        calc_hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        calc_hidden_states = self.proj_out(calc_hidden_states)
        
        # 更新所有缓存
        afcache.update_stability(cache_dic, current, calc_hidden_states)
        afcache.derivative_approximation(cache_dic, current, calc_hidden_states)
        afcache.update_feature_history(cache_dic, current, calc_hidden_states)

        hidden_states = gate * calc_hidden_states

    elif strategy == 'Wavelet':
        try:
            pred_hs = afcache.wavelet_predict(cache_dic, current)
            if pred_hs is None: # 降级
                pred_hs = afcache.taylor_formula(cache_dic, current)
            
            # Gate 也需要被缓存或预测，这里简化为使用上一步的
            # 在一个完整的实现中，gate也应该被缓存
            _, gate = self.norm(hidden_states, emb=temb) 
            hidden_states = gate.unsqueeze(1) * pred_hs
        except Exception as e:
            print(f"Warning: Wavelet prediction failed, falling back to Taylor: {e}")
            try:
                pred_hs = afcache.taylor_formula(cache_dic, current)
                _, gate = self.norm(hidden_states, emb=temb)
                hidden_states = gate.unsqueeze(1) * pred_hs
            except Exception as e2:
                print(f"Warning: Taylor formula also failed: {e2}")
                # 保持原始计算作为最后备选
                pass
            
    elif strategy == 'Taylor':
        try:
            pred_hs = afcache.taylor_formula(cache_dic, current)
            _, gate = self.norm(hidden_states, emb=temb)
            hidden_states = gate.unsqueeze(1) * pred_hs
        except Exception as e:
            print(f"Warning: Taylor formula failed: {e}")
            # 保持原始计算作为最后备选
            pass

    hidden_states = residual + hidden_states
    
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    return hidden_states
