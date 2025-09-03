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
        history.append(feature.detach().cpu().numpy())
        
        if len(history) > self.wavelet_history_size:
            history.pop(0)

    def wavelet_predict(self, cache_dic, current):
        stream, layer, module = current['stream'], current['layer'], cache_dic['module_name']
        history = cache_dic['history'][stream][layer].get(module, [])
        
        if len(history) < self.wavelet_history_size:
            return None

        device = next(self.parameters()).device if isinstance(self, torch.nn.Module) else 'cuda'
        
        signal = np.stack(history, axis=-1)
        original_shape = signal.shape[:-1]
        signal_flat = signal.reshape(-1, self.wavelet_history_size)
        
        coeffs = pywt.wavedec(signal_flat, self.wavelet, level=self.wavelet_levels, axis=1)
        
        cA = coeffs[0]
        cA_pred = cA[:, -1] + (cA[:, -1] - cA[:, -2])
        
        pred_coeffs = [cA_pred.reshape(-1, 1)]
        for cD in coeffs[1:]:
            pred_coeffs.append(np.zeros_like(cD[:, :1]))
        
        reconstructed_flat = pywt.waverec(pred_coeffs, self.wavelet, axis=1)
        
        predicted_feature = torch.from_numpy(reconstructed_flat[:, 0]).reshape(original_shape).to(device)
        return predicted_feature

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
        pred_hs = afcache.wavelet_predict(cache_dic, current)
        if pred_hs is not None:
            hidden_states = pred_hs
            # 注意: encoder_hidden_states 在此简化模型中未被预测，实际可能需要单独预测
            # 为保持流程完整，我们假设它保持不变或使用简单策略
        else: # 降级为 Taylor
            hidden_states = afcache.taylor_formula(cache_dic, current)
            
    elif strategy == 'Taylor':
        hidden_states = afcache.taylor_formula(cache_dic, current)

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
        pred_hs = afcache.wavelet_predict(cache_dic, current)
        if pred_hs is None: # 降级
            pred_hs = afcache.taylor_formula(cache_dic, current)
        
        # Gate 也需要被缓存或预测，这里简化为使用上一步的
        # 在一个完整的实现中，gate也应该被缓存
        _, gate = self.norm(hidden_states, emb=temb) 
        hidden_states = gate.unsqueeze(1) * pred_hs
            
    elif strategy == 'Taylor':
        pred_hs = afcache.taylor_formula(cache_dic, current)
        _, gate = self.norm(hidden_states, emb=temb)
        hidden_states = gate.unsqueeze(1) * pred_hs

    hidden_states = residual + hidden_states
    
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    return hidden_states