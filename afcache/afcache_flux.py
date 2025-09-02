import torch
import math
from typing import Dict, Any, Optional, Tuple
from diffusers.models import FluxTransformer2DModel
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock

class AdaptiveFeatureCache:
    def __init__(self, max_order=4, similarity_threshold=0.1):
        self.cache = {}
        self.strategy_weights = {
            'attention': 0.25,
            'feature': 0.25,
            'taylor': 0.25,
            'difference': 0.25
        }
        self.layer_stability = {}  # 记录每层的稳定性
        self.max_order = max_order
        self.similarity_threshold = similarity_threshold
        self.cache_counter = 0
        self.activated_steps = [0]
        
    def cache_init(self, transformer: FluxTransformer2DModel):   
        '''
        Initialization for cache.
        '''
        cache_dic = {}
        cache = {}
        cache_index = {}
        cache[-1] = {}
        cache_index[-1] = {}
        cache_index['layer_index'] = {}
        cache_dic['attn_map'] = {}
        cache_dic['attn_map'][-1] = {}
        cache_dic['attn_map'][-1]['double_stream'] = {}
        cache_dic['attn_map'][-1]['single_stream'] = {}

        cache[-1]['double_stream'] = {}
        cache[-1]['single_stream'] = {}
        cache_dic['cache_counter'] = 0

        for j in range(transformer.config.num_layers):
            cache[-1]['double_stream'][j] = {}
            cache_index[-1][j] = {}
            cache_dic['attn_map'][-1]['double_stream'][j] = {}
            cache_dic['attn_map'][-1]['double_stream'][j]['total'] = {}
            cache_dic['attn_map'][-1]['double_stream'][j]['txt_mlp'] = {}
            cache_dic['attn_map'][-1]['double_stream'][j]['img_mlp'] = {}

        for j in range(transformer.config.num_single_layers):
            cache[-1]['single_stream'][j] = {}
            cache_index[-1][j] = {}
            cache_dic['attn_map'][-1]['single_stream'][j] = {}
            cache_dic['attn_map'][-1]['single_stream'][j]['total'] = {}

        cache_dic['taylor_cache'] = True
        cache_dic['Delta-DiT'] = False
        cache_dic['feature_cache'] = True
        cache_dic['attention_cache'] = True
        cache_dic['difference_cache'] = True

        # AFCache模式配置
        cache_dic['cache_type'] = 'adaptive'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'adaptive' 
        cache_dic['fresh_ratio'] = 0.0
        cache_dic['fresh_threshold'] = 3
        cache_dic['force_fresh'] = 'global' 
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['max_order'] = self.max_order
        cache_dic['first_enhance'] = 3

        current = {}
        current['activated_steps'] = [0]
        current['step'] = 0
        current['num_steps'] = 50  # 默认值，会在运行时更新

        return cache_dic, current

    def select_strategy(self, layer_idx, timestep, feature_map):
        """
        根据层索引、时间步和特征图自适应选择最佳缓存策略
        """
        # 基于层稳定性的策略选择
        stability = self.layer_stability.get(layer_idx, 0.5)
        
        if timestep < 3:  # 前几个时间步总是完全计算
            return 'full'
        elif stability > 0.8:  # 高稳定性层使用泰勒展开
            return 'taylor'
        elif stability > 0.6:  # 中等稳定性层使用特征相似性
            return 'feature'
        elif stability > 0.4:  # 低稳定性层使用注意力缓存
            return 'attention'
        else:  # 极低稳定性层使用差异检测
            return 'difference'

    def update_stability(self, layer_idx, similarity_score):
        """
        更新层的稳定性评估
        """
        if layer_idx not in self.layer_stability:
            self.layer_stability[layer_idx] = 0.5
        # 指数移动平均更新稳定性
        self.layer_stability[layer_idx] = 0.9 * self.layer_stability[layer_idx] + 0.1 * similarity_score

    def is_feature_similar(self, cache_dict, layer_idx, current_features):
        """
        检查当前特征与缓存特征的相似性
        """
        if -1 not in cache_dict['cache'] or \
           'double_stream' not in cache_dict['cache'][-1] or \
           layer_idx not in cache_dict['cache'][-1]['double_stream'] or \
           'total' not in cache_dict['cache'][-1]['double_stream'][layer_idx]:
            return False
            
        cached_features = cache_dict['cache'][-1]['double_stream'][layer_idx]['total']
        if not cached_features:
            return False
            
        # 计算余弦相似度
        if isinstance(cached_features, dict) and 0 in cached_features:
            cached_feat = cached_features[0]
        else:
            cached_feat = cached_features
            
        # 确保特征维度匹配
        if cached_feat.shape != current_features.shape:
            return False
            
        # 计算余弦相似度
        cos_sim = torch.nn.functional.cosine_similarity(
            current_features.reshape(1, -1), 
            cached_feat.reshape(1, -1)
        )
        similarity = cos_sim.item()
        
        # 更新层稳定性
        self.update_stability(layer_idx, similarity)
        
        return similarity > self.similarity_threshold

    def derivative_approximation(self, cache_dic: Dict, current: Dict, feature: torch.Tensor):
        """
        Compute derivative approximation.
        
        :param cache_dic: Cache dictionary
        :param current: Information of the current step
        """
        if len(current['activated_steps']) < 2:
            difference_distance = 1
        else:
            difference_distance = current['activated_steps'][-1] - current['activated_steps'][-2]

        updated_taylor_factors = {}
        updated_taylor_factors[0] = feature

        for i in range(cache_dic['max_order']):
            if (cache_dic['cache'][-1][current['stream']][current['layer']].get(current['module'], {}).get(i, None) is not None) and (current['step'] > cache_dic['first_enhance'] - 2):
                updated_taylor_factors[i + 1] = (updated_taylor_factors[i] - cache_dic['cache'][-1][current['stream']][current['layer']][current['module']][i]) / difference_distance
            else:
                break
        
        if current['stream'] not in cache_dic['cache'][-1]:
            cache_dic['cache'][-1][current['stream']] = {}
        if current['layer'] not in cache_dic['cache'][-1][current['stream']]:
            cache_dic['cache'][-1][current['stream']][current['layer']] = {}
            
        cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = updated_taylor_factors

    def taylor_formula(self, cache_dic: Dict, current: Dict) -> torch.Tensor: 
        """
        Compute Taylor expansion error.
        
        :param cache_dic: Cache dictionary
        :param current: Information of the current step
        """
        if len(current['activated_steps']) == 0:
            x = 0
        else:
            x = current['step'] - current['activated_steps'][-1]
            
        output = 0
        
        stream = current['stream']
        layer = current['layer']
        module = current['module']
        
        if -1 in cache_dic['cache'] and \
           stream in cache_dic['cache'][-1] and \
           layer in cache_dic['cache'][-1][stream] and \
           module in cache_dic['cache'][-1][stream][layer]:
            cache_data = cache_dic['cache'][-1][stream][layer][module]
            for i in range(len(cache_data)):
                output += (1 / math.factorial(i)) * cache_data[i] * (x ** i)
        
        return output

    def taylor_cache_init(self, cache_dic: Dict, current: Dict):
        """
        Initialize Taylor cache and allocate storage for different-order derivatives in the Taylor cache.
        
        :param cache_dic: Cache dictionary
        :param current: Information of the current step
        """
        if (current['step'] == 0):
            if -1 not in cache_dic['cache']:
                cache_dic['cache'][-1] = {}
            if current['stream'] not in cache_dic['cache'][-1]:
                cache_dic['cache'][-1][current['stream']] = {}
            if current['layer'] not in cache_dic['cache'][-1][current['stream']]:
                cache_dic['cache'][-1][current['stream']][current['layer']] = {}
            cache_dic['cache'][-1][current['stream']][current['layer']][current['module']] = {}

def afcache_flux_double_block_forward(
    self: FluxTransformerBlock,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
):
    
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
        encoder_hidden_states, emb=temb
    )
    joint_attention_kwargs = joint_attention_kwargs or {}

    cache_dic = joint_attention_kwargs['cache_dic']
    current = joint_attention_kwargs['current']

    # 初始化AFCache
    afcache = AdaptiveFeatureCache()
    
    if current['type'] == 'full':
        current['module'] = 'attn'
        afcache.taylor_cache_init(cache_dic=cache_dic, current=current)
        
        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs
            raise NotImplementedError("Not implemented for AFCache yet.") 

        # Process attention outputs for the `hidden_states`.
        current['module'] = 'img_attn'
        afcache.taylor_cache_init(cache_dic=cache_dic, current=current)

        afcache.derivative_approximation(cache_dic=cache_dic, current=current, feature=attn_output)
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        current['module'] = 'img_mlp'
        afcache.taylor_cache_init(cache_dic=cache_dic, current=current)
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        afcache.derivative_approximation(cache_dic=cache_dic, current=current, feature=ff_output)

        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output
        
        if len(attention_outputs) == 3:
            hidden_states = hidden_states + ip_attn_output

        # Process attention outputs for the `encoder_hidden_states`.
        current['module'] = 'txt_attn'
        afcache.taylor_cache_init(cache_dic=cache_dic, current=current)

        afcache.derivative_approximation(cache_dic=cache_dic, current=current, feature=context_attn_output)
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        current['module'] = 'txt_mlp'
        afcache.taylor_cache_init(cache_dic=cache_dic, current=current)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        afcache.derivative_approximation(cache_dic=cache_dic, current=current, feature=context_ff_output)

        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    elif current['type'] == 'Taylor':
        current['module'] = 'attn'
        # Attention.
        # symbolic placeholder

        # Process attention outputs for the `hidden_states`.
        current['module'] = 'img_attn'

        attn_output = afcache.taylor_formula(cache_dic=cache_dic, current=current)
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output
    
        current['module'] = 'img_mlp'

        ff_output = afcache.taylor_formula(cache_dic=cache_dic, current=current)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output
    
        # Process attention outputs for the `encoder_hidden_states`.
        current['module'] = 'txt_attn'

        context_attn_output = afcache.taylor_formula(cache_dic=cache_dic, current=current)

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output
    
        current['module'] = 'txt_mlp'

        context_ff_output = afcache.taylor_formula(cache_dic=cache_dic, current=current)

        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    elif current['type'] == 'Feature':
        # 特征相似性缓存
        current['module'] = 'attn'
        # 这里可以实现基于特征相似性的缓存逻辑
        # 为简化起见，我们暂时复用泰勒展开的逻辑
        # 实际实现中应该根据特征相似性来决定是否使用缓存
        
        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        if len(attention_outputs) == 2:
            attn_output, context_attn_output = attention_outputs
        elif len(attention_outputs) == 3:
            attn_output, context_attn_output, ip_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        current['module'] = 'img_attn'
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        current['module'] = 'img_mlp'
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output
        
        # Process attention outputs for the `encoder_hidden_states`.
        current['module'] = 'txt_attn'
        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        current['module'] = 'txt_mlp'
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states

def afcache_flux_single_block_forward(
    self: FluxSingleTransformerBlock,
    hidden_states: torch.FloatTensor,
    temb: torch.FloatTensor,
    image_rotary_emb=None,
    joint_attention_kwargs=None,
):
    joint_attention_kwargs = joint_attention_kwargs or {}
    cache_dic = joint_attention_kwargs['cache_dic']
    current = joint_attention_kwargs['current']

    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    gate = gate.unsqueeze(1)

    residual = hidden_states
    
    afcache = AdaptiveFeatureCache()
    
    if current['type'] == 'full':
        current['module'] = 'total'
        afcache.taylor_cache_init(cache_dic=cache_dic, current=current)

        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)

        hidden_states = self.proj_out(hidden_states)
        afcache.derivative_approximation(cache_dic=cache_dic, current=current, feature=hidden_states)

    elif current['type'] == 'Taylor':
        current['module'] = 'total'
        hidden_states = afcache.taylor_formula(cache_dic=cache_dic, current=current)
        
    elif current['type'] == 'Feature':
        current['module'] = 'total'
        # 特征相似性缓存逻辑
        # 为简化起见，暂时使用完整计算
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states = self.proj_out(hidden_states)

    hidden_states = gate * hidden_states
    hidden_states = residual + hidden_states

    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    return hidden_states

def cal_type_afcache(cache_dic, current):
    '''
    Determine calculation type for this step with AFCache strategy
    '''
    # 根据AFCache策略确定计算类型
    if (current['step'] < 3):
        # 前几个时间步总是完全计算
        current['type'] = 'full'
        current['activated_steps'].append(current['step'])
    else:
        # 使用自适应策略
        afcache = AdaptiveFeatureCache()
        # 简化实现：交替使用泰勒展开和特征缓存
        if current['step'] % 3 == 0:
            current['type'] = 'full'
            current['activated_steps'].append(current['step'])
        elif current['step'] % 3 == 1:
            current['type'] = 'Taylor'
        else:
            current['type'] = 'Feature'