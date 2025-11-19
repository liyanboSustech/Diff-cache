import os

from diffusers.models import FluxTransformer2DModel


def _get_env_int(name: str):
    value = os.getenv(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        print(f"[TaylorSeer] Ignoring invalid environment variable {name}={value}")
        return None


def _apply_env_overrides(cache_dic):
    """Allow overriding key TaylorSeer parameters via env vars for sweeps."""
    max_order = _get_env_int("TAYLORSEER_MAX_ORDER")
    if max_order is not None:
        cache_dic['max_order'] = max(0, max_order)

    first_enhance = _get_env_int("TAYLORSEER_FIRST_ENHANCE")
    if first_enhance is not None:
        cache_dic['first_enhance'] = max(0, first_enhance)

    fresh_threshold = _get_env_int("TAYLORSEER_FRESH_THRESHOLD")
    if fresh_threshold is not None:
        cache_dic['fresh_threshold'] = max(0, fresh_threshold)
def cache_init(self: FluxTransformer2DModel):   
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache_index = {}
    cache[-1]={}
    cache_index[-1]={}
    cache_index['layer_index']={}
    cache_dic['attn_map'] = {}
    cache_dic['attn_map'][-1] = {}
    cache_dic['attn_map'][-1]['double_stream'] = {}
    cache_dic['attn_map'][-1]['single_stream'] = {}

    cache[-1]['double_stream']={}
    cache[-1]['single_stream']={}
    cache_dic['cache_counter'] = 0

    #for j in range(19):
    for j in range(self.config.num_layers):
        cache[-1]['double_stream'][j] = {}
        cache_index[-1][j] = {}
        cache_dic['attn_map'][-1]['double_stream'][j] = {}
        cache_dic['attn_map'][-1]['double_stream'][j]['total'] = {}
        cache_dic['attn_map'][-1]['double_stream'][j]['txt_mlp'] = {}
        cache_dic['attn_map'][-1]['double_stream'][j]['img_mlp'] = {}

    #for j in range(38):
    for j in range(self.config.num_single_layers):
        cache[-1]['single_stream'][j] = {}
        cache_index[-1][j] = {}
        cache_dic['attn_map'][-1]['single_stream'][j] = {}
        cache_dic['attn_map'][-1]['single_stream'][j]['total'] = {}

    cache_dic['taylor_cache'] = False
    cache_dic['Delta-DiT'] = False

    mode = 'Taylor'

    if mode == 'original':
        cache_dic['cache_type'] = 'random' 
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa'
        cache_dic['fresh_ratio'] = 0.0
        cache_dic['fresh_threshold'] = 1
        cache_dic['force_fresh'] = 'global'
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = 3
        
    elif mode == 'ToCa':
        cache_dic['cache_type'] = 'attention'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa' 
        cache_dic['fresh_ratio'] = 0.1
        cache_dic['fresh_threshold'] = 5
        cache_dic['force_fresh'] = 'global' 
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = 3
    
    elif mode == 'Taylor':
        cache_dic['cache_type'] = 'random'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa' 
        cache_dic['fresh_ratio'] = 0.0
        cache_dic['fresh_threshold'] = 6
        cache_dic['force_fresh'] = 'global' 
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['taylor_cache'] = True
        cache_dic['max_order'] = 4
        cache_dic['first_enhance'] = 3

    elif mode == 'Delta':
        cache_dic['cache_type'] = 'random'
        cache_dic['cache_index'] = cache_index
        cache_dic['cache'] = cache
        cache_dic['fresh_ratio_schedule'] = 'ToCa' 
        cache_dic['fresh_ratio'] = 0.0
        cache_dic['fresh_threshold'] = 3
        cache_dic['force_fresh'] = 'global' 
        cache_dic['soft_fresh_weight'] = 0.0
        cache_dic['Delta-DiT'] = True
        cache_dic['max_order'] = 0
        cache_dic['first_enhance'] = 1

    current = {}
    current['activated_steps'] = [0]
    current['step'] = 0
    current['num_steps'] = self.num_steps

    _apply_env_overrides(cache_dic)

    return cache_dic, current
