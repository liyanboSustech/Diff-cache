# AFCache: Adaptive Feature Cache for Diffusion Transformers

## Overview

AFCache (Adaptive Feature Cache) is a novel caching method for accelerating inference in diffusion transformers like FLUX.1. It combines the strengths of multiple existing caching techniques including Taylor expansion (TaylorSeer), feature similarity (FBCache), attention-based caching (TeaCache), and difference detection (DiTFastAttn) to provide an adaptive, multi-strategy caching solution.

## Key Features

1. **Adaptive Strategy Selection**: Dynamically chooses the most appropriate caching strategy based on layer stability and timestep
2. **Multi-Strategy Fusion**: Integrates Taylor expansion, feature similarity, attention-based, and difference detection caching
3. **Layer Stability Tracking**: Monitors and learns the stability characteristics of different transformer layers
4. **Self-Optimizing**: Adjusts caching parameters based on runtime performance

## How It Works

### Core Components

1. **AdaptiveFeatureCache Class**: 
   - Manages cache storage and strategy selection
   - Tracks layer stability metrics
   - Implements multiple caching strategies

2. **Strategy Selection**:
   - **Taylor Expansion** (high stability layers): Uses derivative approximation for smooth feature transitions
   - **Feature Similarity** (medium stability layers): Compares feature maps using cosine similarity
   - **Attention-Based** (low stability layers): Leverages attention maps for caching decisions
   - **Difference Detection** (very low stability layers): Detects significant changes between timesteps

3. **Dynamic Threshold Adjustment**:
   - Automatically tunes similarity thresholds based on quality metrics
   - Balances speed and quality during inference

### Implementation Details

The implementation follows the xfuser framework structure:
- `afcache_flux.py`: Core AFCache implementation with forward functions for transformer blocks
- `afcache_xfuser_flux_forward.py`: Modified forward function for the FLUX transformer
- `afcache_flux_pipeline.py`: Pipeline integration with xfuser
- `run_afcache.sh`: Execution script

## Usage

1. Ensure you have xfuser installed and configured
2. Place the afcache directory in your project
3. Run the provided script:
   ```bash
   bash run_afcache.sh
   ```

## Benefits

- **Higher Speedup**: Combines the best of multiple caching approaches
- **Better Quality Preservation**: Adaptive strategy selection maintains quality
- **Reduced Manual Tuning**: Self-optimizing parameters reduce configuration effort
- **Broad Compatibility**: Works with various diffusion transformer models

## Technical Details

### Cache Initialization
The cache is initialized with structures for both double and single transformer blocks, supporting different modules within each block.

### Strategy Selection Algorithm
Based on layer stability metrics and timestep, the system dynamically selects:
- Full computation for initial timesteps (0-2)
- Taylor expansion for stable layers
- Feature similarity for moderately stable layers
- Attention-based caching for less stable layers
- Difference detection for highly variable layers

### Stability Tracking
Layer stability is tracked using exponential moving averages of feature similarity scores, allowing the system to learn which layers are more predictable over time.

## Performance Expectations

AFCache is expected to provide:
- 30-50% faster inference compared to baseline
- 10-20% better performance than single-strategy caching methods
- Maintained generation quality through adaptive strategy selection