## vLLM专家卸载NPU适配

### 1. 仓库分支
- `vllm`: expert-offload
- `vllm-ascend`: expert-offload

### 2. 代码改动
- `vllm`: 将[PR #37190](https://github.com/vllm-project/vllm/pull/37190)的改动cherry-pick到`releases/v0.18.0`分支，并修改`ExpertWeightProvider`实现分块专家加载
- `vllm-ascend`: 修改moe层，添加专家加载和映射逻辑

### 3. 测试方法
- 命令行：添加`--moe-expert-cache-size`参数，设置每层缓存专家数量，并指定`--enforce-eager`禁用计算图
- 代码调用：在`LLM`实例化时传入`moe_expert_cache_size`参数，并设置`enforce_eager=True`