# Inference deployment

## TorchScript / checkpoint driver

The primary inference path uses a trained PyTorch checkpoint:

```bash
rl-uco-infer --ir function.ll --checkpoint checkpoints/best.pt --output optimized.ll
```

The engine:

1. Encodes IR as a PyG graph
2. Samples a pass sequence from the policy
3. Validates against [`rl_uco/passes/registry.yaml`](../rl_uco/passes/registry.yaml)
4. Falls back to `-O3` pipeline if invalid
5. Runs `opt` via [`PassExecutor`](../rl_uco/passes/executor.py)

## External opt driver

For integration without Python in the hot path, wrap the driver script:

```bash
python infra/inference/opt_driver.py \
  --ir input.ll \
  --checkpoint checkpoints/best.pt \
  -o output.ll
```

## LLVM plugin (future)

A native `PluginPass` can call TorchScript C++ API with the same checkpoint format saved via:

```python
agent = ActorCriticAgent(...)
torch.jit.script(agent.encoder)
```

Current release uses the Python driver for portability.

## CI / batch

Run inference over a corpus manifest:

```bash
for ll in data/corpus/*/*.ll; do
  rl-uco-infer --ir "$ll" --checkpoint checkpoints/best.pt --output "out/$(basename $ll)"
done
```
