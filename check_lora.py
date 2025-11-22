from safetensors.torch import load_file
import sys

lora_file = sys.argv[1] if len(sys.argv) > 1 else "chani_i2v.lora.safetensors"
sd = load_file(lora_file)
keys = list(sd.keys())

print(f"Total keys: {len(keys)}")
print("\nFirst 20 keys:")
for k in keys[:20]:
    print(f"  {k}")

print(f"\nSample key structure: {keys[0] if keys else 'No keys'}")
print(f"\nKey prefix pattern: {keys[0].split('.')[0] if keys else 'N/A'}")

# Check for common patterns
has_lora_unet = any("lora_unet" in k for k in keys)
has_blocks = any("blocks" in k for k in keys)
print(f"\nHas 'lora_unet' prefix: {has_lora_unet}")
print(f"Has 'blocks' in keys: {has_blocks}")

