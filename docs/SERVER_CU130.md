# PyTorch 2.10 / CUDA 13 server profile

Target image: `pytorch2.10.0-cuda13.0-vllm0.19.0-py312`, on Linux with Python 3.12.
Run commands from the project root. All added profile/report paths are relative.
The supplied Torch/CUDA/vLLM binaries should be retained.

## Dependencies

Install the other dependencies yourself using
`requirements/server-cu130-py312.constraints.txt`:

| Package | Version |
| --- | --- |
| torch / torchaudio | 2.10.0, CUDA 13.0 build from the image |
| torchvision | 0.25.0, matching the Torch build |
| vllm | 0.19.0, from the image |
| torchcodec | 0.10.0 |
| deepspeed | 0.19.8 |
| accelerate | 1.12.0 |
| transformers | 4.57.6 |
| huggingface-hub | 0.36.0 |
| safetensors | 0.5.3 |
| datasets | 3.6.0 |
| numpy | 2.2.6 |
| packaging | 25.0 (also satisfies the supplied image's DALI <=25 requirement) |
| scikit-learn | 1.6.1 (replace the image's NumPy-1-only 1.3.2 wheel) |

The base metadata now admits the original and server dependency profiles.
Use explicit constraints rather than relying on unconstrained latest versions.
The original environment remains available through
`requirements/torch271.constraints.txt`.

Version choices are based on primary upstream metadata:

- [vLLM 0.19 CUDA requirements](https://github.com/vllm-project/vllm/blob/v0.19.0/requirements/cuda.txt)
  pin Torch 2.10.0, torchvision 0.25.0 and torchaudio 2.10.0.
- [TorchCodec's compatibility matrix](https://github.com/meta-pytorch/torchcodec#installing-torchcodec)
  pairs TorchCodec 0.10 with Torch 2.10.
- [vLLM common requirements](https://github.com/vllm-project/vllm/blob/v0.19.0/requirements/common.txt)
  require Transformers >=4.56,<5 and OpenCV-headless >=4.13.
- [OpenCV-headless 4.13 package metadata](https://pypi.org/pypi/opencv-python-headless/4.13.0.92/json)
  requires NumPy >=2 for Python >=3.9. NumPy 2.2.6 also meets vLLM's
  [numba 0.61.2](https://pypi.org/pypi/numba/0.61.2/json) constraint of NumPy <2.3.
- [DeepSpeed 0.19.8 requirements](https://github.com/deepspeedai/DeepSpeed/blob/v0.19.8/requirements/requirements.txt)
  admit this Torch version. A real CUDA/ZeRO probe is still required on the server.

Do not install the old Torch 2.7.1 / TorchCodec 0.5 pins into this image. Keep
simulator dependencies in their existing separate environments; this profile
is for Phoenix policy training, not the legacy simulation extras.

## Installation and checks

The three Hugging Face packages alone are not the full training environment.
The complete direct training/logging package list is in
[`requirements/server-cu130-py312.txt`](../requirements/server-cu130-py312.txt).
It includes all mandatory project dependencies and the `train,logging` extras,
apart from Torch and torchvision, which are already supplied by the base image.
Transitive packages such as `tokenizers` are installed by the resolver.

For the supplied Dockerfile dependency block, keep `RUN uv pip install --system`
unchanged and use this complete list (run inside the supplied base image):

```dockerfile
RUN uv pip install --system \
    "transformers==4.57.6" \
    "huggingface-hub==0.36.0" \
    "safetensors==0.5.3" \
    "accelerate==1.12.0" \
    "deepspeed==0.19.8" \
    "torchcodec==0.10.0" \
    "datasets==3.6.0" \
    "numpy==2.2.6" \
    "av>=16,<18" \
    "einops>=0.8.1,<0.9" \
    "gitpython>=3.1.45,<4" \
    "hydra-core>=1.3.2,<1.4" \
    "omegaconf>=2.3,<2.4" \
    "imageio>=2.37,<3" \
    "imageio-ffmpeg==0.6.0" \
    "jsonlines==4.0.0" \
    "packaging==25.0" \
    "pandas>=2.2.3,<3" \
    "pillow>=12,<13" \
    "pyarrow>=23,<26" \
    "PyYAML==6.0.3" \
    "scipy>=1.15,<2" \
    "scikit-learn==1.6.1" \
    "regex>=2025.11.3" \
    "rich>=14.2,<16" \
    "termcolor>=2.5,<4" \
    "tqdm>=4.66.5,<5" \
    "typing-extensions>=4.15" \
    "wandb>=0.23.1,<1"
```

If the requirements folder has already been copied into the image, the same
list can also be installed as `uv pip install --system -r requirements/server-cu130-py312.txt`.
Do not add the `eval` simulator extras for this Piper training task. `modelscope`
is optional for downloading model resources; installing it is not required to
train from the already downloaded local weights.

vLLM 0.19 requires Transformers >=4.56,<5, so the previous 4.53.2 pin
conflicts. [Transformers 4.57.6's dependency metadata](https://github.com/huggingface/transformers/blob/v4.57.6/setup.py)
requires huggingface-hub >=0.34,<1 and safetensors >=0.4.3. The chosen Hub
version satisfies both this requirement and Phoenix's >=0.33 requirement;
the supplied safetensors 0.5.3 pin can be retained. If more packages follow,
keep the trailing continuation backslash on the safetensors line as well.

After installing the listed packages in the server environment, register the
project without asking pip to replace the image's installed dependencies:

```bash
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python tools/check_server_runtime.py
```

Alternatively, the optional installer resolves project dependencies under both
the profile and a snapshot of the image's existing compiled binaries. It first
checks pip's installation plan and refuses any replacement of protected
Torch/vision/audio/vLLM/Triton/FlashInfer/NVIDIA packages:

```bash
bash scripts/install_server_env.sh --dry-run
bash scripts/install_server_env.sh
```

No installer or CUDA probe has been run on your server by this change. The
installer uses the image's build environment to avoid compiling DeepSpeed
against a different isolated Torch. Native build tools and the appropriate
CUDA toolkit may be needed by DeepSpeed JIT operations.

## Repairing the reported preinstalled image conflicts

The actual server inventory contains scikit-learn 1.3.2, which requires NumPy
<2. Its binary import through Transformers reproduced the reported
`numpy.dtype size changed: Expected 96 ... got 88` error. Replacing it with
scikit-learn 1.6.1 preserves NumPy 2.2.6. Packaging 25.0 is the intersection
between Phoenix's >=25 requirement and the image's DALI <=25 requirement.

The user's image also mixes OpenTelemetry 1.41.1 exporters with 1.45.1 SDK/HTTP
components, OpenAI 1.x with vLLM >=2 requirements, and old TensorBoardX,
Elasticsearch transport and Google API packages with newer Protobuf/urllib3.
The targeted repair versions are in
[`requirements/server-image-repair.txt`](../requirements/server-image-repair.txt).

That complete repair removes only the obsolete **Google Cloud TPU client**
`cloud-tpu-client==0.10`, which pins google-api-python-client==1.8.0; that chain
requires protobuf<4 and cannot coexist with vLLM's protobuf>=5.29.6. The NVIDIA
training task does not use this TPU client. Google API clients and the other
reported conflicting tools are then updated together. GPU distributions are
preserved using the image's actual versions, including its vendor Triton build:

```bash
mkdir -p resources/compat
python -m pip freeze > resources/compat/server-before-repair.txt
python tools/check_server_runtime.py --base-only \
  --write-image-constraints resources/compat/server-image.constraints.txt \
  --output resources/compat/server-base.json
python -m pip uninstall -y cloud-tpu-client
python -m pip install \
  -c resources/compat/server-image.constraints.txt \
  -r requirements/server-image-repair.txt
python -m pip check
CUDA_VISIBLE_DEVICES=0 python tools/check_server_runtime.py --cuda-smoke
```

These commands must be run on the server, with the updated requirements files.
The repair profile resolved against the known public dependency requirements;
it has not been applied to the user's server by this change. Keep the freeze
snapshot and review any additional conflicts from private image packages.
The runtime checker now records the failing phase/import and full traceback.

Server operator checks, using small synthetic tensors and no model weights:

```bash
OMP_NUM_THREADS=1 python tools/check_server_runtime.py --cuda-smoke
python tools/check_piper_data.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

The CUDA probe compares the unchanged masked Joint96 FlexAttention forward and
backward against a dense mathematical reference, including an empty-key sample.
It then runs one tiny BF16/AdamW/ZeRO-1 step using the original DeepSpeed config
with only its `auto` batch fields resolved for the probe. It does not start a
Phoenix training run or save training weights.

Runtime results are written to `resources/compat/server-runtime.json`. The
default command checks exact versions, Torch/vision native operator loading,
training imports and the selected existing video backend. `--cpu-smoke` is a
separate API test; passing it does not certify CUDA, NCCL, DeepSpeed GPU kernels
or the user's actual server image.

## Verification performed during adaptation

On Python 3.12, the training/logging requirements resolved successfully both
alone (112 packages) and together with vLLM 0.19.0 (208 packages). These are
metadata resolutions, not validation of the image's custom CUDA binaries;
the generated resolver output under `resources/compat/` is diagnostic only.

An isolated CPU environment with Torch 2.10.0 / torchvision 0.25.0 / NumPy 2.2.6
passed 15 data-interface and dependency-protection tests. All 448,712 prepared
frames were checked against their source records, and six real RGB windows
decoded successfully. The exact Transformers 4.57.6 / Hub 0.36.0 / safetensors
0.5.3 combination passed local T5 tokenizer loading, a BF16 safetensors
roundtrip, and masked FlexAttention forward/backward checks. These CPU checks
do not verify DeepSpeed or CUDA kernels. Integrity checks found no differences
in the 391 existing source/config/script files captured before the version
adaptation work.

The optional tokenizer check also found a truncated local `tokenizer.json`
(11,677,696 bytes). It was backed up under `resources/compat/` and replaced
with the project's official Wan tokenizer file (16,837,417 bytes), verified
against its published LFS SHA-256
`6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b`.
This repairs a resource file and does not change the tokenizer configuration
or project code.

## Preserved behavior

No training-time compatibility monkeypatch is installed. Model/encoder architecture,
action/state mapping, camera ordering, temporal sampling, masks, noise schedules,
losses, optimizer settings, batch size, distributed recipe and checkpoint
configuration loading flags remain unchanged. The separately authorized Piper
initialization entry now strictly restores the existing Phoenix checkpoint
without loading Wan DiT base shards. The existing strict tensor/metadata checks
still run before optimizer creation. vLLM is retained as an image component;
Phoenix continues to use its original Torch training implementation.

Dependency upgrades can change floating-point roundoff and generated kernels.
Preserving project logic does not assert bit-for-bit equality across Torch/CUDA
versions. Run the server probes before starting the existing `piper_posttrain`
task; the two task T5 caches still need to be prepared on the server. Wan VAE,
DINO and CSVAE remain required; official Wan DiT base shards are not required
by the Piper checkpoint-only initialization entry.
