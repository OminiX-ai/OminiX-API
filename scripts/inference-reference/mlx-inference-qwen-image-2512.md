
# mlx推理Qwen-Image-2512模型

步骤：

1. 先使用uv或者conda创建Python虚拟环境
2. 安装mflux
3. 使用命令进行推理

推理可参考：https://huggingface.co/mlx-community/Qwen-Image-2512-4bit

系统：macOS

以下是在 macOS 系统上使用 MLX 框架和 `mflux` 工具对 `Qwen-Image-2512-4bit` 模型进行推理的完整指南 。该模型专为高质量文本到图像生成而设计，并在 Apple Silicon 设备上进行了本地运行优化 。

## 创建 Python 虚拟环境

为了避免依赖冲突，建议为该项目创建一个独立的 Python 虚拟环境。您可以根据个人偏好选择使用 `uv` 或 `conda`。

**使用 uv 创建环境：**
`uv` 是一个极快的 Python 包和环境管理器。打开终端，使用以下命令即可完成环境的创建和激活：

```bash
# 创建虚拟环境
uv venv

# 激活虚拟环境
source .venv/bin/activate
```

**使用 conda 创建环境：**
如果您更习惯使用 Anaconda 或 Miniconda，可以通过以下命令创建并激活指定 Python 版本的环境：

```bash
# 创建虚拟环境（建议 Python 3.10 或以上）
conda create -n mflux-env python=3.10 -y

# 激活虚拟环境
conda activate mflux-env
```


## 安装 mflux

`mflux` 是专门基于苹果 MLX 框架开发的库，用于在 Mac 本地高效运行包括 Qwen 和 FLUX 在内的先进图像生成模型 。在已激活的虚拟环境中，执行以下命令进行安装：[^3][^4]

```bash
pip install mflux
```


## 使用命令进行推理

环境配置完毕后，可以通过专门的 `mflux-generate-qwen` 命令直接在终端中进行文本到图像的推理 。首次执行该命令时，系统会自动从 Hugging Face 仓库下载所需的量化模型权重文件 。[^5][^2][^3]

运行以下测试命令以生成图像：

```bash
mflux-generate-qwen \
  --model mlx-community/Qwen-Image-2512-4bit \
  --prompt "A photorealistic cat wearing a tiny top hat" \
  --steps 20
```


### 常用推理参数说明

- `--model`：指定模型所在的 Hugging Face 仓库名称，此处使用的是官方提供的 4-bit 量化版本 。[^2]
- `--prompt`：输入所需生成图像的英文提示词 。[^2]
- `--steps`：设置生成图像的推理步数，通常设定为 20 步以平衡图像质量与生成速度 。[^2]
- `--seed`：（可选）设置随机种子数，以便在后续生成中复现完全相同的图像 。[^6][^4]


[^1]: https://skywork.ai/blog/models/qwen-image-2512-gguf-free-image-generate-online/

[^2]: https://huggingface.co/mlx-community/Qwen-Image-2512-4bit

[^3]: https://www.reddit.com/r/LocalLLaMA/comments/1q0wkwc/qwenimage2512_mflux_port_available_now/

[^4]: https://pypi.org/project/mflux/0.14.0/

[^5]: https://x.com/rohanpaul_ai/status/1832559995625361559

[^6]: https://github.com/filipstrand/mflux

[^7]: https://github.com/filipstrand/mflux/issues

[^8]: https://github.com/filipstrand/mflux/pulls

[^9]: https://huggingface.co/mlx-community/Qwen-Image-2512-6bit

[^10]: https://github.com/fblissjr/mflux-annotated

