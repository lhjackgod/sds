cd /autodl-fs/data/llmtexture/step2

/root/miniconda3/envs/castex/bin/python src/generate_smplx_template.py \
  --model-path data/smplx/source/models \
  --uv-template data/smplx/source/smplxuv/smplx_uv_2023.obj \
  --out data/smplx/generated