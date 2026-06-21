cd /autodl-fs/data/llmtexture/step2

/root/miniconda3/envs/castex/bin/python src/apply_uv_offset_to_obj.py \
  --mesh outputs/offset/test_tshirt_jeans_init/offset_mesh.obj \
  --uv data/smplx/generated/uv_data.npz \
  --offset-uv outputs/offset/test_tshirt_jeans_init/offset_scale_uv.png \
  --out-dir outputs/apply_uv_offset_test_tshirt_jeans \
  --part-labels data/smplx/generated/part_labels.json \
  --max-vertex-offset 0.08 \
  --out-name preview_offset_mesh.obj
