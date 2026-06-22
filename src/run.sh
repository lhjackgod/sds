cd /autodl-fs/data/llmtexture/step2

# /root/miniconda3/envs/castex/bin/python src/apply_uv_offset_to_obj.py \
#   --mesh outputs/offset/test_tshirt_jeans_init/offset_mesh.obj \
#   --uv data/smplx/generated/uv_data.npz \
#   --offset-uv outputs/offset/test_tshirt_jeans_init/offset_scale_uv.png \
#   --out-dir outputs/apply_uv_offset_test_tshirt_jeans \
#   --part-labels data/smplx/generated/part_labels.json \
#   --max-vertex-offset 0.08 \
#   --out-name preview_offset_mesh.obj

/root/miniconda3/envs/castex/bin/python src/main_optimize_uv_offset_sds.py \
  --prompt "a person wearing a loose white hoodie and dark blue jeans" \
  --mesh data/smplx/generated/smplx_template.obj \
  --uv data/smplx/generated/uv_data.npz \
  --part-labels data/smplx/generated/part_labels.json \
  --mask-dir outputs/masks/prompt_batch_512/04_a_person_wearing_a_white_hoodie_and_dark_jeans \
  --castex-root ../CasTex \
  --out outputs/offset_sds/hoodie_jeans_sds_part_test \
  --optimize-mode part_scale \
  --stage i \
  --steps 100 \
  --batch-size 1 \
  --render-resolution 256 \
  --guidance-scale 15 \
  --device cuda:0 \
  --save-interval 20