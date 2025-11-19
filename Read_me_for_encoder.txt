Title: Using the Point Cloud Encoder in Training

Required files and locations

Put the Michelangelo config file here:
configs/michelangelo_point_encoder_cfg.yaml

Put the Michelangelo checkpoint here:
checkpoints/michelangelo_point_encoder_state_dict.pt

The training code will read these two paths from the model config:

miche_encoder_cfg_path

miche_encoder_sd_path

Also set point_num_points in the config (for example, 2048).

What the point encoder expects

The dataset should provide point clouds as a float tensor with shape:
(B, N, C)
where:

B = batch size

N = number of points

C = 3 (xyz only) or 6 (xyz + normals)

Michelangelo will internally:

convert xyz → xyz+normals if needed,

sample or pad to point_num_points,

output a single embedding per shape of size (B, 1, D).

What the projector does

The point projector takes the Michelangelo output (B, 1, D) and maps it to the LLM hidden size (B, 1, H).

This “point token” is then concatenated with text tokens (and image tokens, if used) before going into the decoder.

Changes needed in the config

In the model config (CADMLLMConfig), set at least:

miche_encoder_cfg_path = "configs/michelangelo_point_encoder_cfg.yaml"

miche_encoder_sd_path = "checkpoints/michelangelo_point_encoder_state_dict.pt"

point_num_points = 2048 (or the value you trained Michelangelo with)

Make sure projector settings are reasonable, for example:

projector_hidden_dim (e.g. 1024)

projector_num_layers (e.g. 2)

projector_dropout (e.g. 0.0)

Changes needed in the dataset

The dataset should now return point clouds in each sample.

The collator should stack them into a batch field called point_clouds.

The final batch dictionary given to the model should include:

input_ids

attention_mask

labels

point_clouds (optional but present if you want the point encoder)

Changes needed in the training loop

After creating the model, enable use of the point encoder by setting:

model.has_point_encoder = True

When calling the model in the training loop, pass the point clouds:

point_clouds=batch["point_clouds"]

The forward call should include:

input_ids

attention_mask

labels

pixel_values (if using images)

point_clouds (for the point encoder)

Data type notes

Point clouds should be float tensors (float32 is safest).

If the LLM and projectors use mixed precision (e.g. bfloat16), the model code should cast the point embeddings to the same dtype internally.

If you want to avoid dtype issues, you can simply run everything in float32.

Once these pieces are in place (files in the right folders, dataset returning point_clouds, model.has_point_encoder = True, and point_clouds passed into the model), the decoder is trained conditionally on text + point cloud (and image, if enabled).