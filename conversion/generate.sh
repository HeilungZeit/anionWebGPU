#!/bin/sh
# Перегенерация CNN-моделей, которые использует anion. Запуск из anionWebGPU/.
set -e
G=../upstream-anime4k/glsl
python3 conversion/cnn.py "$G/Upscale+Denoise/Anime4K_Upscale_Denoise_CNN_x2_VL.glsl" src/pipelines/upscale/DenoiseCNNx2VL DenoiseCNNx2VL
python3 conversion/cnn.py "$G/Upscale/Anime4K_Upscale_CNN_x2_M.glsl" src/pipelines/upscale/CNNx2M CNNx2M
python3 conversion/cnn.py "$G/Restore/Anime4K_Restore_CNN_M.glsl" src/pipelines/restore/CNNM CNNM
python3 conversion/cnn.py "$G/Upscale+Denoise/Anime4K_Upscale_Denoise_CNN_x2_M.glsl" src/pipelines/upscale/DenoiseCNNx2M DenoiseCNNx2M
python3 conversion/cnn.py "$G/Upscale+Denoise/Anime4K_Upscale_Denoise_CNN_x2_L.glsl" src/pipelines/upscale/DenoiseCNNx2L DenoiseCNNx2L
A=../upstream-artcnn/GLSL
python3 conversion/artcnn.py "$A/ArtCNN_C4F16_DS.glsl" src/pipelines/artcnn/ArtCNNC4F16DS ArtCNNC4F16DS
python3 conversion/artcnn.py "$A/ArtCNN_C4F32_DS.glsl" src/pipelines/artcnn/ArtCNNC4F32DS ArtCNNC4F32DS
