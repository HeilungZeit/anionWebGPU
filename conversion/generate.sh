#!/bin/sh
# Перегенерация CNN-моделей, которые использует anion. Запуск из корня репозитория.
set -e
G=conversion/glsl/anime4k
# VL и L — с упаковкой пар слоёв в rgba32uint (Э9): M1 Pro f32 1.04–1.16×.
CNN_PACK=1 python3 conversion/cnn.py "$G/Anime4K_Upscale_Denoise_CNN_x2_VL.glsl" src/pipelines/upscale/DenoiseCNNx2VL DenoiseCNNx2VL
python3 conversion/cnn.py "$G/Anime4K_Upscale_CNN_x2_M.glsl" src/pipelines/upscale/CNNx2M CNNx2M
python3 conversion/cnn.py "$G/Anime4K_Restore_CNN_M.glsl" src/pipelines/restore/CNNM CNNM
python3 conversion/cnn.py "$G/Anime4K_Upscale_Denoise_CNN_x2_M.glsl" src/pipelines/upscale/DenoiseCNNx2M DenoiseCNNx2M
CNN_PACK=1 python3 conversion/cnn.py "$G/Anime4K_Upscale_Denoise_CNN_x2_L.glsl" src/pipelines/upscale/DenoiseCNNx2L DenoiseCNNx2L
A=conversion/glsl/artcnn
python3 conversion/artcnn.py "$A/ArtCNN_C4F16_DS.glsl" src/pipelines/artcnn/ArtCNNC4F16DS ArtCNNC4F16DS
python3 conversion/artcnn.py "$A/ArtCNN_C4F32_DS.glsl" src/pipelines/artcnn/ArtCNNC4F32DS ArtCNNC4F32DS
