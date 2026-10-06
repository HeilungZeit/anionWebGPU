# Исходники шейдеров

Только то, из чего генерируются модели форка (`conversion/generate.sh`).
Файлы не править: это копии оригиналов.

| Папка | Откуда | Коммит | Лицензия |
| --- | --- | --- | --- |
| `anime4k/` | https://github.com/bloc97/Anime4K (`glsl/`) | `7684e95` (2024-04-25) | MIT |
| `artcnn/` | https://github.com/Artoriuz/ArtCNN (`GLSL/`) | `028ed76` (2026-10-04) | MIT |

`Anime4K_Clamp_Highlights.glsl` генератором не читается — это образец для
ручной реализации `helpers/ClampHighlights`.

Другие модели Anime4K (A/B, UL, GAN) сгенерированы старым
`conversion/shader.py` ещё в апстриме; их исходники — в репозитории Anime4K.
