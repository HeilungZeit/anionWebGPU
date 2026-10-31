# Веса, которые стояли в anion до Э23

Копия `anion/public/upscale/` на 2026-10-31 (anion `f01dcea`) — для отката,
если новые сети (`anion-u48-4`, `anion-u32-2`, Restore внутри, Э23) не
понравятся:

- `anion-u48-3` — «Максимальное» (Э20);
- `anion-u32-1` — «Сильное» (Э19).

Откат: скопировать каталог в `anion/public/upscale/` (если его там уже нет)
и вернуть имя в `COMPACT_WEIGHTS` в
`anion/src/app/components/custom/anime-page/video-player/upscale.ts`. Ядро то
же — выпуск anionWebGPU для отката не нужен.

Лицензии — в `LICENSE.txt` рядом с весами (CC BY-NC-SA 4.0, только
некоммерческое использование).
