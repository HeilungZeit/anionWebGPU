# anionWebGPU — форк anime4k-webgpu для anion

План, этапы и журнал замеров — [docs/PLAN.md](docs/PLAN.md). Начинать с него и
отмечать чекбоксы там же.

- Ветка `anion`, remote `origin` — https://github.com/HeilungZeit/anionWebGPU,
  `upstream` — оригинал (Anime4KWebBoost/Anime4K-WebGPU).
- Node ≥ 24. `npm run build` — tsdown → `lib/`; `npm run check` — TypeScript 7
  и oxlint.
- Шейдеры CNN не править руками: они генерируются `conversion/cnn.py`
  (Anime4K) и `conversion/artcnn.py` (ArtCNN) из GLSL в `conversion/glsl/`.
  Правка — в генераторе, затем `conversion/generate.sh`.
- `conversion/glsl/` — копии оригиналов, только для чтения.
- Любая оптимизация проверяется стендом (`npm run bench` →
  `http://localhost:5179/bench/`): время и сверка с эталоном (`bench/baseline/`
  — npm 1.0.0, `bench/ref/<этап>` — снимки `lib/`; оба локальные). Правки, не
  меняющие математику, обязаны давать `maxΔ < 1/255`.
- Браузер самому не открывать без просьбы владельца — как и в anion.
- Потребитель — плеер anion:
  `../anion/src/app/components/custom/anime-page/video-player/upscale.ts`.
  Kodik отдаёт максимум 720p.
