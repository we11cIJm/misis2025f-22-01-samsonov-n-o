# misis2025f-22-01-samsonov-n-o

## C++ image processing | MISIS 3rd course

Сегментация КТ-срезов грецкого ореха на классы:
- `shell` — скорлупа
- `kernel` — ядро
- `septa` — внутренние перегородки

## Структура проекта

- `src/` — исходный код
- `dataset/Walnut2/Reconstructions/` — исходные TIFF-срезы
- `gt_final/` — ручная разметка `fdk_pos1_<N>_labels_gt.tiff`
- `results/` — результаты работы
- `CMakeLists.txt` — файл сборки

## Сборка

Требуется:
- CMake
- vcpkg
- OpenCV, установленный через vcpkg

Пример конфигурации под Windows:

```powershell
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=<path_to_vcpkg>\scripts\buildsystems\vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows-static
```

Сборка:

```powershell
cmake --build build --config Release
```

Установка:

```powershell
cmake --install build --config Release --prefix install
```

Запуск:

```powershell
.\install\bin\walnut_v1.exe --dataset .\dataset\Walnut2\Reconstructions --gt-dir .\gt_final --results .\results\v1

.\install\bin\walnut_v2.exe --dataset .\dataset\Walnut2\Reconstructions --gt-dir .\gt_final --results .\results\v2
```

Метрики:

```powershell
.\install\bin\walnut_metrics.exe --pred-root .\results\v1 --gt-root .\gt_final --csv .\results\v1\metrics.csv --tol 2

.\install\bin\walnut_metrics.exe --pred-root .\results\v2 --gt-root .\gt_final --csv .\results\v2\metrics.csv --tol 2
```

Error map:

```powershell
.\install\bin\walnut_error_map.exe --pred-root .\results\v1 --gt-root .\gt_final --out-root .\results\v1\error_maps

.\install\bin\walnut_error_map.exe --pred-root .\results\v2 --gt-root .\gt_final --out-root .\results\v2\error_maps
```