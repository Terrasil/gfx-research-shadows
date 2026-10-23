# gfx-research-shadows

Laboratorium cieni C++20 / OpenGL 4.6 na [gfx-research-base](https://github.com/Terrasil/gfx-research-base), rewizja `a419073ff7c5fd781ef52a4ff9fcdb0c67c5275c`. Wersja 0.12.0 udostępnia wspólne menu `Smoothing` dla PCSS mean blocker, geometric i fixed PCF. Dodaje też presety losowych próbek bezpośrednio w tym menu. Badamy nakładanie ostrych cieni bliskich przeszkód i szerokich półcieni dalszych obiektów. Backprojection usunięto.

| ID CLI | Metoda |
| --- | --- |
| `hard` | Hard shadow |
| `pcf` | Fixed PCF, stały promień w tekslach mapy |
| `pcss` | PCSS mean blocker, średnia arytmetyczna odległości, z wyborem wygładzania |
| `arr` | Angular ray reference, przecięcia promieni z bryłami AABB |
| `geometric` | PCSS geometric mean z wygładzaniem, domyślna metoda |
| `dpcf` | Pełny filtr kierunkowy DPCF z Filament v1.23.1 |
| `chs` | Pełny wariant `contactFixedGather4` z AMD ShadowFX |

Metody mapowe korzystają z tej samej kamery światła, projekcji NDC i jednej warstwy D32F. `--warp=0` wyłącza zagęszczenie mapy. Generowanie mapy to jeden przebieg, ewaluacja cienia to osobny, pojedynczy przebieg deferred. G-buffer kamery jest wspólny. Nie ma dodatkowej mapy cieni ani czasowego odszumiania.

## Budowanie i obsługa

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release
cmake-build-release/gfx-research-shadows --scene=crossing --method=geometric
```

Windows: plik wykonywalny ma rozszerzenie `.exe`. CLion obsługuje presety CMake. Lokalną bazę można podać przez `-DGFX_RESEARCH_BASE_LOCAL_PATH=/sciezka/gfx-research-base`.

Kamera: prawy przycisk myszy obraca, środkowy przesuwa, kółko przybliża. WASD przesuwa, Q/E zmienia wysokość, Shift przyspiesza, R resetuje, P zapisuje obraz bez GUI. Parametry CLI: `--camera-yaw`, `--camera-pitch`, `--camera-distance`, `--camera-target-x`, `--camera-target-y`, `--camera-target-z`. Sceny: `single-high`, `single-low`, `mixed-overlap`, `crossing`, `hidden-layer`, `sloped`, `dense`.

## Smoothing: mean blocker, geometric i fixed PCF

W `pcss` promień wynika ze średniej arytmetycznej dodatnich, skorygowanych o płaszczyznę odbiornika odstępów: `A = sum(g) / count`. W `geometric` używamy `G = exp(sum(log(g)) / count)`. Wygładzanie widoczności jest dostępne dla obu metod oraz dla stałego PCF. Hard shadow i ARR pozostają metodami kontrolnymi, a DPCF i CHS korzystają ze swoich oryginalnych rekonstrukcji.

Każdy tap końcowego PCF w trybie bilinearnym pobiera sąsiedztwo 2×2 przez gather4. Dla każdej głębokości osobno liczymy test zasłonięcia z korekcją płaszczyzny, a następnie interpolujemy cztery wyniki. Przy małym promieniu PCSS przejście między kontaktem a filtrem dyskowym jest płynne. Nie interpolujemy surowych głębokości pomiędzy obiektami ani nie zmieniamy kontrastu końcowego cienia.

Tryb wybiera się w panelu `Smoothing` lub przez `--smoothing`. Dawne `--geometric-smoothing` pozostaje aliasem tego samego ustawienia:

| Wartość | Wygładzanie |
| --- | --- |
| `0` | Punktowe porównania; w PCSS przywraca wcześniejszy wariant bez dodatkowego smoothingu |
| `1` | Bilinearne wyniki PCF, pierwotna statystyka blockerów, domyślne |
| `2` | Jak 1, dodatkowo bilinearnie ważone statystyki blockerów |
| `3` | Monotoniczna interpolacja Hermite'a `h(t)=3t²-2t³`, w PCSS również ważone statystyki blockerów |
| `4` | Rekonstrukcja widoczności B-spline 4×4, pierwotna statystyka blockerów |
| `5` | Bicubic Catmull-Rom 4×4, pierwotna statystyka blockerów |
| `6` | Preset: bilinear + losowe próbki dysku, dla mean i geometric |
| `7` | Preset: bilinear + stratyfikowane losowe próbki dysku, dla mean i geometric |

W trybach 2 i 3 mean blocker używa `A = sum(w*g) / sum(w)`, a geometric `G = exp(sum(w*log(g)) / sum(w))`. Każdy odstęp zachowuje własny wkład. Wagi opisują położenie próbek, nie preferencję dużego lub małego odstępu. Tryb Hermite'a przenosi pomysł rekonstrukcji tekstur terenu na wyniki porównań. Nieujemne wagi zachowują zakres [0,1], ale nie gwarantują lepszego dopasowania do ARR.

Fixed PCF udostępnia tryby 0, 1, 3, 4 i 5. Zachowuje stały promień i siatkę tapów; nie wykonuje blocker search. Ustawienia 2, 6 i 7 są dla niego równoważne trybowi 1. Presety 6 i 7 wybierają bilinear oraz wzorzec próbek, więc nie są osobnymi algorytmami wygładzania. Poza presetami każdy filtr mean/geometric można łączyć z dowolnym `PCSS sample pattern`, również bicubic z próbkami losowymi.

```sh
cmake-build-release/gfx-research-shadows --method=pcss --smoothing=5
cmake-build-release/gfx-research-shadows --method=pcss --smoothing=7 --sample-seed=37
cmake-build-release/gfx-research-shadows --method=pcf --smoothing=3
```

Zmiana dotyczy rekonstrukcji i statystyki PCSS, nie reprezentacji mapy analogicznej do VSM, Convolution Shadow Maps czy ESM. Materiały: [Fernando, PCSS](https://developer.download.nvidia.com/shaderlibrary/docs/shadow_PCSS.pdf), [Romain Vergne, Shadows](http://romain.vergne.free.fr/teaching/IS/SI08-shadows.html), [GPU Gems 2, rozdział 17](https://developer.nvidia.com/gpugems/gpugems2/part-ii-shading-lighting-and-shadows/chapter-17-efficient-soft-edged-shadows-using).

**B-spline 4×4.** Tryb 4 rekonstruuje maskę widoczności sześciennym filtrem B-spline z 16 osobnych porównań głębokości. Nie zmienia wyszukiwania blockerów ani wzoru na promień wybranej metody PCSS. Wagi są nieujemne i sumują się do jedności, więc filtr nie tworzy oscylacji charakterystycznych dla jąder z ujemnymi wagami. Szersza rekonstrukcja może dodatkowo zmiękczać bliskie cienie. Koszt wynosi cztery gather4 na tap PCF, w porównaniu z jednym w trybie 1. Nie wymaga dodatkowej tekstury ani przebiegu. Definicja filtra: [Sigg i Hadwiger, Fast Third-Order Texture Filtering](https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-20-fast-third-order-texture-filtering).

```sh
cmake-build-release/gfx-research-shadows --method=geometric --smoothing=4
```

Parametry: `--samples=48`, `--blocker-samples=48`, `--max-radius=3`. Gather4 oznacza cztery porównania zamiast jednego, a tryby 2 i 3 dodatkowe obliczenia logarytmów. Wygładzanie nie jest bezkosztowe. Nie usuwa też utraty geometrii ukrytej w jednej mapie ani ograniczenia PCSS do pojedynczego promienia przy mieszaniu różnych przeszkód.

**Bicubic Catmull-Rom 4×4.** Tryb 5 używa interpolującego jądra Keys z parametrem `a=-0.5`. Podobnie jak B-spline pobiera 16 wyników porównań, ale zachowuje wartości w środkach tekseli. Ujemne wagi mogą powodować oscylacje przy nieciągłościach. Wynik każdego tapu jest ograniczany do [0,1]; nie oznacza to usunięcia wszystkich aureoli. Koszt to cztery gather4 na tap. B-spline także jest filtrem bikubicznym; nazwy w menu rozróżniają użyte jądra.

## Próbkowanie PCSS

| `--sample-pattern` | Rozmieszczenie |
| --- | --- |
| `vogel` | Poprzedni deterministyczny dysk Vogela, domyślne |
| `random` | Losowanie równomierne względem pola dysku |
| `stratified` | Jedna losowa próbka w każdej komórce pierścienia i sektora |

Parametr `--sample-seed=0` wybiera ziarno; dostępny jest też w panelu. Losowania zależą od piksela i ziarna, więc nieruchoma scena daje powtarzalny obraz. Blocker search i końcowy PCF korzystają z oddzielnych strumieni. `--rotate-samples` nadal pozwala obracać wzorzec. Losowe pozycje wyznacza całkowitoliczbowy hash PCG, bez tekstury szumu i bez kolejnego przebiegu. Źródło: [Jarzynski i Olano, Hash Functions for GPU Rendering](https://jcgt.org/published/0009/03/02/).

Wzór `r=sqrt(u)` zapewnia poprawny rozkład pola dysku. Wariant stratyfikowany dzieli dysk na komórki o polu `pi/N`, również gdy liczba próbek nie jest kwadratem. To ogranicza skupiska i puste obszary w porównaniu z niezależnym losowaniem. Losowanie może zamieniać pasma w ziarno; nie ma akumulacji czasowej. Ustawienia dotyczą `pcss` i `geometric`; DPCF, CHS oraz ARR zachowują swoje oryginalne wzorce.

```sh
cmake-build-release/gfx-research-shadows --method=geometric --smoothing=5 --sample-pattern=stratified --sample-seed=37
```

## Pozostałe filtry

**DPCF.** Port całego kierunkowego `ShadowSample_DPCF` z [Filament v1.23.1](https://github.com/google/filament/blob/v1.23.1/shaders/src/shadowing.fs), commit `a67d50b9e29cf7bf2f0a6f0b7bf4668cdb87770a`. Oryginalny dysk Poissona, wspólne próbki PCF i średniej głębokości blockerów, receiver-plane bias, penumbra ratio i sześcienny `hardenedKernel`. Dysk zawsze obraca przestrzenny interleaved gradient noise. Parametry: `--dpcf-taps=12` (1..64), `--dpcf-radius=6`, `--dpcf-ratio-scale=1`. [Shadows of Cold War](https://research.activision.com/publications/2021/10/shadows-of-cold-war--a-scalable-approach-to-shadowing).

**CHS.** Port `contactFixedGather4` z [AMD ShadowFX](https://github.com/GPUOpen-Effects/ShadowFX), commit `c6d931c27275b1bb65ba0abfb71eaba86f355387`. Zachowuje `uniformBlockerSearch`, oryginalne macierze wag 7, 9, 11, 13, 15, sześcienną interpolację Bezier i normalizację. Parametry: `--chs-kernel=11`, `--chs-sun-size=4`; do filtra trafia kwadrat rozmiaru, jak w przykładzie AMD. Macierze umieszczono w stałym UBO o rozmiarze 10400 bajtów. Rekonstrukcja sumuje bezpośrednio wkłady gather4, eliminując dynamicznie indeksowaną tablicę lokalną. To przekształcenie tego samego filtra, kontrolowane względem oryginalnego backendu AMD PCF.

Każda metoda ma osobny, buforowany program shadera. CHS ma wariant kompilowany dla wybranego rozmiaru jądra, a PCF, mean i geometric dla wybranego wygładzania. Presety losowe korzystają z istniejącego programu bilinear i uniformów. Kod CHS nie obciąża programu geometric. Kompilacja odbywa się przed zakresem pomiarów GPU. Pierwsze wybranie nieużywanego wariantu może wymagać kompilacji.

Porty dotyczą filtrów, nie całych silników i SDK. Integracja używa naszego G-buffera, forward Z i ortograficznej mapy NDC. DPCF przelicza znak biasu i penumbra ratio z reverse Z. Promień DPCF i rozmiar Słońca CHS skalujemy przez `tan(angle)/tan(1.5°)`. Jest to kalibracja laboratorium, nie identyczne fizyczne odwzorowanie Słońca we wszystkich metodach. CHS zachowuje oryginalne porównania głębokości bez dodatkowej korekcji płaszczyzny odbiornika. Źródła, identyfikatory i licencje są w `upstream/`.

**ARR.** Dokładne przecięcia promień-AABB dla skończonej liczby kierunków na dysku kątowym. Wynik uwzględnia cosinus odbiornika i miarę kąta bryłowego. `--reference-samples=512` steruje przybliżeniem całki; 1024 lub 4096 pozwala sprawdzić zbieżność. ARR korzysta z geometrii i stanowi referencję, nie metodę z jednej mapy.

## Pomiary

```sh
python tools/run_experiment.py --profile quick --exe cmake-build-release/gfx-research-shadows --reference-samples=1024
python tools/run_experiment.py --profile manual --exe cmake-build-release/gfx-research-shadows --scene=crossing --method=geometric --geometric-smoothing=0 --output=results/point
python tools/run_experiment.py --profile manual --exe cmake-build-release/gfx-research-shadows --scene=crossing --method=geometric --geometric-smoothing=1 --output=results/linear
```

Profile: `smoke` to 14 przypadków, `quick` 28, `full` 588. Linux obsługuje bezokienkowy EGL przez `--headless`. Domyślnie używamy rasteryzacji; `--map-backend=analytic` to oddzielny oracle brył do diagnostyki mapy.

Runner zapisuje żądany smoothing, faktyczne jądro rekonstrukcji, skonfigurowany i faktyczny wzorzec próbek, ziarno oraz kamerę, hashe źródeł i programu, surowe czasy, PNG, liniowe PFM oraz RMSE, błąd półcienia i gradientów, maksymalny błąd i rozjaśnienie po dodaniu przeszkody. `configured_filter_samples` i `configured_blocker_samples` opisują ustawienia PCSS; DPCF używa `dpcf_taps`, CHS dwóch etapów po `((kernel+1)/2)^2` gather4 bez wcześniejszego zakończenia. Czas obejmuje mapę i ewaluację cienia, bez wspólnego G-buffera, kompilacji, odczytu i GUI.

Przed pomiarami GPU wykonuje 320 kontroli filtrów i 1568 kontroli pozycji próbek. Niezależny oracle CPU porównuje DPCF z jego definicją, CHS gather4 z oryginalnym `contactFixedPCF`, a rekonstrukcję bilinearną i Hermite'a z osobnymi testami czterech tekseli oraz ważonymi średnimi arytmetyczną i geometryczną. B-spline i Catmull-Rom 4×4 porównywane są z niezależnymi, skalarnymi definicjami funkcji bazowych i 16 osobnymi testami CPU. Samplery mają kontrolę pozycji GPU/CPU, zakresu dysku i statystyk pełnych zestawów dla 48, 96 i 256 próbek. Analyzer odrzuca niezgodności, niepełne serie i naruszenie kontraktu jednej mapy. ARR ma niezależny ray-box CPU i test zbieżności. `--publication` oraz `--require-hardware` odrzucają czasy renderera programowego.

Przykłady, porównania filtrów i rozmieszczenie próbek są w `examples/`. Pliki bez prefiksu `smoothing` dokumentują wersję 0.11.0; nowe wyniki, porównanie i zapis weryfikacji mają prefiks `smoothing`. Domyślny pozostaje tryb bilinearny z dyskiem Vogela. Wyższy rząd filtra i losowanie nie gwarantują mniejszego błędu względem ARR w każdej scenie. Wydajność sprawdzono na llvmpipe, nie na RX 9070 XT.

Weryfikacja 0.12.0 objęła 60 przypadków renderowania, 320 kontroli filtrów i 1568 kontroli próbek GPU/CPU na serię oraz 3 testy CTest. Weighted mean ma dodatkowy niezależny oracle arytmetycznej średniej blockerów. Wszystkie 24 obrazy metod kontrolnych pozostały identyczne z 0.11.0; tryb 0 odtwarza poprzedni mean. Presety 6 i 7 dały identyczny PFM jak ręcznie ustawione bilinear z odpowiednim wzorcem. W jednym teście `crossing` RMSE mean zmalał z 0.009208 do 0.008748 po włączeniu bilinear. To wynik dla jednej konfiguracji, nie gwarancja poprawy wszystkich scen.
