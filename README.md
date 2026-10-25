# gfx-research-shadows

Laboratorium porównawcze cieni C++20 / OpenGL 4.6 na [gfx-research-base](https://github.com/Terrasil/gfx-research-base), rewizja `a419073ff7c5fd781ef52a4ff9fcdb0c67c5275c`. Wersja **0.31.0** dodaje osobne **GSM z filtrowaną średnią geometryczną mapy**, a dotychczasowe **PCSS geometric mean** umieszcza w grupie PCSS. Zachowuje wszystkie cztery pozycje Multiple light view i referencje ARR.

## Grupy metod

Najpierw wybierz **Group**, następnie metodę z krótkiej listy danej grupy.

| Grupa | Metody | Identyfikatory CLI |
| --- | --- | --- |
| Basic | Hard shadow, Fixed PCF | `hard`, `pcf` |
| PCSS | PCSS mean blocker, PCSS geometric mean, Geometric depth bands, DPCF, CHS | `pcss`, `geometric`, `geometric-layered`, `dpcf`, `chs` |
| Filtered maps | VSM, CSM, ESM, GSM | `vsm`, `csm`, `esm`, `gsm` |
| Multiple light view | Angular RGBA, Angular separate, Angular RGBA cascaded, Angular separate cascaded | `multiple-light-view` + `--multiview-variant` |
| Geometry reference | ARR exact boxes i ARR triangulation | `geometry-reference`, `arr-triangulation` |

**PCSS geometric mean** zachowuje dotychczasowy algorytm: geometryczną średnią dodatnich odstępów przeszkód od odbiorcy i PCF o zmiennym promieniu. **GSM** jest nowym, osobnym wariantem opisanym poniżej. **CSM** nadal oznacza Convolution Shadow Maps, czyli bazę Fouriera; kaskady są opcją Multiple light view.

Dostępnych jest 14 rodzin algorytmów i ich dotychczasowe identyfikatory w CSV. Multiple light view ma teraz cztery jawne pozycje na liście; pole `multiview_variant` rozróżnia je w wynikach. Aliasy: `pcss-geometric` = `geometric`, `multiple-light-view` = `angular-rgba`, `geometry-reference` = `arr`. Od 0.31 identyfikator `gsm` uruchamia nową mapę geometryczną. Do odtwarzania starego geometric PCSS używaj `geometric` albo `pcss-geometric`.

## GSM: filtrowana mapa geometryczna

Dla głębokości mapy `z` w przedziale 0..1 zapisujemy do R32F:

```text
B(z) = log(1 + z)
L = GaussianFilter(B)
G = exp(L)
V(t) = min(1, (G / (1 + max(t, 0)))^k)
```

`G` jest ważoną średnią geometryczną wartości `1+z`. Przesunięcie o 1 pozwala obsłużyć głębokość zero bez `log(0)`. `t` jest głębokością odbiorcy po biasie. Shader ocenia równoważne `exp(min(0, k*(L-log(1+max(t,0)))))`, bez tworzenia dużych potęg. Na GPU filtrujemy kodowaną mapę przed oceną cienia, podobnie organizacyjnie jak VSM/CSM/ESM. ESM filtruje wartości wykładnicze, a ten wariant logarytmy głębokości, więc wynik nie jest zamianą nazwy funkcji `exp` na `pow`.

**Gaussian radius** reguluje szerokość przestrzennego filtra 0..16 texeli. **GSM sharpness**, domyślnie 64, ma zakres 1..128 i reguluje spadek widoczności za odtworzoną średnią. Nie ma tu szukania przeszkód ani promienia PCSS zależnego od odległości. Rozmiar słońca nie dobiera automatycznie tego stałego filtra.

Koszt: jedna mapa D32F, kodowanie i dwa przebiegi rozmycia do dwóch R32F, następnie jeden bilinearny odczyt bazy na piksel i funkcja geometryczna. Dane map zajmują łącznie `12*N*N` B, czyli 12 MiB przy 1024²; pomocnicze mapy zajmują 8 MiB. Liczniki GPU obejmują kodowanie i oba rozmycia.

GSM jest **eksperymentalną funkcją porównawczą projektu**. Pojedyncza średnia geometryczna nie odtwarza dowolnego rozkładu przesłonięcia. Dla dwóch równoważnych próbek 0.2 i 0.8, odbiorcy 0.5 i `k=64` daje około 0.271, podczas gdy średnia binarnych porównań wynosi 0.5. Może więc przyciemniać półcienie i przesuwać granice. Wynik zależy też od normalizacji głębokości i ustawienia `k`. Nie oznacza to rozwiązania problemu mieszanych warstw ani dokładnej integracji tarczy światła. Skrót GSM opisuje tutaj powyższą funkcję projektową.

```sh
cmake-build-release/gfx-research-shadows --method=gsm --prefilter-radius=4 --gsm-sharpness=64
cmake-build-release/gfx-research-shadows --method=pcss-geometric --samples=48 --blocker-samples=48
```

## Budowanie i sterowanie

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release
cmake-build-release/gfx-research-shadows --scene=mixed-overlap --method=gsm
```

Windows: plik wykonywalny ma rozszerzenie `.exe`. CLion obsługuje presety CMake. Lokalną bazę można wskazać przez `-DGFX_RESEARCH_BASE_LOCAL_PATH=/sciezka/gfx-research-base`. Konfiguracja pobiera zależności przy pierwszym uruchomieniu. ZIP zawiera źródła, bez katalogów build i pobranych zależności. Daty plików w archiwum są stałe i nie wybiegają w przyszłość, aby uniknąć ciągłej regeneracji CMake/Ninja.

Każdy suwak ma przyciski `-` i `+`; przytrzymanie powtarza zmianę. Ctrl+klik wartości umożliwia wpisanie liczby. RMB obraca kamerę, MMB przesuwa, kółko przybliża. WASD przesuwa, Q/E zmienia wysokość, Shift przyspiesza, R resetuje, P zapisuje obraz.

## Multiple light view

To **cztery widoki jednego słońca**: centralny oraz trzy na pierścieniu co 120 stopni. Każdy ma własną macierz, UV, zakres głębokości, bias i opcjonalny zasięg kaskady. Przesunięcia kątowe można wyłączyć. Grupa służy badaniu rekonstrukcji i mieszania widoczności tych widoków.

Po wybraniu **Group → Multiple light view** lista metod zawiera cztery pozycje:

| Pozycja w menu | Zapis map | Pokrycie | `--multiview-variant` |
| --- | --- | --- | --- |
| Angular RGBA (one texture) | R/G/B/A jednej tekstury | Jednakowe zasięgi | `rgba` |
| Angular separate (one texture per view) | Osobna tekstura na widok | Jednakowe zasięgi | `separate` |
| Angular RGBA cascaded (one texture) | R/G/B/A jednej tekstury | Zasięgi rosną kaskadowo | `rgba-cascaded` |
| Angular separate cascaded (one texture per view) | Osobna tekstura na widok | Zasięgi rosną kaskadowo | `separate-cascaded` |

Wybór pozycji zmienia tylko zapis i włączenie kaskad. Kamera, kierunki światła, próbki, filtr i mieszanie pozostają bieżące. W UI presety Fixed/World PCF i PCSS zachowują wybraną pozycję. Zmiana checkboxa kaskad również aktualizuje nazwę wybranego wariantu. Zrzuty P zawierają jego nazwę, więc cztery warianty nie nadpisują wzajemnie swoich obrazów. Widoki kątowe pozostają włączone domyślnie. Domyślne **Angular mixture over cascades = 1** miesza dostępne kierunki również w kaskadach; zero nadal pozwala porównać wybór najbliższej kaskady.

Panel jest podzielony na **Storage and cost**, domyślnie otwarte **Light views**, **Comparison presets**, **Per-view filtering**, **Cascade coverage and blending**, **Mixing final visibility** oraz zwiniętą **Legacy angular reconstruction**. Wybór zapisu nie resetuje kamery, światła, próbek, kaskad, filtra ani mieszania.

| Map storage | Zapis | Tekstury | Przebiegi geometrii | Surowa pamięć map |
| --- | --- | ---: | ---: | ---: |
| One RGBA32F texture | Głębokości w R/G/B/A, zapis przez MIN | 1 | 4 | `16*N*N` B |
| One depth array | Cztery warstwy D32F w jednej teksturze | 1 | 4 | `16*N*N` B |
| Four separate D32F textures | Osobna mapa każdego widoku | 4 | 4 | `16*N*N` B |

`N` jest rozdzielczością każdej mapy. Dla 1024² każdy wariant przechowuje 16 MiB danych map, przed narzutem i ewentualną kompresją sterownika. Licznik w UI dotyczy aktywnych map tej metody, nie całej rezydentnej pamięci renderera. Zasoby poprzedniego wariantu Multiple light view są zwalniane przy zmianie zapisu lub rozdzielczości.

**Array kontra separate** porównuje układ pamięci przy tym samym D32F, teście głębokości i algorytmie odczytu. **RGBA kontra D32F** zmienia również ścieżkę rasteryzacji: kolor i MIN zamiast załącznika depth. Wynik tego pomiaru nie opisuje wyłącznie liczby obiektów tekstur. Shader jest specjalizowany dla każdego wariantu; nie ma dodatkowego przebiegu pakowania. Odczyt głębokości jest wspólnym interfejsem używanym przez wszystkie filtry i rekonstrukcje.

```sh
cmake-build-release/gfx-research-shadows --pcss-angular --multiview-variant=rgba
cmake-build-release/gfx-research-shadows --pcss-angular --multiview-variant=separate
cmake-build-release/gfx-research-shadows --pcss-angular --multiview-variant=rgba-cascaded
cmake-build-release/gfx-research-shadows --pcss-angular --multiview-variant=separate-cascaded
```

Tablica depth pozostaje dodatkową diagnostyką w **Storage and cost → Additional depth-array diagnostic**. Nie zastępuje czterech głównych pozycji. `--rgba-storage` jest zgodnym aliasem `--multiview-storage`. Domyślny zapis pozostaje RGBA, aby zachować dotychczasowy wynik. Presety filtrów zachowują wybrany zapis.

### Filtrowanie i mieszanie

**Per-view filtering** zawiera porównania punktowe, Fixed PCF, Adaptive PCF legacy, Shared world PCF i PCSS per channel. Rekonstrukcja wyników porównań: nearest, bilinear, Hermite i bicubic B-spline. Dostępne są 8/16/32/64 próbki filtra, a PCSS osobno wybiera 1..128 próbek przeszkód.

**PCSS per channel** szuka przeszkód także wtedy, gdy środek jest zacieniony. Środek jest jedną z próbek, pozostałe tworzą dysk Vogela. Średnia arytmetyczna albo geometryczna dodatnich odległości daje promień `mean_gap_m * tan(sun_angle) * penumbra_scale`. Każdy widok przelicza go na własne współrzędne i ogranicza lokalnym limitem. Brak przeszkód daje światło. Zerowa siła lub limit filtra zachowuje wynik bez PCSS. Ochrona bliskiego ostrego cienia działa osobno dla każdego widoku przed mieszaniem.

**Camera-independent blocker search** i **Stable focus across coverage modes**, domyślnie włączone od 0.28, zachowują poprawkę odległości i skupienia NDC. Obszar poszukiwań wynika z odstępu odbiorcy od granicy przeszkód we właściwej kamerze, nie od dowolnego położenia jej płaszczyzny near. Różnica głębokości jest przeliczana zakresem tej kamery na metry. Dla dużego stress granica jest konserwatywną obwiednią całego pola z kosztem CPU O(1). Opcje `--rgba-pcss-bounds=0 --rgba-stable-focus=0` przywracają poprzednie zachowanie do porównania.

**Mixing final visibility** łączy ukończone wyniki filtrów: średnia ważona, minimum, iloczyn, średnia geometryczna albo hybryda arytmetycznej i geometrycznej. Nie uśredniamy surowych głębokości różnych kamer. Geometryczne mieszanie widoczności jest innym ustawieniem niż geometryczna średnia odległości przeszkód. Może nadmiernie przyciemniać półcień; jest wariantem porównawczym.

**Shared world PCF** najpierw miesza widoki dla każdego wspólnego punktu dysku na powierzchni odbiorcy, a potem uśrednia przestrzennie. Pozostałe tryby filtrują widoki oddzielnie i dopiero potem łączą wyniki. Przy minimum, iloczynie i innych nieliniowych funkcjach kolejność ma znaczenie.

**PCF moments** opcjonalnie rekonstruują lokalną granicę z położeń już użytych próbek. Działają przy średniej ważonej i wystarczająco zgodnych obserwacjach; niespójne lub nierozdzielone krawędzie wracają do bazowego filtra. Fixed PCF dodatkowo szuka odległości przeszkód, PCSS wykorzystuje własne poszukiwanie. **Joint soft-shadow smoothing** dodaje ekranowy filtr 3×3 albo 5×5 i R32F, 4 B na piksel. Rekonstrukcja legacy 0.21 wykonuje dodatkowe poszukiwania granic i pozostaje osobnym, kosztowniejszym wariantem.

Przy kaskadach widoki obejmują `extent * multiplier^i`. Większy zasięg zmniejsza gęstość texeli; zmiana sposobu zapisu tego nie naprawia. Dla pierwszej szerokości 32 m, mnożnika 4, 1024² i warp 8 ostatni widok ma szerokość 2048 m i texel w centrum około 0.22 m. Przy mnożniku 2 jest to około 0.028 m. Poza zasięgiem małych map dostępnych jest mniej kierunków tarczy.

### Porównanie czterech wariantów

```sh
python tools/compare_multiview_variants.py --exe=cmake-build-release/gfx-research-shadows --profile=quality --output=results/four-variants-quality --pcss-angular --rgba-cascade-scale=2
python tools/compare_multiview_variants.py --exe=cmake-build-release/gfx-research-shadows --profile=stress --output=results/four-variants-stress --pcss-angular --object-counts=1000,10000,100000
```

Linux EGL: dodaj `--headless`. Dla pomiaru sprzętowego dodaj `--require-hardware`. Skrypt odrzuca niepusty katalog wynikowy, uruchamia kolejno cztery pozycje z tabeli, sprawdza zgodność pozostałych ustawień i zapisuje `comparison.csv`. Różnicę PFM dla osobnych tekstur liczy wobec RGBA o takim samym ustawieniu kaskad. Plik zawiera identyfikator wariantu, stan kaskad, błędy wobec ARR oraz czasy poszczególnych etapów. Każdy wariant zachowuje własne surowe pomiary, hashe źródeł i analizę.

Dotychczasowy `compare_multiview_storage.py` pozostaje narzędziem porównującym RGBA, depth array i separate przy jednym ustawieniu kaskad. W CLI presety podaj przed `--multiview-variant`, zgodnie z kolejnością nadpisywania parametrów.

Profil **quality** porównuje wynik z ARR oraz zapisuje różnice PFM względem RGBA. Jego czas obejmuje generowanie map i ocenę cienia, bez wspólnego G-buffer. Profil **stress** mierzy G-buffer + mapy + ocenę cienia i nie wylicza jakości. CSV podaje oddzielnie czas map, shadera cienia, liczbę tekstur i pamięć. Alokacje, kompilacja shaderów, zrzuty i UI są poza zakresami GPU. Domyślne rozgrzewanie pozostaje 12 klatek, pomiar 30. Stała kolejność trzech wariantów może mieć wpływ przy zmianach temperatury lub taktowania; do wniosków o GPU warto powtórzyć pomiar w tych samych warunkach.

## PCSS, GSM i filtrowane mapy

PCSS mean, PCSS geometric i depth bands zachowują smoothing, próbki Vogel/random/stratified/blue-noise, limit promienia, wariant inner, packed min/max, ochronę kontaktu i blend wyniku. `--sample-pattern=blue-noise --blue-noise-tile=1` używa dostarczonej tekstury 2048² R8 do obrotów dysków w przestrzeni świata. Nie jest TAA ani niezależnym zestawem punktów blue noise dla każdej próbki. `--clean-blue-noise` ustawia Hermite, packed endpoints, ochronę kontaktu i blend, zachowując budżety próbek.

Packed min/max używa R32UI z dwoma końcami UNORM16, więc ma 4 B/texel i mniejszą precyzję każdej składowej niż D32F. Przechowuje końce przedziału powierzchni, nie dowolną liczbę ukrytych warstw. Geometric depth bands zachowuje warunkowe mieszanie dwóch zakresów odległości i opcjonalny minimalny promień filtra. Wariant inner jest ograniczeniem artystycznym.

VSM zapisuje filtrowane `z` i `z²` w RG32F. CSM filtruje współczynniki Fouriera w warstwach RGBA32F, domyślnie 16 harmonicznych. ESM zapisuje `exp(c*(z-1))` w R32F i opcjonalnie stosuje klasyfikację z filtrem awaryjnym. Pozostają odpowiednio light bleeding, ringing i ograniczenia aproksymacji wykładniczej. Samo filtrowanie map nie zapewnia poprawnego kontaktowego półcienia przy dowolnym nakładaniu obiektów.

DPCF zachowuje dysk Poissona, receiver-plane bias i hardened kernel z Filament v1.23.1. CHS zachowuje `contactFixedGather4`, oryginalne tabele i rozmiary 7/9/11/13/15 z AMD ShadowFX. Są to porty filtrów. CHS pozostaje dostępny do porównań; wybór grupy nie uruchamia go automatycznie.

## Referencja i sceny

**Geometry reference / ARR** wystrzeliwuje promienie do deterministycznych kierunków tarczy i przecina geometrię AABB. Nie rekonstruuje przesłonięcia z pojedynczej mapy depth. Jest odniesieniem jakości dla brył obsługiwanych przez to laboratorium; liczba kierunków nadal ogranicza dokładność całkowania światła. Profil jakości sprawdza zbieżność oraz niezależny oracle CPU. Triangulacja pozostaje porównaniem mniejszego budżetu kierunków. Opcjonalny smoothing obu ARR nie zmienia niefiltrowanej referencji używanej do metryk.

Sceny obejmują nakładające się cienie, skrzyżowane belki, **crossing-four**, powierzchnię pochyloną i **dense-stress**. W dense-stress można ustawić **0..10 000 000 obiektów**, bez zmiany metody ani budżetu próbek. Obiekty są generowane proceduralnie, jeden instancjonowany draw brył na przebieg geometrii. Oba ARR pozostają dostępne i sprawdzają pełną populację. Koszt przecięć rośnie z liczbą brył i kierunków. Analityczny backend diagnostyczny map ma limit 32 brył; duże stress korzysta z rasteryzacji.

```sh
cmake-build-release/gfx-research-shadows --scene=crossing-four --pcss-angular --multiview-storage=array
cmake-build-release/gfx-research-shadows --scene=dense-stress --objects=1000000 --method=multiple-light-view --multiview-storage=separate
cmake-build-release/gfx-research-shadows --scene=dense-stress --objects=1000 --method=arr --arr-samples=4
python tools/run_experiment.py --profile=stress --exe=cmake-build-release/gfx-research-shadows --method=all --object-counts=33 --arr-samples=4
```

## Walidacja

GSM ma niezależny oracle CPU: bezpośredni splot Gaussa 2D dla kodowanej mapy oraz ważony iloczyn do sprawdzenia średniej geometrycznej i widoczności. Macierz obejmuje promienie 0/3/16, ostrość 1/64/128, głębokości 0 i 1 oraz 1152 przypadki GSM. Razem z dotychczasowymi 384 przypadkami VSM/CSM/ESM daje 1536 kontroli. Maksymalny błąd bazy GSM w pierwszym uruchomieniu wyniósł `1.17e-7`, a widoczności `7.37e-6`.

Porównanie z ARR przy 320×240, mapie 256² i 512 kierunkach odniesienia. GSM: promień 4 i `k=64`; geometric PCSS: 16/16 próbek i bilinear.

| Scena | GSM RMSE | PCSS geometric RMSE | GSM RMSE półcienia | PCSS geometric RMSE półcienia |
| --- | ---: | ---: | ---: | ---: |
| mixed-overlap | 0.049646 | 0.017747 | 0.313479 | 0.118907 |
| crossing-four | 0.034826 | 0.019954 | 0.269769 | 0.117536 |

W tych ustawieniach geometric PCSS jest dokładniejsze. GSM służy porównaniu osobnej funkcji filtrowanej mapy. Przeszło 28 wierszy jakości, wszystkie 14 metod w dense-stress oraz cztery warianty Multiple light view.

Zachowano 240 renderowań kontroli trzech sposobów zapisu Multiple light view i wcześniejsze testy GPU/CPU PCSS, mieszania, odległości, blue noise, packed oraz ARR. CTest obejmuje geometrię, kamerę i 38 testów analizatora. Dane bieżącej wersji są w `examples/gsm0310_*`, a starsze przykłady zachowują wersje. Lokalna walidacja korzysta z llvmpipe i nie rozstrzyga wydajności na RX 9070 XT.

## Źródła metod

- [PCSS, NVIDIA](https://developer.download.nvidia.com/shaderlibrary/docs/shadow_PCSS.pdf) i [PCF, GPU Gems](https://developer.nvidia.com/gpugems/gpugems/part-ii-lighting-and-shadows/chapter-11-shadow-map-antialiasing).
- [VSM, Donnelly i Lauritzen](https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-8-summed-area-variance-shadow-maps), [Convolution Shadow Maps](https://jankautz.com/publications/csmEGSR07.pdf), [Exponential Shadow Maps](https://jankautz.com/publications/esm_gi08.pdf).
- [DPCF, Filament v1.23.1](https://github.com/google/filament/blob/v1.23.1/shaders/src/shadowing.fs), rewizja `a67d50b9e29cf7bf2f0a6f0b7bf4668cdb87770a`.
- [CHS, AMD ShadowFX](https://github.com/GPUOpen-Effects/ShadowFX), rewizja `c6d931c27275b1bb65ba0abfb71eaba86f355387`.

Jedyny dokument Markdown w projekcie to README. Projekt służy badaniu jakości i kosztu, bez przypisywania nowości samemu pakowaniu map, średniej geometrycznej ani mieszaniu wyników.
