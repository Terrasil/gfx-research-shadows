# gfx-research-shadows

Projekt badawczy C++20 / OpenGL 4.6 oparty na **gfx-research-base**. Służy do porównywania metod miękkich cieni w etapie deferred, ze szczególnym uwzględnieniem nakładania szerokiego półcienia wysokiej przeszkody i ostrzejszego cienia przeszkody blisko podłoża.

To działający renderer i zestaw eksperymentów. Żaden wariant eksperymentalny nie jest przedstawiany jako potwierdzona nowa metoda ani rozwiązanie wszystkich artefaktów LoM.

## Powiązanie z bazą

Projekt używa `gfx::research_base`, `Application`, `OrbitCamera`, `Shader`, `Mesh`, `Framebuffer`, `FullscreenTriangle`, obsługi zrzutów PNG i GUI ImGui dostarczanych przez bazę. CMake domyślnie pobiera repozytorium `Terrasil/gfx-research-base` z przypiętym commitem `a419073ff7c5fd781ef52a4ff9fcdb0c67c5275c`.

Można wskazać lokalny checkout. W takim przypadku manifest zapisuje jego rzeczywisty commit zamiast udawać, że użyto wersji domyślnej.

## Budowanie na Windows / MinGW

Wymagania: CMake 3.25+, Ninja, kompilator C++20, Python 3 oraz sterownik OpenGL 4.6. Możesz użyć swojego GCC 16.2. Pierwsza konfiguracja pobiera zależności bazy. Generator GLAD wymaga modułu Python `jinja2`.

```powershell
python -m pip install jinja2
cmake --preset release
cmake --build --preset release
ctest --preset release
```

Z własną bazą:

```powershell
cmake --preset release -DGFX_RESEARCH_BASE_LOCAL_PATH=E:/gfx-research-base
cmake --build --preset release
```

Cel wykonywalny: `gfx-research-shadows`. Baza zachowuje swoje ustawienia statycznego runtime na Windows.

Na Linux potrzebne są zależności kompilacji GLFW. Przykład dla Ubuntu:

```bash
sudo apt install build-essential cmake ninja-build python3-jinja2 libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
cmake --preset release -DGLFW_BUILD_WAYLAND=OFF
cmake --build --preset release
ctest --preset release
```

## Uruchomienie interaktywne

```powershell
.\cmake-build-release\gfx-research-shadows.exe
```

Panel pozwala zmieniać metodę, scenę, promień kątowy słońca, siłę zagęszczenia NDC, liczbę próbek, rodzaj generacji map i preset kamery. Dostępny jest widok oświetlony i surowa widoczność światła. Przycisk zapisu tworzy obraz bez GUI.

## Metody

Wartość widoczności `0` oznacza całkowite zasłonięcie, a `1` pełną widoczność źródła.

| Identyfikator | Metoda | Używane mapy | Przebiegi ewaluacji |
| --- | --- | ---: | ---: |
| `hard` | Pojedynczy test głębokości | 1 | 1 |
| `pcf` | PCF o stałym promieniu | 1 | 1 |
| `pcss` | PCSS ze średnią odległością przeszkód | 1 | 1 |
| `pcss-min-gap` | PCSS z minimalną odległością przeszkody od odbiornika | 1 | 1 |
| `ndc-pcss` | Ten sam PCSS w nieliniowo zagęszczonej mapie | 1 | 1 |
| `split-min` | Dwie niezależne grupy przeszkód, PCSS każdej, minimum widoczności | 2 | 1 |
| `split-product` | Te same dwie grupy, iloczyn widoczności | 2 | 1 |
| `binned-min` | Jedna mapa, dwa przedziały odległości, minimum ich estymat | 1 | 1 |
| `ray16` | 16 kierunków źródła, dokładne przecięcia z bryłami sceny | 0 | 1 |
| `reference` | Wieloprzebiegowa całka widoczności całej sceny | 0 | N + rozstrzygnięcie sum |

`split-min`, `split-product` i `binned-min` używają zagęszczenia NDC. `ray16` ma dostęp do całej listy brył. To kontrolny punkt porównania dla tej sceny, a nie zamiennik dowolnego renderera siatek.

Grupy w metodach `split-*` są **jawnie oznaczone w scenie**, nie są automatycznie rozpoznawane na podstawie obrazu. Nie należy porównywać ich kosztu jako metody z jedną mapą. `binned-min` ma wyłącznie najbliższą warstwę głębokości, przez co może zgubić przesłonięte powierzchnie. Te ograniczenia są celowo zachowane w badaniu.

## Sceny

| Scena | Badany przypadek |
| --- | --- |
| `single-high` | Szeroki półcień pojedynczej wysokiej przeszkody |
| `single-low` | Wąski półcień przeszkody przy podłożu |
| `mixed-overlap` | Podstawowy problem LoM: szeroki i ostry cień w tym samym obszarze |
| `crossing` | Przecinające się cienie poprzecznych przeszkód |
| `hidden-layer` | Istotna powierzchnia ukryta w mapie głębokości |
| `sloped` | Pochylona powierzchnia przyjmująca cień |
| `dense` | Wiele przeszkód o różnych wysokościach |

Sceny są proceduralne. Składają się z prostopadłościanów i planarnego odbiornika. Ta ograniczona geometria pozwala policzyć niezależną referencję bez korzystania z map cieni. Baza jest gotowa do dalszego dodania siatek, lecz aktualna referencja dotyczy wyłącznie opisanych brył. Płaszczyzna odbiornika nie rzuca cienia.

## Generowanie map

Domyślny backend `raster` rysuje rzeczywistą geometrię do mapy głębokości. Przy aktywnym NDC używa teselacji do przybliżenia zakrzywionych krawędzi projekcji. Głębokość fragmentu jest odtwarzana przez przecięcie promienia z płaszczyzną ściany. Poziom teselacji wynosi 32, a przy projekcji liniowej 1.

Backend `analytic` buduje mapę przez dokładne przecięcia promieni z tymi samymi prostopadłościanami. To diagnostyka wpływu rasteryzacji i odkształcenia NDC. Wyników czasu tego backendu nie wolno opisywać jako kosztu rasteryzacji dowolnej sceny gry.

Próbki PCSS są rozmieszczane w niezdeformowanej przestrzeni światła i dopiero potem mapowane do tekstury. Maksymalny promień jest wyrażony w jednostkach świata. Pozwala to uniknąć zmiany fizycznej szerokości półcienia tylko z powodu zagęszczenia NDC. Korekcja płaszczyzny odbiornika jest wspólna dla metod opartych na mapach.

## Eksperymenty

Krótka kontrola wszystkich metod:

```powershell
python tools/run_experiment.py --profile smoke
```

Szybka seria: 4 sceny, 640x360, mapa 512, 128 kierunków referencji, 3 klatki rozgrzewki i 8 pomiarów dla przybliżeń. Łącznie **40 przypadków**.

```powershell
python tools/run_experiment.py --profile quick --output results/quick-raster
python tools/run_experiment.py --profile quick --analytic --output results/quick-analytic
```

Pełna seria: 7 scen, 1280x720 i 1920x1080, mapy 1024 i 2048, promienie kątowe 0.27, 1.5 i 3 stopnie, 10 metod. Łącznie **840 przypadków**. Przybliżenia mają 120 klatek rozgrzewki i 300 pomiarów. Kosztowna referencja ma jeden zarejestrowany pomiar na konfigurację i 1024 kierunki. Liczba ta jest zapisana osobno w CSV.

```powershell
python tools/run_experiment.py --profile full --output results/publication-raster
python tools/analyze_results.py results/publication-raster --publication
```

Własny przypadek, np. większy budżet referencji:

```powershell
.\cmake-build-release\gfx-research-shadows.exe --benchmark --method=all --scene=mixed-overlap --reference-samples=2048 --angle=3 --output=results/overlap
python tools/analyze_results.py results/overlap
```

Opcje profilu podawaj przed opcjami nadpisującymi jego ustawienia. Na Windows `--headless` ukrywa okno GLFW. Na Linux ten przełącznik uruchamia kontekst EGL bez serwera okien i używa tego samego renderera oraz tych samych klas bazy. Wymaga `libEGL.so.1` i obsługi OpenGL 4.6 przez sterownik. Domyślne uruchomienie interaktywne korzysta z aplikacji GLFW/ImGui z bazy.

## Zapisywane materiały

- `raw.csv`: wynik każdego przypadku, dokładne wymiary, RMSE, RMSE półcienia, błąd gradientu, błąd maksymalny, naruszenia monotoniczności i mediany czasu GPU.
- `timings.csv`: każdy pomiar znaczników GPU, oddzielnie generacja map i ewaluacja cienia.
- `reference_convergence.csv`: porównanie N/2 i N kierunków oraz kontrola referencji przez niezależny kod CPU w podwójnej precyzji.
- `run_manifest.json`: wersje, rzeczywisty renderer, ustawienia, profil i znacznik ukończenia.
- `source_manifest.json`: komenda i hashe źródeł, gdy użyto wrappera Python.
- `capture_manifest.csv`: indeks wszystkich materiałów obrazowych.
- `captures/`: surowa widoczność PFM, maska odbiornika oraz PNG oświetlenia, widoczności i różnicy względem referencji.
- `summary.json` i `report.html`: wyniki walidatora i zestawienie pomiarów.

PFM zachowuje wartości liniowe, bez gammy i tonemappingu. Czas GPU obejmuje generację wymaganych map i ewaluację cienia; wyłącza wspólny G-buffer, GUI, odczyty i zapis plików. Jednoprzebiegowość dotyczy ewaluacji deferred, a nie przygotowania wszystkich danych.

Pole `map_bytes` opisuje używane warstwy głębokości. Implementacja rezerwuje wspólną pulę dwóch warstw także dla metod używających jednej, aby przełączanie w GUI nie wymagało ponownej alokacji. Pamięć G-buffera i obrazów referencyjnych nie jest w tym polu uwzględniona.

Test monotoniczności zachowuje identyczną kamerę i G-buffer, wyłączając jedynie rzucanie cienia przez grupę niskich przeszkód. Dodanie nieprzezroczystej przeszkody nie powinno zwiększać widoczności światła. Metryki są liczone na widocznym odbiorniku, bez tła i brył zasłaniających odbiornik.

Walidator odrzuca niekompletne serie, NaN/Inf, błędne zakresy, brakujące pomiary i zrzuty oraz niezgodność metryk z zapisanymi obrazami. `--publication` dodatkowo wymaga pełnego profilu, renderera sprzętowego i spełnienia jawnego kryterium zbieżności referencji. Próby wykonane przez llvmpipe służą weryfikacji działania i nie są wynikami wydajności na RX 9070 XT.

Plan badań, źródła i ograniczenia znajdują się w `docs/RESEARCH_PLAN.md` i `docs/RELATED_WORK.md`.
