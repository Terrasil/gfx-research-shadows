# gfx-research-shadows

Laboratorium cieni C++20 / OpenGL 4.6 na [gfx-research-base](https://github.com/Terrasil/gfx-research-base), rewizja `a419073ff7c5fd781ef52a4ff9fcdb0c67c5275c`. Wersja 0.16.1 udostępnia oba ARR dla całej populacji `dense-stress`, bez przełączania metody ani ograniczania liczby badanych brył. Wersja 0.16.0 dodała scenę wydajnościową `dense-stress` z 0..10 milionów proceduralnych instancji i osobnymi pomiarami geometrii, mapy oraz filtra. Wersja 0.15.3 dodała preset wygładzania packed min/max, wspólny losowy wzorzec ograniczający ziarno oraz scenę czterech krzyżujących się belek. Zachowuje złożenie wynikowych masek promieni z 0.15.2 i przyciski precyzyjnej regulacji z 0.15.1. Wersja 0.15.0 wprowadziła **ARR triangulation** i opcjonalny smoothing obu metod ARR. Zwykłe ARR liczy kierunki w jednym przebiegu. Projekt zawiera 12 metod; zachowuje geometric z warunkowym mieszaniem, opcję R32UI min/max oraz VSM, Convolution Shadow Maps i ESM. Badamy zachowanie ostrego cienia bliskiej przeszkody wewnątrz półcienia dalszego obiektu.

| ID CLI | Metoda |
| --- | --- |
| `hard` | Punktowy test głębokości |
| `pcf` | Fixed PCF |
| `pcss` | PCSS mean blocker, średnia arytmetyczna |
| `arr` | Angular ray reference, przecięcia promieni z bryłami AABB |
| `geometric` | PCSS geometric mean, dotychczasowy wariant domyślny |
| `dpcf` | Kierunkowy DPCF z Filament v1.23.1 |
| `chs` | `contactFixedGather4` z AMD ShadowFX |
| `geometric-layered` | Geometric depth bands, dwa promienie, warunkowe mieszanie i opcjonalne min/max |
| `vsm` | Variance Shadow Maps, filtrowane momenty |
| `csm` | Convolution Shadow Maps, filtrowana baza Fouriera |
| `esm` | Exponential Shadow Maps, klasyfikacja progiem i filtr awaryjny |
| `arr-triangulation` | Rekonstrukcja widoczności tarczy z trójkątów i miary przecięcia promieni |

## Budowanie i kamera

```sh
cmake --preset release
cmake --build --preset release
ctest --preset release
cmake-build-release/gfx-research-shadows --scene=mixed-overlap --method=geometric-layered
```

Windows: plik wykonywalny ma rozszerzenie `.exe`. CLion obsługuje presety CMake. Lokalną bazę można podać przez `-DGFX_RESEARCH_BASE_LOCAL_PATH=/sciezka/gfx-research-base`.

Każdy suwak ma przyciski `-` i `+`; przytrzymanie powtarza zmianę. Liczniki mają krok 1, a liczba harmonicznych CSM krok 2, zgodnie z dopuszczonymi wartościami. Parametry ułamkowe mają mniejsze kroki, widoczne w podpowiedzi przycisku. Ctrl+klik wartości umożliwia bezpośrednie wpisanie liczby.

Prawy przycisk myszy obraca kamerę, środkowy przesuwa, kółko przybliża. WASD przesuwa, Q/E zmienia wysokość, Shift przyspiesza, R resetuje, P zapisuje obraz bez GUI. CLI: `--camera-yaw`, `--camera-pitch`, `--camera-distance`, `--camera-target-x`, `--camera-target-y`, `--camera-target-z`. Sceny: `single-high`, `single-low`, `mixed-overlap`, `crossing`, `hidden-layer`, `sloped`, `dense`, `crossing-four`, `dense-stress`.

Scena `crossing-four` ma cztery belki naprzemiennie wzdłuż osi X/Z, na wysokościach 0.5, 1.8, 3.3 i 5.0. Ich cienie przecinają się we wspólnym obszarze. Centralny promień światła z punktu podłoża `(-2.5, 0, -2)` przecina wszystkie cztery belki. Scena trafia także do profili smoke, quick i full. Służy do badania utraty środkowych warstw przez min/max, nie zakłada poprawności przy większej liczbie obiektów.

## Scena wydajnościowa z regulowaną liczbą obiektów

Wybierz **`dense-stress`**. `Requested objects` ma przyciski `-`/`+`, skalę logarytmiczną i bezpośrednie wpisywanie przez Ctrl+klik. Presety: 16, 1K, 10K, 100K, 1M, 2M, 10M. Zmianę zatwierdza **`Apply object count`**. Zakres to **0..10 000 000**, domyślnie 4096. Licznik pokazuje rzeczywiście wysłane instancje i ich trójkąty bazowe, bez automatycznego ograniczania populacji do 32.

```sh
cmake-build-release/gfx-research-shadows --scene=dense-stress --objects=1000000 --method=geometric-layered --layer-depth=interval
python tools/run_experiment.py --exe=cmake-build-release/gfx-research-shadows --profile=stress --method=all --object-counts=0,1000,10000,100000,1000000 --warmup=3 --measure=30
```

Bryły powstają z `gl_InstanceID` w jednym instancjonowanym rysowaniu sześcianu na etap geometrii. Pole ma stały rozmiar 12×12; liczba kolumn to `ceil(sqrt(N))`. Zwiększanie liczby obiektów zmniejsza ich szerokość, a wysokości pozostają 0.65 i 3.5 ponad podstawą y=0.1. Dzięki temu obiekty mieszczą się w dotychczasowym zakresie kamery światła. Zmieniają się gęstość, widoczność i overdraw, więc nie jest to test wyłącznie liczby identycznie widocznych brył. Przy milionach wiele elementów jest mniejszych od piksela lub teksela mapy.

Nie tworzymy wielomilionowej listy obiektów w CPU ani bufora transformacji GPU. Cała populacja jest wysyłana bez indywidualnego cullingu CPU. GPU nadal przetwarza geometrię: każda bryła ma **12 trójkątów przed teselacją mapy**. Wzrost overdraw i konflikty atomowe packed min/max mogą zwiększać koszt, mimo stałej liczby draw calls. Shader PCSS/geometric korzysta z mapy, bez pętli po wszystkich bryłach.

Interfejs pokazuje oddzielnie **G-buffer GPU**, **Map GPU**, **Shadow GPU**, sumę tych etapów oraz rzeczywisty FPS pętli aplikacji. Suma GPU nie zawiera GUI, prezentacji, alokacji, kompilacji ani odczytu obrazów. Pomiary znaczników czasu są synchroniczne, więc wpływają na FPS aplikacji. W CSV `wall_ms` oznacza czas `prepare + render` wraz z wywołaniami CPU i oczekiwaniem na wyniki znaczników, a nie sam czas pracy CPU.

Dla tej sceny NDC używa adaptacyjnej teselacji krawędzi, ocenianej z odchylenia rzutowanej krzywej od odcinka. Jest to heurystyka, bez gwarantowanego ograniczenia błędu. **`Stress tessellation limit`**, CLI `--stress-tessellation=32`, ogranicza poziom do 1..32. **`Adaptive stress tessellation`**, CLI `--stress-adaptive-tessellation=0|1`, wybiera adaptacyjny lub stały poziom. `--warp=0` używa poziomu 1. Pozostałe sceny zachowują dotychczasową teselację 32 dla NDC. Porównując koszty, utrzymuj te ustawienia i kamerę takie same.

`--stress-benchmark` uruchamia pomiary bez osobnego oracle jakości i bez fikcyjnych wartości RMSE. `--benchmark --scene=dense-stress --objects=N` automatycznie wybiera ten tryb dla N>32. Profil `stress` zapisuje surowe próbki czasów, mediany, PNG oraz liczebności i ustawienia. **`--method=all` obejmuje wszystkie 12 metod, również ARR i ARR triangulation.** Oba ARR odtwarzają AABB z indeksu instancji i badają pełną ustawioną populację do 10 milionów. Wybór sceny lub zmiana liczebności nie przełącza metody ani nie zmniejsza liczby kierunków. Nie ma przycięcia do pierwszych 32 brył ani zamiany na filtr mapy.

ARR ma pętlę testów ray-box kończącą się po trafieniu. Signed boundary reconstruction triangulacji zbiera miary ze wszystkich brył, a wariant binarny i maski używają testów promieni. Koszt może być bardzo duży przy milionach obiektów, wielu kierunkach i dużym obrazie. Budżet pozostaje pod kontrolą użytkownika przez `ARR directions`, `Triangulation directions`, `--arr-samples` i `--tri-samples`. Smoothing i tryb wyników triangulacji pozostają dostępne.

```sh
cmake-build-release/gfx-research-shadows --scene=dense-stress --objects=1000 --method=arr --arr-samples=4
cmake-build-release/gfx-research-shadows --scene=dense-stress --objects=1000 --method=arr-triangulation --tri-samples=4
```

Limit 32 pozostaje dla diagnostycznej analitycznej mapy cienia oraz niezależnego CPU oracle małych scen. Małe populacje można sprawdzać zwykłym benchmarkiem jakości. Profile jakości smoke, quick i full zachowują 8 scen i nie dodają automatycznie testu milionów obiektów. Przed pomiarami stress wykonujemy testy GPU/CPU ostatniej instancji w populacjach 33, 65, 1024 i 1 000 000, wraz z trafieniem, usunięciem grupy i chybieniem. Potwierdza to dostęp do geometrii poza dawnym limitem.

## ARR i triangulacja

Zwykłe ARR wybiera kierunki **deterministycznie, bez losowania między klatkami**: `r=sqrt(radicalInverse(i+1))`, `a=2*pi*fract((i+0.5)*0.61803398875)`. Kierunek powstaje przez odwzorowanie punktu dysku na stożek wokół Słońca. Wagi uwzględniają Jacobian i cosinus względem normalnej. Pierwsze próbki ilustruje `examples/angular_patterns0150.png`.

`--arr-samples=4` ustawia budżet zwykłego ARR. Domyślne `0` dziedziczy `--reference-samples`, którego wartość początkowa wynosi 512. Kierunki są przygotowywane przed pomiarem w SSBO o pojemności 256 KiB. Wszystkie promienie danego piksela sprawdzamy teraz w jednej pętli shadera, zamiast uruchamiać osobny przebieg dla każdego kierunku. Nadal potrzebny jest opis geometrii AABB; metoda nie korzysta wyłącznie z G-buffera ani z mapy głębokości światła.

Wybierz **`ARR triangulation`**. Menu **`Triangulation result`** pozwala porównać trzy warianty:

- **Ray masks (weighted average):** nowy wariant użytkownika. Każdy kierunek zwraca maskę 0/1, a wynik jest średnią ważoną Jacobianem i cosinusem odbiornika. Układ to trójkąt, trójkąt ze środkiem lub większy pierścień. `--tri-result-masks=1`.
- **Signed boundary reconstruction:** dotychczasowy wariant domyślny, `--tri-result-masks=0 --tri-signed-margin=1`.
- **Binary boundary reconstruction:** dopasowanie granicy z wartości -1/+1, `--tri-result-masks=0 --tri-signed-margin=0`.

Dwa ostatnie warianty odtwarzają granicę zasłonięcia:

```sh
cmake-build-release/gfx-research-shadows --method=arr-triangulation --tri-samples=4 --tri-radius=0.8 --tri-rotation=0
```

- **3 zapytania:** trzy punkty co 120 stopni, dopasowanie jednej liniowej funkcji zasłonięcia i analityczne pole wycinka koła.
- **4 zapytania:** te same trzy punkty oraz środek. Powstają trzy sektory z osobnymi liniowymi funkcjami.
- **5..65 zapytań:** środek i równomierny pierścień, czyli gęstszy wachlarz sektorów. Domyślne 4 daje układ 3+1.

`--tri-radius` określa położenie próbek względem promienia tarczy, `--tri-rotation` obrót w stopniach. Rekonstrukcja obejmuje **całą tarczę kołową**, także łuki poza trójkątem wpisanym. Dopasowaną funkcją przycinamy sektor, a jego pole liczymy przez przecięcie wielokąta z kołem. Dla wachlarza stosujemy stałą wagę oświetlenia w środku ciężkości każdego sektora. To aproksymacja pełnego ważonego całkowania ARR.

Domyślne `--tri-signed-margin=1` korzysta z ciągłej miary przecięcia stożka kierunków z AABB. Znak zgadza się z trafieniem promienia, a wartość wynika z ograniczeń płaszczyzn bryły w przestrzeni kierunków. Nie jest to długość pierwszego trafienia ani dokładna odległość euklidesowa. Płaszczyzny bryły współdzielimy między kierunkami. **Zapytanie z tą miarą dostarcza więcej informacji i ma inny koszt niż zwykły binarny test ray-box.** `--tri-signed-margin=0` przełącza na wartości -1/+1 z prostych testów promieni, umożliwiając kontrolę wkładu dodatkowej informacji.

Trzy lub cztery wartości nie opiszą dowolnego zarysu zasłonięcia. Rekonstrukcja może pominąć cienką przeszkodę między próbkami, błędnie odtworzyć narożnik lub połączyć kilka zarysów. Implementacja dotyczy brył AABB i stałego kierunku Słońca. Nie jest gotowym filtrem dla dowolnych siatek trójkątów i nie potwierdza nowości naukowej samego pomysłu.

### Smoothing obu ARR

W obu metodach jest osobne menu **`Result smoothing`**. Ustawienia PCSS pozostają niezależne.

| `--arr-smoothing` | Rekonstrukcja widoczności |
| --- | --- |
| `0` | Wyłączona, domyślna |
| `1` | Bilinear |
| `3` | Hermite |
| `4` | B-spline 4×4 |
| `5` | Bicubic Catmull-Rom 4×4 |

`--arr-smoothing-radius=0.75` oznacza pełną szerokość czterech symetrycznych podpikselowych miejsc rekonstrukcji, w **pikselach ekranu**. To filtr gotowej widoczności, a nie dodatkowe kierunki światła. Porównujemy normalne i odległość od płaszczyzny odbiornika, aby ograniczyć mieszanie przez krawędzie geometrii. Bilinear/Hermite wykonują 16 odczytów rekonstrukcji widoczności, warianty cubic 64; powtarzające się teksele może obsłużyć cache GPU. Nie interpolujemy surowej głębokości jako widoczności.

W trybie masek filtrujemy złożony wynik. Dla stałych znormalizowanych wag promieni i tego samego filtra liniowego zachodzi `F(sum(w_i*M_i))=sum(w_i*F(M_i))`, więc oddzielne tekstury dla każdego promienia nie są potrzebne. Hermite zmienia współczynniki interpolacji przestrzennej, a nie remapuje poziomy 0, 1/3, 2/3, 1. Ograniczenie Catmull-Rom do [0,1] stosujemy po złożeniu, nie osobno dla każdego promienia. Na zmieniających się normalnych lub przy indywidualnych filtrach promieni równoważność nie musi zachodzić.

Bez smoothingu oba ARR mają **jeden przebieg obliczania widoczności i zero map cienia**. Filtr dodaje drugi przebieg i aktywny ekranowy RG32F o rozmiarze `width*height*8` bajtów. Może złagodzić pasy małej liczby próbek, ale również osłabić poprawnie ostry cień. Catmull-Rom ma ujemne wagi i może dawać aureole. Domyślny smoothing pozostaje wyłączony. Nie ma TAA ani historii.

```sh
cmake-build-release/gfx-research-shadows --method=arr --arr-samples=4 --arr-smoothing=1
cmake-build-release/gfx-research-shadows --method=arr-triangulation --tri-samples=4 --tri-result-masks=1 --arr-smoothing=4 --arr-smoothing-radius=1.5
```

W benchmarku `--reference-samples=1024` ustala **osobny, zawsze niefiltrowany oracle ARR**, niezależny od budżetu i smoothingu badanej metody. Kontrole CPU i zbieżności używają tego oracle. Wiersz `arr` może więc mieć niezerowy błąd, gdy jego budżet jest mniejszy lub włączono smoothing.

## Zachowanie ostrego cienia

Zwykły geometric wyznacza jeden odstęp `G = exp(sum(log(g)) / count)` z dodatnich odległości skorygowanych o płaszczyznę odbiornika. Małe wartości wpływają na wynik mocniej niż przy średniej arytmetycznej, ale dalsza przeszkoda nadal może zwiększyć wspólny promień.

`geometric-layered` zbiera statystyki i zakres odstępów. Gdy stosunek największego do najmniejszego jest poniżej `--layer-separation=3`, używa jednego promienia geometric. W przeciwnym przypadku dzieli dodatnie odstępy progiem `sqrt(g_min*g_max)`, oblicza osobną średnią geometryczną bliskich i dalekich próbek i dwa promienie. Dla każdego tego samego kierunku dysku wykonuje test obu grup.

W 0.14.1 próbka dalszej przeszkody nie jest automatycznie uznawana za brak bliskiej przeszkody. Odróżniamy bliskie zasłonięcie `N`, potwierdzone światło `E` i ukrytą bliską warstwę `U`. Estymujemy `P_lit=sum(E)/(sum(E)+sum(N))`, a wynik wynosi `mean((E_i+U_i*P_lit)*V_far(i))`. Mały składnik stabilizuje dzielenie; przy całkowitym braku wiedzy wybieramy pełne światło dla bliskiej grupy. Jest to heurystyka warunkowej widoczności, nie dokładne odtworzenie ukrytej geometrii. Zachowujemy parowanie kierunków, zamiast mnożyć dwa końcowe uśrednione cienie.

Domyślne `--layer-depth=nearest` korzysta z jednej mapy D32F. Przy dwóch grupach koszt rośnie do dwóch wyszukiwań blockerów i dwóch testów filtra na kierunek. **Jeżeli górny obiekt zasłonił dolny w mapie, samo mieszanie nie odzyska utraconej informacji.** Dwie grupy nie opiszą dokładnie dowolnej liczby głębokości; nie jest to odpowiednik ray tracingu. Wartość `--warp=0` wyłącza projekcję NDC.

```sh
cmake-build-release/gfx-research-shadows --method=geometric-layered --layer-separation=3 --sample-pattern=stratified --sample-seed=35
```

### Opcja zachowania ukrytej głębokości

W GUI wybierz `Geometric depth bands (paired samples)` i włącz **`Packed min/max depth (R32UI)`**. Odpowiednik CLI:

```sh
cmake-build-release/gfx-research-shadows --method=geometric-layered --layer-depth=interval --sample-pattern=stratified
```

Ta opcja zmienia przechowywane dane. Zamiast D32F używa jednej tekstury R32UI, po 16 bitów UNORM na najbliższą i najdalszą powierzchnię wszystkich casterów w tekselu. To skrajne końce, nie dwie najbliższe warstwy i nie dwa obiekty. Shader widoczności odczytuje wyłącznie tę mapę i dane odbiornika; nie sprawdza analitycznych brył. Oba końce jednego teksela odczytuje razem i testuje oddzielnie. Nie wypełniamy automatycznie całego odcinka jako nieprzezroczystego.

Mapa nadal ma 4 MiB przy 1024² i powstaje w jednym przebiegu z jednej kamery światła. Rasteryzuje jednak wszystkie powierzchnie bez testu depth i redukuje końce atomowym `imageAtomicCompSwap`, z ponawianiem przy konflikcie. Nie ma dodatkowej mapy ani przebiegu rozmycia, ale generowanie mapy może być droższe. Alokacja i kompilacja pozostają poza czasem GPU. Dane `map_storage` i `stored_depth_components` w CSV rozróżniają D32F i dwa końce R32UI.

Przy zakresie głębokości 80 krok UNORM16 wynosi około 0.00122 jednostki świata. Tracimy precyzję D32F oraz wszystkie pośrednie powierzchnie, więc układ trzech i więcej przeszkód może być błędny. Zakładamy nieprzezroczyste powierzchnie; wariant nie jest rozwiązaniem dla transmitancji. Domyślna pozostaje zwykła mapa D32F.

Więcej niż jedna głębokość w mapie cienia ma wcześniejsze opracowania, np. [Weiskopf i Ertl, Shadow Mapping Based on Dual Depth Layers, 2003](https://diglib.eg.org/items/13ceeb27-b96e-4568-ac9f-363e99af7fc9). Tam przechowywano dwie warstwy najbliższe światłu do adaptacyjnego biasu. Nie przypisujemy nowości samemu zapisowi dwóch wartości; ocena nowości proponowanego filtrowania wymaga osobnego przeglądu literatury.

## Smoothing i próbki

Menu `Smoothing` dotyczy mean, geometric i geometric-layered; fixed PCF korzysta z wariantów widoczności. Filtrujemy wyniki porównań głębokości, zachowując osobny test i korekcję płaszczyzny dla każdego teksela.

| `--smoothing` | Wariant |
| --- | --- |
| `0` | Punktowy filtr, z rekonstrukcją kontaktu przy małym promieniu |
| `1` | Bilinear, domyślny |
| `2` | Bilinear oraz ważone statystyki blockerów |
| `3` | Hermite `3t²-2t³` i ważone statystyki blockerów |
| `4` | B-spline 4×4, cztery gather4 na tap |
| `5` | Bicubic Catmull-Rom 4×4, cztery gather4 na tap |
| `6` | Preset bilinear + losowe próbki dysku |
| `7` | Preset bilinear + stratyfikowane losowe próbki dysku |

`--geometric-smoothing` pozostaje aliasem. Fixed PCF udostępnia 0, 1, 3, 4, 5; ustawienia 2, 6, 7 oznaczają dla niego bilinear. Tryby 2 i 3 ważone są po położeniu próbki: mean używa `sum(w*g)/sum(w)`, geometric `exp(sum(w*log(g))/sum(w))`. B-spline ma nieujemne wagi. Catmull-Rom ma także ujemne wagi i może oscylować; ograniczenie wyniku do [0,1] nie usuwa wszystkich aureoli.

`--sample-pattern=vogel|random|stratified`, `--sample-seed=0`, `--samples=48`, `--blocker-samples=48`. Losowanie jest powtarzalne dla piksela i ziarna, bez TAA. Stratyfikacja dzieli dysk na komórki o równym polu także dla niekwadratowej liczby próbek. Blocker search i końcowy PCF mają osobne strumienie, natomiast dwie grupy nowej metody współdzielą kierunki końcowego filtra. `--rotate-samples` dodatkowo obraca wzorzec.

### Wygładzanie packed i ograniczanie ziarna

W `Geometric depth bands (paired samples)` przycisk **`Apply soft packed preset`** włącza R32UI, Hermite dla widoczności i statystyk blockerów, 64 próbki filtra i 64 próbki wyszukiwania oraz minimalny promień 1 teksela. Włącza też **`Share random sample pattern`** i wyłącza obracanie wzorca per piksel. Zachowuje aktualny kąt światła, rozdzielenie pasm i kamerę.

```sh
cmake-build-release/gfx-research-shadows --scene=crossing-four --soft-packed
cmake-build-release/gfx-research-shadows --scene=mixed-overlap --soft-packed --layer-separation=5.24 --angle=1
```

Samo losowanie punktów nie usuwa schodków ostrego cienia: przy promieniu mniejszym od teksela wiele próbek wykonuje ten sam test. Hermite interpoluje wyniki porównań, a `--layer-min-filter=1` wprowadza przestrzenny promień minimalny o rozmiarze lokalnego teksela drobniejszej z dwóch osi mapy, z uwzględnieniem NDC. Zakres to 0..4; domyślne 0 zachowuje promienie wynikające z geometrii. Większe wartości zmiękczają także fizycznie ostry kontakt, więc jest to regulacja wyglądu. Ograniczenie maksymalnego promienia nadal obowiązuje.

Niezależny dysk w każdym pikselu daje ziarno. **`Share random sample pattern`** (`--shared-pattern=1`) używa tych samych stratyfikowanych punktów w całym obrazie, przy nadal różnych strumieniach wyszukiwania i filtra. Pary bliskiej i dalekiej grupy zachowują wspólne kierunki. Przy wyłączonym `Rotate sample pattern` błąd próbkowania staje się spójny przestrzennie. Nie dodaje to tekstur, historii ani przebiegu rozmycia obrazu. Nie jest to dokładne całkowanie: przy zbyt małym budżecie mogą pozostać pasma i błędy mieszania. `--shared-pattern=0` zachowuje dotychczasowe losowanie per piksel, a `--sample-seed` wybiera powtarzalny wzorzec.

Filtr działa na cieniu. Nie wygładza zewnętrznego obrysu modelu z G-buffera ani nie odtwarza pośrednich głębokości utraconych przez min/max. Większa liczba obiektów nie zwiększa liczby próbek shadera widoczności przy stałych ustawieniach, ale zwiększa pracę generowania mapy, overdraw i możliwe konflikty atomowe. Dokładność nie jest niezależna od złożoności sceny.

## VSM, CSM i ESM

Nazwy odpowiadają metodom z [materiałów Romaina Vergne](http://romain.vergne.free.fr/teaching/IS/SI08-shadows.html). CSM oznacza tutaj **Convolution Shadow Maps**. Implementujemy warianty z liniową głębokością i separowalnym filtrem Gaussa, bez mipmap i MSAA. Promień `--prefilter-radius=4` jest stały w tekslach mapy; nie wynika z odległości blockera ani z rozmiaru Słońca. Te przykłady służą do porównania reprezentacji i artefaktów filtracji.

- **VSM:** RG32F zawiera `z` i `z²`. Po filtracji stosujemy jednostronne ograniczenie Czebyszewa, `variance/(variance+(t-mean)²)`, a dla `t<=mean` pełne światło. `--vsm-min-variance=1e-7`, `--bleed-reduction=0`. Nie stosujemy opcjonalnego zwiększenia drugiego momentu pochodnymi. Redukcja bleeding może przyciemniać poprawny półcień. [Donnelly i Lauritzen, omówienie i implementacja](https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-8-summed-area-variance-shadow-maps).
- **CSM:** domyślne `--csm-terms=16` oznacza 16 nieparzystych harmonicznych, czyli 32 współczynniki sinus/cosinus w ośmiu warstwach RGBA32F. Częstotliwości to `pi*(2k+1)`. Baza jest filtrowana przed odtworzeniem funkcji skokowej. Mapujemy głębokość do `0.05+0.9*z`, aby uniknąć okresowej nieciągłości na końcach przedziału. `--csm-bias=0.032` przesuwa test odbiornika; większy offset może usuwać cienie małych odstępów. Dopuszczamy parzyste liczby harmonicznych 4..16. Pozostają ringing i problemy z jasnością kontaktu. [Annen i in., Convolution Shadow Maps](https://jankautz.com/publications/csmEGSR07.pdf).
- **ESM:** R32F przechowuje `exp(c*(z-1))`, `--esm-exponent=80` (1..80). Stałe przesunięcie wykładnika i odtworzenie przez logarytm ograniczają zakres liczb FP32. `--esm-fallback=1` wykrywa odpowiedź ponad 1.02 i stosuje niestandardowe 2×2: ograniczenie każdego wyniku wykładniczego przed bilinearnym mieszaniem, zgodnie z klasyfikacją threshold z pracy. Dodatkowe rozmycie Gaussa nadal może dawać artefakty, których taki filtr nie usunie. [Annen i in., Exponential Shadow Maps](https://jankautz.com/publications/esm_gi08.pdf).

Przed kodowaniem tych trzech reprezentacji renderujemy do depth również podłoże, ponieważ odbiornik musi występować w filtrowanym rozkładzie głębokości. Ta zmiana dotyczy tylko VSM/CSM/ESM. Każda metoda używa jednej kamery światła, ale potrzebuje dwóch pomocniczych tekstur ping-pong. Kodowanie oraz poziome i pionowe rozmycie to **trzy dodatkowe dispatches**, wliczone do czasu przetwarzania mapy. FP32 w CSM zużywa więcej pamięci niż pierwotna implementacja ze współczynnikami 8-bitowymi.

| Metoda | Tekstury pomocnicze | Łączna pamięć przy mapie 1024², bez G-buffera |
| --- | --- | --- |
| PCSS / geometric / geometric-layered, również R32UI | Brak | 4 MiB |
| VSM | 2 × RG32F | 20 MiB |
| CSM, 16 harmonicznych | 2 × 8 warstw RGBA32F | 260 MiB |
| ESM | 2 × R32F | 12 MiB |

## Pozostałe warianty

DPCF jest portem kierunkowego filtra z [Filament v1.23.1](https://github.com/google/filament/blob/v1.23.1/shaders/src/shadowing.fs), commit `a67d50b9e29cf7bf2f0a6f0b7bf4668cdb87770a`: dysk Poissona, receiver-plane bias, penumbra ratio i hardened kernel. CHS przenosi [AMD ShadowFX](https://github.com/GPUOpen-Effects/ShadowFX), commit `c6d931c27275b1bb65ba0abfb71eaba86f355387`: `contactFixedGather4`, oryginalne tabele i rozmiary 7, 9, 11, 13, 15. Tabele CHS są w stałym UBO 10400 B. `--dpcf-taps=12`, `--dpcf-radius=6`, `--dpcf-ratio-scale=1`, `--chs-kernel=11`, `--chs-sun-size=4`. Support DPCF i sun size CHS skalujemy względem 1.5 stopnia. Są to porty filtrów, nie całych silników.

Eksperymentalne `--penumbra=inner` ogranicza mean i oba geometric do maski centralnego hard shadow i przelicza `clamp(2*V,0,1)`. Wariant zmienia fizyczny półcień i może usuwać cienkie cienie. Domyślny pozostaje `full`. Osobne programy metod i smoothingu są kompilowane przed zakresem pomiaru GPU. ARR zachowuje ważone całkowanie kierunków i dokładne przecięcia AABB; nie jest filtrem depth ani sprzętowym ray tracingiem.

## Eksperymenty i wyniki

W 0.16.1 przeszło **61 przypadków renderowania i 3 testy CTest**, w tym 9 przypadków analizatora Python. Profil stress obejmuje wszystkie 12 metod dla 0 i 65 obiektów. Zwykłe ARR oraz signed/binary/mask warianty triangulacji, z 3 i 4 kierunkami oraz smoothingiem, sprawdzono także dla 33 i 1000 obiektów. 12 testów CPU/GPU w `stress_gpu.csv` potwierdza granice i test promienia przez ostatnią instancję dla populacji 33, 65, 1024 i miliona; największa różnica granic wyniosła 0. Trafienie w indeks 999999, usunięcie jego grupy oraz chybienie dały wynik zgodny z niezależną pętlą CPU. To test dostępu do pełnej populacji, nie pomiar pełnej klatki ARR dla miliona obiektów.

Wszystkie 24 obrazy z zachowanych zestawów 0.16.0, po 12 dla sceny mieszanej i małej sceny proceduralnej, oraz ich oracle są identyczne. Dane nowego wydania są w `examples/arr_dense0161.csv` i `arr_dense0161_validation.json`. Czasy pochodzą z llvmpipe i służą walidacji, nie ocenie karty użytkownika.

W 0.16.0 przeszło **71 przypadków renderowania i 3 testy CTest**, w tym 8 przypadków analizatora Python. Testy obejmują 45 pomiarów wydajności i 26 porównań jakości. Wszystkie 10 metod mapowych wyrenderowano przy liczebnościach 0, 16, 1000 i 10000. Natywnie sprawdzono również **1 i 2 miliony instancji**, czyli 12 i 24 miliony bazowych trójkątów na etap geometrii. Te dwie próby używały obrazu i mapy 32×32, warp=0 oraz poziomu teselacji 1 na llvmpipe; potwierdzają poprawność wysłania populacji i wyników, nie wydajność RX 9070 XT. Zakres do 10 milionów sprawdza też test geometrii bez tworzenia pełnej listy brył.

Przy 16 proceduralnych obiektach wykonano porównanie wszystkich 12 metod z ARR 512. Maksymalna różnica oracle CPU/GPU wyniosła `1.02e-7`, a test zbieżności 256/512 miał RMSE `0.000837`. Wszystkie 12 dotychczasowych obrazów sceny mieszanej, packed w scenie czterech belek oraz ich oracle pozostały identyczne z 0.15.3. Nie zmieniono sposobu mieszania cieni ani utraty pośrednich głębokości.

Dane nowej wersji: `examples/stress0160.csv`, `stress_quality0160.csv`, `stress0160_validation.json` i `stress0160.png`. Raport wydajności pomija RMSE i oracle; zawiera G-buffer, generowanie mapy, filtr oraz sumę etapów. Wyniki z renderera programowego są oznaczone w metadanych i odrzucane przez `--require-hardware`.

W 0.15.3 przeszło **51 przypadków renderowania i 3 testy CTest**. Dane `examples/soft_packed0153.csv`, manifest `soft_packed0153_validation.json` oraz obrazy `soft_packed0153.png` i `crossing_four0153.png` zawierają porównania z surowym ARR 1024, także dla wszystkich 12 metod na nowej scenie. Dotychczasowe 12 obrazów i oracle są identyczne z zachowanym zestawem 0.15.2. Test geometrii potwierdza przecięcie wszystkich czterech belek przez wspólny promień światła. Kontrole GPU reprezentacji, próbkowania i filtrów oraz kontrole CPU oracle nadal przechodzą.

W zbliżeniu `front`, przy kącie 1 stopnia i rozdzieleniu pasm 5.24, packed z punktowymi porównaniami i 32/32 losowymi próbkami ma RMSE `0.009445`; nowy preset `0.007107`. W drugim zbliżeniu odpowiednio `0.012490` i `0.008422`. W tym drugim ujęciu samo współdzielenie dysku, przy tych samych 64/64 próbkach, Hermite i promieniu minimalnym, zmniejsza wysokoczęstotliwościowy składnik błędu z `0.025838` do `0.021809`. To miara błędu względem ARR w półcieniu podłoża, nie niezależny pomiar samego ziarna; dokładny wzór i maska są w manifeście. B-spline w tym ujęciu daje RMSE `0.008054`, kosztem czterech gather4 na tap i mocniejszego wygładzenia.

Na `crossing-four` packed point ma RMSE `0.009108`, preset `0.008785`. Pozostają błędy: największe lokalne rozjaśnienie po dodaniu górnej grupy wynosi `0.178641` w skali widoczności 0..1, naruszenia monotoniczności obejmują około `0.406%` pikseli odbiornika. Przy kącie 3 stopni RMSE presetu rośnie do `0.015349`. Cztery warstwy pokazują granice zapisu min/max i dwóch pasm, mimo gładszych krawędzi. Preset zwiększa liczbę próbek z 32 do 64 względem porównywanego punktowego wariantu; zachowuje jedną mapę 4 MiB przy 1024² i jeden przebieg widoczności. Pomiary llvmpipe służą walidacji, nie ocenie szybkości GPU użytkownika.

```sh
python tools/run_experiment.py --exe=cmake-build-release/gfx-research-shadows --profile=quick
python tools/run_experiment.py --exe=cmake-build-release/gfx-research-shadows --profile=full
python tools/compare_near_shadow.py results/close-mixed results/close-low
```

Profile: smoke 36, quick 60, full 1152 przypadków. Linux obsługuje `--headless` przez EGL. `--map-backend=analytic` pozostaje oddzielnym oracle mapy brył i płaszczyzny. Runner zapisuje ustawienia, kamerę, hashe źródeł i programu, surowe czasy, PNG, liniowe PFM, błędy względem ARR oraz pamięć i liczbę przebiegów. Czas mapy obejmuje jej generowanie i ewentualne kodowanie/rozmycie, bez G-buffera, kompilacji, odczytu i GUI. CSV rozróżnia liczbę zapytań kątowych, smoothing, jego pamięć i bufor kierunków.

Weryfikacja obejmuje dotychczasowe 320 kontroli DPCF/CHS i rekonstrukcji, 1568 kontroli samplerów, 384 kontrole VSM/CSM/ESM oraz 256 kontroli zapisu R32UI. Dochodzi 160 porównań rekonstrukcji z niezależnym wzorem na pole odcinka koła, 96 kontroli znaku miary względem CPU ray-box oraz 1024 kontrole smoothingu ARR z niezależnym splotem CPU i skokiem głębokości. Test R32UI wykonuje 4096 konkurujących aktualizacji GPU. ARR ma niezależny ray-box CPU i test zbieżności. `--require-hardware` i `--publication` odrzucają czasy renderera programowego.

W 0.15.2 przeszło **46 przypadków renderowania i 3 testy CTest**. Nowe dane są w `examples/result_masks0152*`: porównanie filtrów dla 3 i 4 promieni, niezależne testy CPU/GPU oraz surowe wyniki. 160 nowych testów sprawdza średnią masek dla 3/4/7/13/25 kierunków; największy błąd względem CPU wyniósł `1.51e-7`. Pozostałe 256 testów kątowych i 1024 testy filtra nadal przechodzą. 12 obrazów wszystkich dotychczasowych metod z wyłączonym nowym trybem jest identycznych z zachowanym ujęciem 0.15.0; oracle pozostaje identyczny po zmianach filtrów.

W zbliżeniu mieszanych cieni (`front`, 640×480) cztery maski trójkąta ze środkiem mają RMSE `0.013164`; bilinear o szerokości 0.75 px zmniejsza go do `0.010083`. B-spline daje `0.010262`, a szerokość 3 px pogarsza wynik do `0.020295`. Dotychczasowa rekonstrukcja z ciągłej miary geometrii nadal jest dokładniejsza w tym ujęciu: `0.002490`. To wygładzanie przestrzenne ogranicza stopnie, ale nie odzyskuje brakujących kierunków światła. Wyniki z llvmpipe służą walidacji, nie ocenie wydajności RX 9070 XT.

Przykłady 0.15.0 są w `examples/angular0150.png`, `angular_patterns0150*`, `angular_budgets0150.csv`, `comparison0150*` i danych GPU z numerem 0150. Przeszło **112 przypadków renderowania i 3 testy CTest**. 36 obrazów pozostałych metod opartych na mapach jest identycznych z zachowanym zestawem 0.14.0. Źródła shaderów geometric-layered i R32UI pozostały identyczne z 0.14.1. Zmiana oracle z wielu przebiegów na jeden jest kontrolowana względem poprzednich obrazów; warianty budżetu i smoothingu zachowują identyczny surowy oracle.

Na zbliżeniu pod belką zwykłe ARR 4 ma RMSE 0.01321, triangulacja 4 z wartościami binarnymi 0.02569, a triangulacja 4 z ciągłą miarą 0.00249. Bilinear zmniejsza błąd zwykłego ARR do 0.01056, lecz zwiększa błąd dobrej rekonstrukcji triangulowanej do 0.00568. W scenie dense przy promieniu 5 stopni triangulacja 4 ma błąd 0.01785, a wariant 13 zapytań 0.00210. Dodatkowe dane i budżet mają znaczenie; nie przypisujemy całej poprawy samemu układowi trójkąta. Pomiary wykonano na llvmpipe, nie RX 9070 XT.

Przykłady 0.14.1: `examples/mixing0141*`, `comparison0141*`, `near_shadow0141.csv` i `interval0141_gpu.csv`. Testy: 186 przypadków renderowania, wszystkie tryby smoothingu przedziałów, raster i oracle analityczny, `inner`, NDC i projekcja bez warpu oraz 3 testy CTest. 40 obrazów pozostałych metod jest identycznych z 0.14.0. Wyniki pochodzą z llvmpipe i nie są pomiarami RX 9070 XT. Starsze pliki zachowują wersję w swoim zapisie walidacji.

W ujęciu odtwarzającym jasny pas pod belką, dla 6051 pikseli wspólnego pełnego cienia dolnego obiektu RMSE względem ARR wynosi 0.13260 dla 0.14.0, 0.11005 po zmianie samego mieszania D32F i 0.00389 dla mieszania z R32UI. RMSE całego obrazu wynosi odpowiednio 0.01979, 0.01694 i 0.00678. Maska wynika z ARR dla scen z górnym obiektem i bez niego, przy tej samej kamerze. Nie jest ręcznie wyciętym pasem.

W czterech scenach quick średnie RMSE nowego wariantu wynosi 0.008255 dla D32F i 0.005915 dla R32UI. Nie oznacza to uniwersalnej przewagi: w drugim zbliżeniu `close-mixed` R32UI ma RMSE 0.00903 wobec 0.00855 dla 0.14.0. Dotychczasowy zwykły geometric pozostaje dostępny i domyślny.
