# Scientific Paper Topic Modeling + Lightweight Recommender (NLP Unsupervised + Supervised)

**Autor:** Joanna Konieczny

## 1. Streszczenie

Celem projektu jest automatyczne odkrywanie głównych tematów badawczych (klastrów tematycznych) w zbiorze artykułów naukowych oraz budowa prostego systemu rekomendacyjnego. System ma umożliwiać identyfikację struktur tematycznych w danych tekstowych oraz wskazywanie użytkownikowi artykułów o zbliżonej tematyce.

W podejściu modelowym zastosowane zostaną zarówno metody klasyczne, jak i nowoczesne techniki oparte na embeddingach. Jako baseline wykorzystany zostanie model LDA, natomiast główne podejście będzie opierało się na wektorowych reprezentacjach zdań (sentence embeddings).

Projekt wykonano w języku Julia.

## 2. Wstęp teoretyczny

Modelowanie tematów (topic modeling) jest jedną z kluczowych technik przetwarzania języka naturalnego (NLP), której celem jest automatyczne odkrywanie ukrytej struktury semantycznej w zbiorach dokumentów tekstowych. Metody te pozwalają grupować dokumenty w spójne tematy oraz identyfikować charakterystyczne słowa dla każdego z nich, bez konieczności wcześniejszego etykietowania danych. W tej sekcji krótko przedstawiono podstawy teoretyczne metod eksploracji tematów używanych w projekcie.

### Latent Dirichlet Allocation (LDA)

Jest to model probabilistyczny oparty na założeniu, że każdy dokument jest mieszaniną tematów, a każdy temat jest rozkładem prawdopodobieństwa nad słowami. LDA wykorzystuje rozkłady Dirichleta jako priory dla rozkładów temat–dokument oraz temat–słowo, a proces generatywny zakłada, że słowa w dokumentach są generowane poprzez losowanie tematów i odpowiadających im słów. W praktyce LDA pozwala na interpretowalne grupowanie dokumentów, jednak jego skuteczność zależy od założeń dotyczących współwystępowania słów i często bywa ograniczona w przypadku złożonych semantycznie danych tekstowych.

### BERTopic

Alternatywą dla podejść probabilistycznych są metody oparte na reprezentacjach wektorowych, w szczególności modele wykorzystujące embeddingi. Przykładem takiego podejścia jest BERTopic, który łączy reprezentacje semantyczne tekstu z algorytmami klasteryzacji. W pierwszym etapie dokumenty są przekształcane w gęste wektory (embeddingi), zazwyczaj z wykorzystaniem modeli językowych typu transformer. Następnie redukcja wymiarowości (np. UMAP) oraz algorytmy klasteryzacji (np. HDBSCAN) są wykorzystywane do grupowania dokumentów w tematy. Każdy temat jest następnie reprezentowany przez słowa wyodrębnione na podstawie miar ważności, takich jak c-TF-IDF. W przeciwieństwie do LDA, BERTopic lepiej uchwytuje semantyczne podobieństwa między dokumentami, co czyni go bardziej odpornym na synonimię i złożone zależności językowe.

W ramach projektu zastosowano uproszczoną i zmodyfikowaną wersję podejścia BERTopic, dostosowaną do ograniczeń obliczeniowych oraz charakteru eksperymentu. Zamiast pełnego pipeline’u zastosowano bezpośrednią reprezentację dokumentów w przestrzeni embeddingów, a następnie grupowanie oparte na podobieństwie kosinusowym względem centroidów tematów. Takie rozwiązanie czyni model bardziej lekkim obliczeniowo, ale jednocześnie bardziej zależnym od jakości reprezentacji embeddingowych.

## 3. Opis danych

W projekcie wykorzystany został zbiór metadanych artykułów z serwisu arXiv udostępniony w ramach Kaggle (Cornell University arXiv dataset: https://www.kaggle.com/datasets/Cornell-University/arxiv). Dane obejmują informacje o publikacjach naukowych, takie jak tytuły, abstrakty oraz przypisane kategorie.

Cały zbiór zawiera informacje o 1.7 milionie artykułów z lat 2007-2019 i z 8 kategorii głównych.

> ![alt text](image.png)
> Liczba artykułów opublikowanych w danym roku

> ![alt text](image-1.png)
> Liczba artykułów w danej kategorii

Na potrzeby projektu zdecydowano się uwzględnić artykuły opublikowane po roku 2011 z kategorii głównej `math` (łącznie 4350 artykułów).

## 4. Przebieg analizy

W ramach przeprowadzonej analizy zrealizowano kompleksowy pipeline przetwarzania danych tekstowych obejmujący zarówno modele klasyczne, jak i podejścia oparte na reprezentacjach wektorowych. Celem było porównanie skuteczności różnych metod modelowania tematów oraz ocena ich przydatności w zadaniu grupowania dokumentów i budowy prostego systemu rekomendacyjnego.

Analiza została podzielona na kilka etapów obejmujących przygotowanie danych, budowę reprezentacji dokumentów, uczenie modeli oraz ocenę jakości wyników. W szczególności porównano klasyczny model probabilistyczny LDA z nowoczesnym podejściem opartym na embeddingach oraz klasteryzacji semantycznej (BERTopic-like), a także jego rozszerzenie z wykorzystaniem uczenia kontrastywnego.

### 4.1 Wstępne czyszczenie danych

Rozważany dataset został zmodyfikowany w następujący sposób:

- ograniczenie kategorii (`math`) i roku startowego (2012),
- usunięcie *filler words* - wyrazów powszechnie występujących w języku angielskim, które mogłyby negatywnie wpłynąć na wyniki klasyfikacji,
- parsowanie autorów przez połączenie imienia i nazwiska w 1 ciąg znaków,
- normalizowanie tekstu przez usunięcie szumu i zamianę wszystkich liter na małą literę.

> ![alt text](image-2.png)
> Pierwsze 5 rzędów wynikowego dataframe'a

> ![alt text](image-3.png)
> Liczba artykułów w danej kategorii

Tak przygotowany zbiór został podzielony na zbiory treningowy i testowy.

### 4.2 LDA

W projekcie użyto gotowej implementacji LDA z odgórnie nałożoną liczbą docelowych tematów $K=30$.

> ![alt text](image-4.png)
> Najważniejsze słowa dla przykładowej kategorii

> ![alt text](image-5.png)
> Word Map z 5 najważniejszymi słowami w każdej kategorii

> Wyniki ewaluacji modelu:
> ``` 
> Perplexity: 9.154762961445211
> Accuracy (mapped topics): 0.38059485994802195
> Topic purity: 0.38059485994802195
> Topic diversity: 0.8633333333333333
> Topic coherence (proxy): 63.07748
> ```

Model LDA osiąga niską wartość perplexity, co wskazuje na dobre dopasowanie statystyczne do rozkładu danych. Różnorodność tematów (topic diversity) jest wysoka (0.86), co sugeruje dobrze rozdzielone rozkłady słów dla poszczególnych tematów. Jednak zgodność tematów z etykietami (topic-to-label alignment) pozostaje umiarkowana (0.38), co jest oczekiwane ze względu na nienadzorowany charakter LDA oraz rozbieżność między ukrytymi tematami a rzeczywistymi etykietami klas. Koherencja tematów wskazuje na semantycznie spójne grupy słów, co wspiera interpretowalność wyodrębnionych tematów.

### 4.3 BERTopic

Implementacja modelu, z której korzystano w projekcie znajduje się w pliku `BERTopicJL.jl`.

Pipeline składa się z kilku etapów: embeddingu dokumentów, uczenia kontrastowego reprezentacji oraz klasteryzacji w przestrzeni embeddingów. Model wykorzystuje następujące parametry:

- `vocab_size`: rozmiar słownika wejściowego, definiuje wymiar warstwy embeddingowej.
- `embed_dim`: wymiar przestrzeni embeddingów (domyślnie 256), określa rozmiar wektorowej reprezentacji dokumentów.
- `train_ratio`: proporcja danych treningowych do testowych (np. 0.8), używana do podziału zbioru danych.
- `epochs`: liczba epok uczenia modelu kontrastowego (domyślnie 10), kontroluje czas treningu i dopasowanie embeddingów.
- `batch_size`: liczba próbek w jednej iteracji optymalizacji (domyślnie 32), wpływa na stabilność gradientów i zużycie pamięci.
- `lr`: learning rate optymalizatora Adam (np. 1e-3), kontroluje szybkość aktualizacji wag.
- `threshold` (klasteryzacja): próg podobieństwa kosinusowego (np. 0.75), decyduje o przypisaniu dokumentu do istniejącego klastra tematycznego.

Dodatkowo pipeline wykorzystuje cosine similarity jako miarę podobieństwa w przestrzeni embeddingów, contrastive loss, który zwiększa separację między dokumentami o różnych etykietach i zbliża dokumenty o tej samej etykiecie oraz hierarchiczną klasteryzację przyrostową (greedy clustering) do budowy tematów.

> ![alt text](image-6.png)
> Najważniejsze słowa dla przykładowej kategorii

> Wyniki ewaluacji modelu:
> ``` 
> Accuracy: 0.8383371824480369
> Topic purity: 0.8383371824480369
> Topic diversity: 0.00743321718931475
> Topic coherence: 0.9431105781433983
> ```

Model osiągnął accuracy na poziomie 0.838, co wskazuje na dobrą zgodność przypisań tematów z rzeczywistymi etykietami klas. Wartość purity na identycznym poziomie sugeruje, że wygenerowane klastry są w dużym stopniu jednorodne pod względem przypisanych kategorii, co oznacza relatywnie niską wewnętrzną mieszaność tematów. Wysoka wartość coherence (0.943) wskazuje na silne rozdzielenie centroidów tematów w przestrzeni embeddingów, co oznacza, że wyodrębnione klastry są dobrze separowalne semantycznie. Jednocześnie bardzo niska wartość diversity (0.0074) sugeruje ograniczoną różnorodność reprezentowanych cech w topowych elementach tematów, co może wynikać z przyjętej definicji tej metryki opartej na przestrzeni embeddingów zamiast rzeczywistych rozkładów słów i wskazuje na potrzebę ostrożnej interpretacji tej miary w kontekście modelu opartego na embeddingach.

## 5. Wnioski
- Modele tematyczne skutecznie identyfikują struktury semantyczne w danych tekstowych, jednak ich charakterystyka różni się w zależności od podejścia: LDA opiera się na rozkładach probabilistycznych słów, natomiast modele oparte o embeddingi (BERTopic-like) wykorzystują reprezentacje semantyczne, co zwiększa spójność znaczeniową klastrów.
- LDA osiąga bardzo dobre dopasowanie statystyczne danych (niska perplexity), co potwierdza poprawne modelowanie rozkładu słów. Jednocześnie nie przekłada się to bezpośrednio na wysoką zgodność z etykietami klas, co wynika z nienadzorowanego charakteru modelu.
- Odwzorowanie tematów na etykiety klas jest umiarkowane, co wskazuje, że struktury latentne odkryte przez LDA nie są jednoznacznie zgodne z rzeczywistą klasyfikacją dokumentów. Jest to typowe dla metod unsupervised i wynika z różnicy między „tematem” a „kategorią” w danych rzeczywistych.
- Modele oparte o embeddingi wykazują lepszą separację semantyczną dokumentów, co przekłada się na bardziej spójne tematy i lepszą interpretowalność lokalnych grup dokumentów w przestrzeni wektorowej.
- Wysoka różnorodność tematów wskazuje na dobrą separację przestrzeni semantycznej, co oznacza, że model nie generuje silnie redundantnych klastrów i skutecznie rozróżnia główne obszary tematyczne.
- Głównym ograniczeniem podejścia LDA jest brak uwzględnienia semantyki kontekstowej, co prowadzi do mieszania semantycznie odległych słów w jednym temacie oraz ogranicza jakość dopasowania do etykiet klas.


## 6. Bibliografia

1. Implementacja LDA: https://github.com/yng87/LDA_CGS.jl/blob/master/README.md

2. Overview modelu BERTopic:  https://maartengr.github.io/BERTopic/algorithm/algorithm.html#visual-overview 
