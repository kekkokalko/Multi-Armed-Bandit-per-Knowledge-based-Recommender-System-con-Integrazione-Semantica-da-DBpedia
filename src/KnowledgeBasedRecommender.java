import java.io.*;
import java.util.*;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.jena.query.*;

/*
 * Classe principale del sistema di raccomandazione basato su conoscenza (Knowledge-Based Recommender).
 * Questo sistema combina informazioni del dataset MovieLens con conoscenza estratta da DBpedia tramite SPARQL
 * per costruire un profilo utente e suggerire film rilevanti, utilizzando un approccio Multi-Armed Bandit (softmax).
 */
public class KnowledgeBasedRecommender {
    // Percorso del file contenente l'elenco dei film del dataset MovieLens
    private static final String MOVIELENS_DATASET = "Dataset/movies.csv";
    // Percorso del file contenente i rating assegnati dagli utenti ai film
    private static final String RATINGS_DATASET = "Dataset/ratings.csv";
    // Percorso del file contenente i tag associati ai film
    private static final String TAGS_DATASET = "Dataset/tags.csv";
    // Endpoint SPARQL pubblico di DBpedia, usato per interrogare la base di conoscenza
    private static final String DBPEDIA_SPARQL_ENDPOINT = "http://dbpedia.org/sparql";
    // ID dell'utente target su cui si basa la raccomandazione personalizzata
    private static final int TARGET_USER_ID = 1;
    // Mappa che associa ID di film al loro titolo
    private final Map<Integer, String> movieIdToTitle = new HashMap<>();
    // Mappa che associa ID di film a insiemi di tag (potrebbero rappresentare concetti semantici associati)
    private final Map<Integer, Set<String>> movieTags = new HashMap<>();
    // Mappa che associa ID di film ai loro generi (stringa descrittiva)
    private final Map<Integer, String> movieGenres = new HashMap<>();
    // Insieme di ID dei film che sono stati valutati positivamente dall'utente target
    private final Set<Integer> likedMovieIds = new HashSet<>();
    // Cache dei risultati SPARQL per titolo (actor/director)
    private final Map<String, Map<String, List<String>>> dbpediaCache = new HashMap<>();
    // Generatore di numeri casuali
    private final Random random = new Random();

    // Metodo main: punto di ingresso dell'applicazione, avvia l'esperimento
    public static void main(String[] args) {
        new KnowledgeBasedRecommender().run();
    }

    public void run() {
        System.out.println("🚀 Avvio del sistema di raccomandazione basato sulla conoscenza...");
        // Carica i film, i tag e i rating dal dataset MovieLens
        loadMovieLensData();
        loadTagsData();
        loadRatingsData();
        // Estrae i generi preferiti (top 3) dagli utenti
        System.out.println("📥 Estrazione dei generi preferiti...");
        Set<String> topGenres = extractTopGenres();
        // Estrae i tag preferiti (top 3) dagli utenti
        System.out.println("📥 Estrazione dei tag preferiti...");
        Set<String> topTags = extractTopTags();
        // Mostra la frequenza dei generi
        System.out.println("📊 Classifica generi:");
        printFrequencies(cleanFrequencies(extractAllGenres()));
        // Mostra la frequenza dei tag
        System.out.println("📊 Classifica tag:");
        printFrequencies(cleanFrequencies(extractAllTags()));
        // Estrae i principali attori da DBpedia (proprietà dbo:starring)
        System.out.println("📡 Estrazione degli attori e registi principali da DBpedia...");
        Set<String> topActors = extractTopEntities( "starring", true);
        // Estrae i principali registi da DBpedia (proprietà dbo:director)
        Set<String> topDirectors = extractTopEntities( "director", false);
        System.out.println("🎯 Top 3 generi: " + topGenres);
        System.out.println("🏷️ Top 3 tag: " + topTags);
        System.out.println("🎭 Top 3 attori: " + topActors);
        System.out.println("🎬 Top 3 registi: " + topDirectors);
        // Crea un set per contenere i candidati provenienti da DBpedia
        Map<String, Set<String>> filmAspectMap = new HashMap<>();
        Set<String> dbpediaCandidates = new HashSet<>();
        System.out.println("🎞️ Film trovati per generi preferiti:");
        final int[] genreCount = {0}; // Contatore dei film trovati per generi, definito come array per essere modificabile nel lambda
        for (String genre : topGenres) { // Itera su ciascun genere preferito dell’utente
            // Esegue la query SPARQL unificata passando solo il genere corrente come filtro
            Set<String> results = unifiedSparqlSearch(
                    Set.of(genre.replaceAll("\"", "").trim()), // Rimuove eventuali virgolette e spazi dal nome del genere
                    Set.of(), // Nessun filtro per tag
                    Set.of(), // Nessun filtro per attori
                    Set.of(), // Nessun filtro per registi
                    Set.of()  // Nessun filtro per decenni
            );
            // Per ogni titolo ottenuto nei risultati
            results.forEach(f -> {
                // Pulisce il titolo rimuovendo lingua, virgolette e anno
                String cleaned = f.replaceAll("@en$", "")
                        .replaceAll("^\"+|\"+$", "")
                        .replaceAll("\\(.*?\\)", "")
                        .trim();
                // Aggiorna la mappa film→aspetti, aggiungendo "genre" come aspetto soddisfatto
                filmAspectMap.computeIfAbsent(cleaned, _ -> new HashSet<>()).add("genre");
                // Stampa il titolo del film associato al genere attuale
                System.out.println("   [Genre → " + genre + "] " + cleaned);
                // Incrementa il contatore per i film di genere trovati
                genreCount[0]++;
            });
            // Aggiunge tutti i risultati all’insieme generale dei candidati
            dbpediaCandidates.addAll(results);
        }
        System.out.println("🏷️ Film trovati per tag preferiti:");
        // Inizializza un contatore per tenere traccia del numero di film trovati tramite tag
        final int[] tagCount = {0};
        // Itera su ciascun tag presente tra i top preferiti
        for (String tag : topTags) {
            // Esegue una ricerca SPARQL specificando solo il tag corrente
            Set<String> results = unifiedSparqlSearch(
                    Set.of(),                                      // nessun genere
                    Set.of(tag.replaceAll("\"", "").trim()),       // solo il tag attuale
                    Set.of(),                                      // nessun attore
                    Set.of(),                                      // nessun regista
                    Set.of()                                       // nessun decennio
            );
            // Itera su ciascun film restituito dalla query
            results.forEach(f -> {
                // Pulisce il titolo del film rimuovendo etichette e caratteri speciali
                String cleaned = f.replaceAll("@en$", "")
                        .replaceAll("^\"+|\"+$", "")
                        .replaceAll("\\(.*?\\)", "")
                        .trim();
                // Aggiunge alla mappa film → aspetti l'informazione che questo film soddisfa un "tag"
                filmAspectMap.computeIfAbsent(cleaned, _ -> new HashSet<>()).add("tag");
                // Stampa l'associazione tra il tag preferito e il film corrispondente
                System.out.println("   [Tag → " + tag + "] " + cleaned);
                // Incrementa il contatore dei film trovati tramite tag
                tagCount[0]++;
            });
            // Aggiunge i film trovati alla lista generale dei candidati da DBpedia
            dbpediaCandidates.addAll(results);
        }
        System.out.println("🎭 Film trovati per attori preferiti:");
        // Inizializza un contatore per il numero di film trovati tramite attori
        final int[] actorCount = {0};
        // Per ciascun attore nella top 3
        for (String actor : topActors) {
            // Esegue una ricerca SPARQL unificata specificando solo l'attore corrente
            Set<String> results = unifiedSparqlSearch(
                    Set.of(),                                       // nessun genere
                    Set.of(),                                       // nessun tag
                    Set.of(actor.replaceAll("\"", "").trim()),      // solo l'attore attuale
                    Set.of(),                                       // nessun regista
                    Set.of()                                        // nessun decennio
            );
            // Itera sui risultati trovati per l'attore
            results.forEach(f -> {
                // Pulisce il titolo del film rimuovendo eventuali metadati
                String cleaned = f.replaceAll("@en$", "")
                        .replaceAll("^\"+|\"+$", "")
                        .replaceAll("\\(.*?\\)", "")
                        .trim();
                // Aggiorna la mappa degli aspetti del film, aggiungendo l'aspetto "actor"
                filmAspectMap.computeIfAbsent(cleaned, _ -> new HashSet<>()).add("actor");
                // Stampa il film con l'attore associato
                System.out.println("   [Starring → " + actor + "] " + cleaned);
                // Incrementa il contatore dei film trovati tramite attori
                actorCount[0]++;
            });
            // Aggiunge i film trovati all'elenco generale dei candidati
            dbpediaCandidates.addAll(results);
        }
        System.out.println("🎬 Film trovati per registi preferiti:");
        // Inizializza un contatore per il numero di film trovati tramite registi
        final int[] directorCount = {0};
        // Per ciascun regista nella top 3
        for (String director : topDirectors) {
            // Esegue una ricerca SPARQL unificata specificando solo il regista corrente
            Set<String> results = unifiedSparqlSearch(
                    Set.of(),                                       // nessun genere
                    Set.of(),                                       // nessun tag
                    Set.of(),                                       // nessun attore
                    Set.of(director.replaceAll("\"", "").trim()),   // solo il regista attuale
                    Set.of()                                        // nessun decennio
            );
            // Itera sui risultati trovati per il regista
            results.forEach(f -> {
                // Pulisce il titolo del film da eventuali etichette e metadati
                String cleaned = f.replaceAll("@en$", "")
                        .replaceAll("^\"+|\"+$", "")
                        .replaceAll("\\(.*?\\)", "")
                        .trim();
                // Aggiorna la mappa degli aspetti del film aggiungendo "director"
                filmAspectMap.computeIfAbsent(cleaned, _ -> new HashSet<>()).add("director");
                // Stampa il film trovato e il regista associato
                System.out.println("   [Director → " + director + "] " + cleaned);
                // Incrementa il contatore dei film trovati per registi
                directorCount[0]++;
            });
            // Aggiunge i risultati trovati all'elenco generale dei candidati DBpedia
            dbpediaCandidates.addAll(results);
        }
        // Estrae film vicini temporalmente agli anni preferiti (±10 anni)
        // Crea una mappa che conta quanti film apprezzati dall'utente ricadono in ciascun anno
        Map<Integer, Long> yearFrequency = likedMovieIds.stream()
                .map(movieIdToTitle::get) // Ottiene il titolo del film partendo dall'ID
                .filter(Objects::nonNull) // Scarta i titoli nulli (caso in cui l'ID non ha titolo)
                .map(title -> {
                    // Estrae l'anno tra parentesi nel titolo (es. "Titanic (1997)" → "1997")
                    String match = title.replaceAll(".*\\((\\d{4})\\).*", "$1");
                    try {
                        return Integer.parseInt(match); // Converte la stringa in intero
                    } catch (Exception e) {
                        return null; // Se fallisce la conversione, ritorna null
                    }
                })
                .filter(Objects::nonNull) // Rimuove eventuali valori null (errori di parsing)
                .collect(Collectors.groupingBy(y -> y, Collectors.counting())); // Raggruppa per anno e conta quante volte compare
        System.out.println("📆 Anni preferiti:");
        yearFrequency.entrySet().stream()
                .sorted((e1, e2) -> Long.compare(e2.getValue(), e1.getValue())) // Ordina gli anni per frequenza decrescente
                .limit(5) // Considera solo i primi 5 anni più frequenti
                .forEach(e -> System.out.println("   → " + e.getKey() + " (" + e.getValue() + " film)")); // Stampa ogni anno e il numero di film associati
        // Aggiunge i film legati agli anni preferiti
        System.out.println("📆 Film trovati per anni preferiti:");
        // Inizializza un contatore per il numero di film trovati per aspetto "anno"
        final int[] yearCount = {0};
        // Estrae i 5 anni più ricorrenti tra i film apprezzati (in base alla frequenza)
        Set<Integer> preferredYears = yearFrequency.entrySet().stream()
                .sorted(Map.Entry.<Integer, Long>comparingByValue().reversed()) // Ordina per frequenza decrescente
                .limit(5) // Considera solo i primi 5 anni più frequenti
                .map(Map.Entry::getKey) // Estrae l'anno (chiave della mappa)
                .collect(Collectors.toSet()); // Raccoglie gli anni in un set
        // Lancia la query SPARQL unificata filtrando per gli anni preferiti (ciascuno con range ±5 anni)
        Set<String> results = unifiedSparqlSearch(
                Set.of(),          // Nessun filtro per genere
                Set.of(),          // Nessun filtro per tag
                Set.of(),          // Nessun filtro per attori
                Set.of(),          // Nessun filtro per registi
                preferredYears     // Filtro per anni preferiti (±5 anni già gestiti nella query)
        );
        // Per ogni film ottenuto dalla query per anni
        results.forEach(f -> {
            // Pulisce il titolo del film rimuovendo metadati e formattazioni
            String cleaned = f.replaceAll("@en$", "")
                    .replaceAll("^\"+|\"+$", "")
                    .replaceAll("\\(.*?\\)", "")
                    .trim();
            // Aggiorna la mappa degli aspetti indicando che il film è legato a un anno preferito
            filmAspectMap.computeIfAbsent(cleaned, _ -> new HashSet<>()).add("year");
            // Stampa il film trovato per anno
            System.out.println("   [Year] " + cleaned);
            // Incrementa il contatore dei film trovati per l’aspetto anno
            yearCount[0]++;
        });
        // Aggiunge i risultati trovati all’elenco generale dei candidati
        dbpediaCandidates.addAll(results);
        // Rimuove i film senza alcun aspetto associato
        Set<String> cleanedToRemove = dbpediaCandidates.stream()
                .filter(f -> {
                    // Pulisce il titolo rimuovendo etichette linguistiche, virgolette e parentesi
                    String cleaned = f.replaceAll("@en$", "")
                            .replaceAll("^\"+|\"+$", "")
                            .replaceAll("\\(.*?\\)", "")
                            .trim();
                    // Verifica se il film non ha aspetti oppure non è presente nella mappa
                    return !filmAspectMap.containsKey(cleaned) || filmAspectMap.get(cleaned).isEmpty();
                })
                .collect(Collectors.toSet()); // Raccoglie tutti i titoli da rimuovere
        // Rimuove dalla lista dei candidati quelli che non hanno aspetti semantici associati
        dbpediaCandidates.removeAll(cleanedToRemove);
        // Stampa un riepilogo numerico di quanti film sono stati trovati per ogni aspetto
        System.out.printf("📊 Totale film trovati → Generi: %d | Tag: %d | Attori: %d | Registi: %d | Anni: %d ",
                genreCount[0], tagCount[0], actorCount[0], directorCount[0], yearCount[0]);
        // Avvisa l'inizio della fase di raccomandazione vera e propria
        System.out.println("\n📌 Inizio della fase di costruzione del suggerimento personalizzato basato sui dati raccolti...");
        // Stampa ciascun film candidato e il numero di aspetti che soddisfa
        System.out.println("📋 Film candidati da DBpedia (con aspetti soddisfatti):");
        dbpediaCandidates.forEach(c -> {
            // Pulisce il nome del film
            String clean = c.replaceAll("@en$", "")
                    .replaceAll("^\"+|\"+$", "")
                    .replaceAll("\\(.*?\\)", "")
                    .trim();
            // Recupera quanti aspetti soddisfa
            int aspects = filmAspectMap.getOrDefault(clean, Collections.emptySet()).size();
            System.out.println("   → " + clean + " (aspetti: " + aspects + ")");
        });
        // Pulisce e normalizza i titoli dei candidati per la selezione finale
        List<String> candidates = dbpediaCandidates.stream()
                .map(s -> s.replaceAll("@en$", "").replaceAll("^\"+|\"+$", "").trim())
                .collect(Collectors.toList());
        // Applica la strategia softmax per selezionare il film consigliato
        String recommended = recommendSoftmax(candidates, filmAspectMap);
        // Calcola e stampa l'Entity Error per il film suggerito
        double entityError = computeEntityError(recommended, filmAspectMap);
        System.out.printf("📐 Entity Error del film consigliato: %.2f%n", entityError);
    }

    // Metodo per caricare i dati dei film dal file MovieLens.
    private void loadMovieLensData() {
        try (BufferedReader br = new BufferedReader(new FileReader(MOVIELENS_DATASET))) { // Apre il file CSV dei film usando un BufferedReader.
            br.readLine(); // Salta la prima riga (intestazione del CSV).
            String line; // Variabile per contenere ogni riga del file.
            while ((line = br.readLine()) != null) { // Legge riga per riga finché non finisce il file.
                String[] parts = line.split(",", 3); // Divide la riga in massimo 3 parti: movieId, titolo, generi (quest’ultimo può contenere virgole).
                if (parts.length >= 3) { // Verifica che ci siano almeno 3 colonne nella riga.
                    int movieId = Integer.parseInt(parts[0]); // Converte il primo campo (ID film) in intero.
                    String title = parts[1]; // Estrae il titolo del film.
                    String genres = parts[2]; // Estrae i generi del film (separati da '|').
                    movieIdToTitle.put(movieId, title); // Salva l'associazione ID → titolo nella mappa.
                    movieGenres.put(movieId, genres); // Salva l'associazione ID → generi nella mappa.
                }
            }
        } catch (IOException e) { // In caso di errore nella lettura del file...
            System.err.println("❌ Errore: " + e.getMessage());
        }
    }


    // Metodo per caricare i tag dei film dal dataset MovieLens.
    private void loadTagsData() {
        try (BufferedReader br = new BufferedReader(new FileReader(TAGS_DATASET))) { // Apre il file CSV dei tag usando un BufferedReader.
            br.readLine(); // Salta la prima riga (intestazione del CSV).
            String line; // Variabile per contenere ogni riga del file.
            while ((line = br.readLine()) != null) { // Legge riga per riga finché non finisce il file.
                String[] parts = line.split(","); // Divide la riga in colonne separate da virgole.
                if (parts.length >= 3) { // Verifica che ci siano almeno 3 colonne (userId, movieId, tag).
                    int movieId = Integer.parseInt(parts[1]); // Estrae e converte in intero il secondo campo: l’ID del film.
                    String tag = parts[2]; // Estrae il terzo campo: il tag associato al film.
                    movieTags.computeIfAbsent(movieId, _ -> new HashSet<>()).add(tag); // Inserisce il tag nella mappa movieTags (usa una HashSet per evitare duplicati); se non esiste ancora una entry per quel movieId, la crea.
                }
            }
        } catch (IOException e) { // Se si verifica un errore nella lettura del file...
            System.err.println("❌ Errore: " + e.getMessage());
        }
    }

    // Metodo per caricare i rating degli utenti dal file MovieLens.
    private void loadRatingsData() {
        Map<Integer, List<Double>> ratingAccumulator = new HashMap<>(); // Mappa temporanea per accumulare tutti i rating di ciascun film.
        try (BufferedReader br = new BufferedReader(new FileReader(RATINGS_DATASET))) { // Apre il file CSV dei rating.
            br.readLine(); // Salta l’intestazione del file.
            String line; // Variabile per contenere ogni riga del file.
            while ((line = br.readLine()) != null) { // Legge ogni riga finché non finisce il file.
                String[] parts = line.split(","); // Divide la riga nei suoi campi (userId, movieId, rating, timestamp).
                if (parts.length >= 4) { // Verifica che ci siano almeno 4 campi.
                    int userId = Integer.parseInt(parts[0]); // Estrae e converte l'ID dell’utente.
                    int movieId = Integer.parseInt(parts[1]); // Estrae e converte l'ID del film.
                    double rating = Double.parseDouble(parts[2]); // Estrae e converte il rating assegnato.
                    ratingAccumulator.computeIfAbsent(movieId, _ -> new ArrayList<>()).add(rating); // Aggiunge il rating alla lista associata a quel film nella mappa.
                    if (userId == TARGET_USER_ID && rating >= 3.0) { // Se il rating è dell’utente target ed è positivo (≥3)...
                        likedMovieIds.add(movieId); // ...salva l’ID del film tra quelli apprezzati dall’utente.
                    }
                }
            }
        } catch (IOException e) { // In caso di errore di I/O...
            System.err.println("❌ Errore: " + e.getMessage());
        }
    }

    // Metodo che restituisce i top N generi più frequenti tra tutti i film.
    private Set<String> extractTopGenres() {
        int topN=3;
        return extractAllGenres().entrySet().stream() // Ottiene tutte le coppie (genere, conteggio) e le trasforma in uno stream.
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed()) // Ordina i generi in base alla frequenza in ordine decrescente.
                .limit(topN) // Prende solo i primi N generi più frequenti.
                .map(Map.Entry::getKey) // Estrae solo il nome del genere (la chiave della entry).
                .collect(Collectors.toSet()); // Colleziona i generi selezionati in un Set ed è il risultato finale del metodo.
    }

    // Metodo che estrae tutti i generi dei film apprezzati, contando la frequenza di ciascuno.
    private Map<String, Integer> extractAllGenres() {
        Map<String, Integer> freq = new HashMap<>(); // Crea una mappa per memorizzare la frequenza di ogni genere.
        for (int id : likedMovieIds) { // Itera su tutti gli ID dei film apprezzati dall’utente target.
            String[] genres = movieGenres.getOrDefault(id, "").split("\\|"); // Ottiene i generi associati al film e li divide in base al separatore "|".
            for (String g : genres) { // Per ogni genere trovato...
                freq.put(g, freq.getOrDefault(g, 0) + 1); // Incrementa il contatore del genere nella mappa `freq`.
            }
        }
        return freq; // Restituisce la mappa contenente tutti i generi con le rispettive frequenze.
    }

    // Metodo che restituisce i top N tag più frequenti tra tutti i film.
    private Set<String> extractTopTags() {
        int topN=3;
        return extractAllTags().entrySet().stream() // Ottiene tutte le coppie (tag, conteggio) e le trasforma in uno stream.
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed()) // Ordina i tag per frequenza decrescente.
                .limit(topN) // Prende solo i primi N tag più ricorrenti.
                .map(Map.Entry::getKey) // Estrae solo il nome del tag da ciascuna entry.
                .collect(Collectors.toSet()); // Colleziona i tag in un Set e lo restituisce come risultato del metodo.
    }

    // Metodo che estrae tutti i tag associati ai film apprezzati e ne calcola la frequenza.
    private Map<String, Integer> extractAllTags() {
        Map<String, Integer> freq = new HashMap<>(); // Crea una mappa per memorizzare la frequenza di ogni tag.
        for (int id : likedMovieIds) { // Itera su tutti gli ID dei film apprezzati dall’utente.
            for (String tag : movieTags.getOrDefault(id, Collections.emptySet())) { // Per ogni tag associato al film (o insieme vuoto se assente)...
                tag = tag.replaceAll(".*\"(.*)\".*", "$1") // Se il tag contiene virgolette, estrae solo il contenuto interno.
                        .trim() // Rimuove spazi bianchi iniziali/finali.
                        .replaceAll("\\s*-.*", ""); // Rimuove qualsiasi parte dopo un trattino (es. "action - movie" → "action").
                freq.put(tag, freq.getOrDefault(tag, 0) + 1); // Incrementa il contatore del tag nella mappa `freq`.
            }
        }
        return freq; // Restituisce la mappa contenente tutti i tag e la loro frequenza.
    }

    // Metodo che ripulisce le chiavi di una mappa di frequenze testuali (es. da DBpedia).
    private Map<String, Integer> cleanFrequencies(Map<String, Integer> rawMap) {
        Map<String, Integer> cleaned = new HashMap<>(); // Crea una nuova mappa per contenere le chiavi ripulite con le frequenze aggregate.
        for (var entry : rawMap.entrySet()) { // Itera su ogni coppia (chiave, valore) nella mappa originale.
            String cleanedKey = entry.getKey() // Applica una serie di pulizie alla chiave testuale:
                    .replaceAll(".*\\)", "")         // Rimuove tutto ciò che precede una parentesi chiusa inclusa (es. alias o disambiguazioni).
                    .replaceAll("\"", "")            // Rimuove eventuali virgolette doppie.
                    .replaceAll("@en", "")           // Rimuove la marcatura linguistica "@en".
                    .replaceAll("\\s*-.*", "")       // Rimuove trattini e ciò che segue (es. "director - film" → "director").
                    .replaceAll("^,+", "")           // Rimuove virgole iniziali.
                    .trim();                                          // Elimina spazi bianchi iniziali e finali.
            cleaned.put(cleanedKey, cleaned.getOrDefault(cleanedKey, 0) + entry.getValue()); // Inserisce la chiave pulita nella nuova mappa, sommando eventuali duplicati.
        }
        return cleaned; // Restituisce la mappa con le chiavi ripulite e frequenze aggregate.
    }

    // Metodo che stampa a console le frequenze di una mappa (es. generi o tag).
    private void printFrequencies(Map<String, Integer> freqMap) {
        freqMap.entrySet().stream() // Converte la mappa in uno stream di entry (coppie chiave-valore).
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed()) // Ordina le entry per valore decrescente (frequenze più alte prima).
                .forEach(entry -> { // Per ogni entry ordinata...
                    String key = entry.getKey().replaceAll("^,+", "").trim(); // Pulisce la chiave: rimuove eventuali virgole iniziali e spazi.
                    System.out.println("  " + key + " → " + entry.getValue()); // Stampa la chiave (es. genere/tag) e la sua frequenza.
                });
    }

    // Metodo che estrae le top N entità (es. registi o attori) più ricorrenti tra i film apprezzati usando una sola query SPARQL per ogni entità
    private Set<String> extractTopEntities(String property, boolean verbose) {
        // Numero massimo di entità da restituire
        int topN = 3;
        // Mappa per conteggiare le occorrenze di ogni entità (es. regista/attore)
        Map<String, Integer> freq = new HashMap<>();
        // Costruisce una lista dei titoli puliti dei film apprezzati
        List<String> cleanedTitles = likedMovieIds.stream()
                .map(movieIdToTitle::get)
                .filter(Objects::nonNull)
                .map(title -> title
                        .replaceAll("\"", "")                   // Rimuove virgolette
                        .replaceAll("\\(\\d{4}\\)", "")         // Rimuove l'anno tra parentesi
                        .replaceAll("\\s*-.*", "")              // Rimuove eventuali sottotitoli dopo trattino
                        .trim())                                // Elimina spazi iniziali/finali
                .distinct()                                     // Rimuove eventuali duplicati
                .toList();
        // Itera sui titoli e per ognuno esegue una query SPARQL aggregata
        for (String title : cleanedTitles) {
            // Ottiene tutti i valori semantici tramite una sola query SPARQL
            Map<String, List<String>> result = queryDbpediaProperties(title, verbose);
            // Estrae solo la lista relativa alla proprietà richiesta ("starring" o "director")
            List<String> values = result.getOrDefault(
                    property.equals("starring") ? "actor" : "director",
                    Collections.emptyList()
            );
            // Conta la frequenza per ogni valore trovato
            for (String value : values) {
                String cleanValue = value.replaceAll("@en$", "").trim();
                freq.put(cleanValue, freq.getOrDefault(cleanValue, 0) + 1);
            }
        }
        // Ordina le entità per frequenza decrescente, limita a topN e restituisce i nomi
        return freq.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .limit(topN)
                .map(Map.Entry::getKey)
                .collect(Collectors.toSet());
    }

    /**
     * Metodo che esegue una singola query SPARQL con OPTIONAL per ottenere attori e registi associati a un film.
     *
     * @param title   Titolo del film (usato per cercare il match su DBpedia)
     * @param verbose Se true, stampa i dettagli della query e dei risultati
     * @return Mappa con due chiavi ("actor" e "director") e i relativi valori trovati
     */
    private Map<String, List<String>> queryDbpediaProperties(String title, boolean verbose) {
        if (verbose)
            System.out.println("🔍 DBpedia → Film: " + title);
        if (dbpediaCache.containsKey(title))
            return dbpediaCache.get(title);
        // Mappa dei risultati finali da restituire
        Map<String, List<String>> resultMap = new HashMap<>();
        // Pulisce il titolo da alias, parentesi, virgolette, sottotitoli e spazi inutili
        String cleanTitle = title
                .replaceAll("\\(a\\.k\\.a\\..*?\\)", "")
                .replaceAll("\\(.*?\\)", "")
                .replaceAll("-.*", "")
                .replaceAll("\"", "")
                .trim();
        // Costruisce la query SPARQL con OPTIONAL e uso di IN per maggiore copertura su Film/Movie
        String query = String.format("""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?actorLabel ?directorLabel WHERE {
          ?film a ?type ;
                rdfs:label ?label .
          FILTER (?type IN (dbo:Film, dbo:Movie))
          FILTER (lang(?label) = 'en')
          FILTER CONTAINS(LCASE(?label), LCASE("%s"))

          OPTIONAL {
            ?film dbo:starring ?actor .
            ?actor rdfs:label ?actorLabel .
            FILTER (lang(?actorLabel) = 'en')
          }

          OPTIONAL {
            ?film dbo:director ?director .
            ?director rdfs:label ?directorLabel .
            FILTER (lang(?directorLabel) = 'en')
          }
        }
        LIMIT 5
        """, cleanTitle);
        // Esegue la query SPARQL sull'endpoint di DBpedia
        try (QueryExecution qexec = QueryExecution.service(DBPEDIA_SPARQL_ENDPOINT)
                .query(QueryFactory.create(query)).build()) {
            ResultSet rs = qexec.execSelect(); // Ottiene il risultato
            while (rs.hasNext()) {
                QuerySolution sol = rs.next();
                // Se viene trovato un attore, lo aggiunge alla mappa
                if (sol.contains("actorLabel")) {
                    String actor = sol.get("actorLabel").toString().replaceAll("@en$", "").trim();
                    resultMap.computeIfAbsent("actor", _ -> new ArrayList<>()).add(actor);
                    if (verbose) System.out.println("🎭 Attore trovato → " + actor);
                }
                // Se viene trovato un regista, lo aggiunge alla mappa
                if (sol.contains("directorLabel")) {
                    String director = sol.get("directorLabel").toString().replaceAll("@en$", "").trim();
                    resultMap.computeIfAbsent("director", _ -> new ArrayList<>()).add(director);
                    if (verbose) System.out.println("🎬 Regista trovato → " + director);
                }
            }
        } catch (Exception e) {
            System.err.println("❌ Errore query SPARQL per: " + title);
            System.err.println("❌ Errore: " + e.getMessage());
        }
        dbpediaCache.put(title, resultMap);
        return resultMap;
    }

    // Metodo che esegue una query SPARQL unificata per trovare film che corrispondono
// ad almeno uno degli aspetti preferiti (generi, tag, attori, registi, decenni).
    private Set<String> unifiedSparqlSearch(Set<String> genres, Set<String> tags, Set<String> actors, Set<String> directors, Set<Integer> preferredYears) {
        Set<String> titles = new HashSet<>();
        // Filtri dinamici
        // Se l'insieme degli attori è vuoto, il filtro è una stringa vuota, altrimenti costruisce un filtro SPARQL
        String actorFilters = actors.isEmpty() ? "" :
                // Per ogni attore, crea una stringa SPARQL che verifica se l'etichetta dell'attore contiene il nome (case insensitive)
                actors.stream()
                        .map(a -> String.format("CONTAINS(LCASE(STR(?actorLabel)), LCASE(\"%s\"))", a))
                        // Unisce tutte le condizioni con "||" e le incapsula dentro FILTER (...)
                        .collect(Collectors.joining(" || ", "FILTER (", ")"));
        // Stessa logica del filtro attori, applicata ai registi
                String directorFilters = directors.isEmpty() ? "" :
                        directors.stream()
                                .map(d -> String.format("CONTAINS(LCASE(STR(?directorLabel)), LCASE(\"%s\"))", d))
                                .collect(Collectors.joining(" || ", "FILTER (", ")"));

        // Costruisce il filtro SPARQL per i generi (es. Action, Comedy) usando la stessa logica dei filtri precedenti
                String genreFilters = genres.isEmpty() ? "" :
                        genres.stream()
                                .map(g -> String.format("CONTAINS(LCASE(STR(?genreLabel)), LCASE(\"%s\"))", g))
                                .collect(Collectors.joining(" || ", "FILTER (", ")"));

        // Costruisce il filtro SPARQL per i tag (match sull'etichetta del film)
                String tagFilters = tags.isEmpty() ? "" :
                        tags.stream()
                                .map(t -> String.format("CONTAINS(LCASE(STR(?filmTitle)), LCASE(\"%s\"))", t))
                                .collect(Collectors.joining(" || ", "FILTER (", ")"));

        // Costruisce il filtro SPARQL sugli anni di uscita: considera gli anni preferiti ±5 anni
                String yearFilter = preferredYears.isEmpty() ? "" :
                        preferredYears.stream()
                                // Per ogni anno, crea una condizione sull'intervallo di +/- 5 anni
                                .map(y -> String.format("(YEAR(?releaseDate) >= %d && YEAR(?releaseDate) <= %d)", y - 5, y + 5))
                                // Unisce le condizioni con "||" per creare un filtro su più intervalli
                                .collect(Collectors.joining(" || "));
        // Query SPARQL con blocchi OPTIONAL e filtri semplificati
        String query = String.format("""
            PREFIX dbo: <http://dbpedia.org/ontology/>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            SELECT DISTINCT ?filmTitle WHERE {
              ?film a ?type ;
                    rdfs:label ?filmTitle .
              FILTER (?type IN (dbo:Film, dbo:Movie))
              FILTER (lang(?filmTitle) = 'en')
              OPTIONAL {
                ?film dbo:starring ?actor .
                ?actor rdfs:label ?actorLabel .
                FILTER (lang(?actorLabel) = 'en')
                %s
              }
              OPTIONAL {
                ?film dbo:director ?director .
                ?director rdfs:label ?directorLabel .
                FILTER (lang(?directorLabel) = 'en')
                %s
              }
              OPTIONAL {
                ?film dbo:genre ?genre .
                ?genre rdfs:label ?genreLabel .
                FILTER (lang(?genreLabel) = 'en')
                %s
              }
              OPTIONAL {
                ?film dbo:releaseDate ?releaseDate .
                %s
              }
              %s
            }
            LIMIT 100
            """,
                actorFilters,
                directorFilters,
                genreFilters,
                yearFilter.isBlank() ? "" : "FILTER (" + yearFilter + ")",
                tagFilters
        );
        // Esecuzione della query SPARQL (usa POST con build())
        try (QueryExecution qexec = QueryExecution.service(DBPEDIA_SPARQL_ENDPOINT)
                .query(QueryFactory.create(query))
                .build()) {
            ResultSet rs = qexec.execSelect();
            while (rs.hasNext()) {
                QuerySolution sol = rs.next();
                titles.add(sol.get("filmTitle").toString());
            }
        } catch (Exception e) {
            System.err.println("❌ Errore nella query SPARQL unificata:");
            System.err.println("❌ Errore: " + e.getMessage());
        }
        return titles;
    }

    /**
     * Applica la strategia di selezione Softmax per raccomandare un film tra i candidati.
     * Calcola per ciascun film una probabilità basata sul numero di aspetti soddisfatti
     * (generi, tag, attori, registi, anni). Elimina a priori i film senza aspetti,
     * calcola le probabilità normalizzate tramite softmax e seleziona stocasticamente un film
     * in base a tali probabilità. Stampa inoltre i dettagli di ogni film candidato,
     * inclusi aspetti soddisfatti e probabilità.
     *
     * @param movies Lista di film candidati ottenuti da DBpedia
     * @param filmAspectMap Mappa che associa a ciascun film gli aspetti preferiti soddisfatti
     * @return Il titolo del film raccomandato
     */
    private String recommendSoftmax(List<String> movies, Map<String, Set<String>> filmAspectMap) {
        //Filtra i film mantenendo solo quelli che hanno almeno un aspetto associato
        movies = movies.stream()
                .filter(m -> {
                    String clean = m.replaceAll("@en$", "").replaceAll("^\"+|\"+$", "").trim();
                    return filmAspectMap.containsKey(clean) && !filmAspectMap.get(clean).isEmpty();
                })
                .collect(Collectors.toList());
        //Se non ci sono film validi, restituisce un messaggio
        if (movies.isEmpty()) return "Nessun film trovato";
        // Ricostruisce una mappa locale con solo i film validi e i relativi aspetti
        Map<String, Set<String>> aspectMap = new HashMap<>();
        for (String movie : movies) {
            String cleanTitle = movie.replaceAll("@en$", "").replaceAll("^\"+|\"+$", "").trim();
            aspectMap.put(cleanTitle, filmAspectMap.getOrDefault(cleanTitle, Collections.emptySet()));
        }
        //Parametro alpha per regolare la "temperatura" della softmax (quanto è marcata la differenza tra probabilità)
        double alpha = 1.0;
        //Calcola i punteggi grezzi per ogni film (basati sul numero di aspetti soddisfatti)
        Map<String, Double> scores = new HashMap<>();
        for (String movie : movies) {
            String cleanTitle = movie.replaceAll("@en$", "").trim();
            double rating = aspectMap.getOrDefault(cleanTitle, Collections.emptySet()).size();
            scores.put(cleanTitle, rating);
        }
        // Calcola il denominatore della formula softmax (somma degli esponenziali pesati)
        double denominator = scores.values().stream()
                .mapToDouble(score -> Math.exp(alpha * score))
                .sum();
        //Calcola la probabilità per ciascun film usando la formula softmax
        Map<String, Double> probabilities = new HashMap<>();
        for (var entry : scores.entrySet()) {
            double probability = Math.exp(alpha * entry.getValue()) / denominator;
            probabilities.put(entry.getKey(), probability);
        }
        //Stampa ordinata delle probabilità con il dettaglio degli aspetti soddisfatti
        System.out.println("📋 Probabilità di raccomandazione:");
        probabilities.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .forEach(e -> {
                    Set<String> aspects = aspectMap.getOrDefault(e.getKey(), Collections.emptySet());
                    System.out.printf("   🎞️ \"%s\" → %.4f (aspetti: %d → %s)%n",
                            e.getKey(), e.getValue(), aspects.size(), aspects);
                });
        // Esegue una selezione stocastica (probabilistica) sulla base delle probabilità
        double rand = random.nextDouble();
        double cumulative = 0.0;
        for (var entry : probabilities.entrySet()) {
            cumulative += entry.getValue();
            if (rand <= cumulative) {
                System.out.println("👉 Film scelto (softmax): " + entry.getKey());
                return entry.getKey(); // Restituisce il film selezionato
            }
        }
        // Fallback in caso di problemi numerici: seleziona il film con la probabilità più alta
        return probabilities.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse("Nessun film selezionato");
    }

    // Calcola l'errore di entità per il film raccomandato basato su rating o aspetti semantici
    private double computeEntityError(String recommendedTitle, Map<String, Set<String>> filmAspectMap) {
        // Rimuove etichette linguistiche e virgolette dal titolo suggerito
        recommendedTitle = recommendedTitle.replaceAll("@en$", "").replaceAll("^\"+|\"+$", "").trim();
        // Cerca l'ID corrispondente al titolo nel dataset MovieLens
        Integer recommendedId = null;
        for (Map.Entry<Integer, String> entry : movieIdToTitle.entrySet()) {
            String movieTitle = entry.getValue().replaceAll("\\(.*?\\)", "").trim();
            if (movieTitle.equalsIgnoreCase(recommendedTitle)) {
                recommendedId = entry.getKey();
                break;
            }
        }
        System.out.println("🆔 ID trovato: " + (recommendedId != null ? recommendedId : "❌ Nessun ID trovato"));
        // Caso A: il film ha un ID ed è stato valutato positivamente dall'utente
        if (recommendedId != null && likedMovieIds.contains(recommendedId)) {
            try (BufferedReader br = new BufferedReader(new FileReader(RATINGS_DATASET))) {
                br.readLine(); // Salta intestazione CSV
                String line;
                while ((line = br.readLine()) != null) {
                    String[] parts = line.split(",");
                    if (parts.length >= 3) {
                        int userId = Integer.parseInt(parts[0]);
                        int movieId = Integer.parseInt(parts[1]);
                        double rating = Double.parseDouble(parts[2]);
                        // Se il film è stato valutato dall’utente target
                        if (userId == TARGET_USER_ID && movieId == recommendedId) {
                            return 1.0 - (rating / 5.0); // Calcola errore inversamente proporzionale al rating
                        }
                    }
                }
            } catch (IOException e) {
                System.err.println("❌ Errore: " + e.getMessage()); // Gestione errore lettura file
            }
        }
        // Caso B: confronta con aspetti semantici (se rating non disponibile)
        int matchCount = 0; // Numero di aspetti soddisfatti
        int totalChecks = 0; // Numero totale di aspetti verificati
        // Se il film è nel dataset, confronta generi e tag
        if (recommendedId != null) {
            // 1. Generi
            Set<String> recommendedGenres = new HashSet<>(Arrays.asList(movieGenres.getOrDefault(recommendedId, "").split("\\|")));
            Set<String> topGenres = extractTopGenres();
            for (String g : recommendedGenres) {
                if (!g.isBlank()) {
                    totalChecks++;
                    if (topGenres.contains(g)) matchCount++;
                }
            }
            // 2. Tag
            Set<String> recommendedTags = movieTags.getOrDefault(recommendedId, Collections.emptySet());
            Set<String> topTags = extractTopTags();
            for (String t : recommendedTags) {
                if (!t.isBlank()) {
                    totalChecks++;
                    if (topTags.contains(t)) matchCount++;
                }
            }
        }
        // Esegue una sola query SPARQL per ottenere attori e registi
        Map<String, List<String>> propertyMap = queryDbpediaProperties(recommendedTitle, false);
        // 3. Attori
        Set<String> topActors = extractTopEntities("starring", false);
        List<String> actors = propertyMap.getOrDefault("actor", Collections.emptyList());
        for (String actor : actors) {
            totalChecks++;
            if (topActors.contains(actor)) matchCount++;
        }
        // 4. Registi
        Set<String> topDirectors = extractTopEntities("director", false);
        List<String> directors = propertyMap.getOrDefault("director", Collections.emptyList());
        for (String director : directors) {
            totalChecks++;
            if (topDirectors.contains(director)) matchCount++;
        }
        // 5. Anno: controlla se l'anno del titolo suggerito è vicino a uno degli anni preferiti
        Set<Integer> preferredYears = likedMovieIds.stream()
                .map(movieIdToTitle::get)
                .filter(Objects::nonNull)
                .map(title -> {
                    String match = title.replaceAll(".*\\((\\d{4})\\).*", "$1");
                    try {
                        return Integer.parseInt(match);
                    } catch (Exception e) {
                        return null;
                    }
                })
                .filter(Objects::nonNull)
                .collect(Collectors.toSet());
        if (recommendedTitle.matches(".*\\((\\d{4})\\).*")) {
            String match = recommendedTitle.replaceAll(".*\\((\\d{4})\\).*", "$1");
            try {
                int year = Integer.parseInt(match);
                totalChecks++;
                if (preferredYears.stream().anyMatch(y -> Math.abs(y - year) <= 2)) matchCount++;
            } catch (Exception ignored) {}
        }
        // Fallback: se non è stato trovato l’ID, usa filmAspectMap per il conteggio
        if (recommendedId == null && filmAspectMap.containsKey(recommendedTitle)) {
            matchCount = filmAspectMap.getOrDefault(recommendedTitle, Collections.emptySet()).size();
            totalChecks = 5; // Considera massimo 5 aspetti
        }
        System.out.println("✅ MatchCount: " + matchCount + " / TotalChecks: " + totalChecks);
        if (totalChecks == 0) return 1.0; // Evita divisione per zero
        return 1.0 - ((double) matchCount / totalChecks); // Calcola entity error
    }
}