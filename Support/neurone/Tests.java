package neurone;

import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.IOException;

public class Tests {

    public static void main(String[] args) {
        // Paramètres configurables
        int nombreExecutions = 50; // Tu peux passer ça à 50
        String fichierSortie = "resultats_executions.txt";

        System.out.println("Lancement de " + nombreExecutions + " exécutions de testNeurone...");

        // Le paramètre "true" dans FileWriter permet d'écrire à la suite du fichier (append) sans l'écraser
        try (FileWriter writer = new FileWriter(fichierSortie, false)) {

            for (int i = 1; i <= nombreExecutions; i++) {

                // On prépare l'exécution de la commande "java testNeurone"
                // On récupère le chemin exact où IntelliJ a rangé les fichiers compilés
                String classpath = System.getProperty("java.class.path");

// On lance Java en lui donnant ce chemin avec l'option -cp (classpath)
                ProcessBuilder pb = new ProcessBuilder("java", "-cp", classpath, "testNeurone");
                Process process = pb.start();

                // On lit la sortie standard (ce qui s'affiche normalement dans la console)
                BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                String ligne;

                String derniereLigneIteration = "";
                String synapses = "";
                String biais = "";

                // Parcours de chaque ligne affichée par testNeurone
                while ((ligne = reader.readLine()) != null) {
                    if (ligne.startsWith("Itération")) {
                        derniereLigneIteration = ligne;
                    } else if (ligne.startsWith("Synapses :")) {
                        // On récupère juste les valeurs après "Synapses :"
                        synapses = ligne.substring("Synapses :".length()).trim();
                    } else if (ligne.startsWith("Biais :")) {
                        // On récupère juste la valeur après "Biais :"
                        biais = ligne.substring("Biais :".length()).trim();
                    }
                }

                // On attend que l'exécution se termine
                int exitCode = process.waitFor();

                if (exitCode == 0) {
                    // Nettoyage pour récupérer uniquement le numéro de l'itération
                    // La ligne ressemble à "Itération 134, mse:  0.000999"
                    String nbIterations = "Inconnu";
                    if (!derniereLigneIteration.isEmpty()) {
                        String[] parties = derniereLigneIteration.split(",");
                        if (parties.length > 0) {
                            nbIterations = parties[0].replace("Itération", "").trim();
                        }
                    }

                    // Écriture dans le fichier
                    writer.write("Exécution " + i + "\n");
                    writer.write("Itération finale : " + nbIterations + "\n");
                    writer.write("Synapses       : " + synapses + "\n");
                    writer.write("Biais          : " + biais + "\n");
                    writer.write("----------------------------------------\n");

                    System.out.println("Exécution " + i + "/" + nombreExecutions + " terminée.");
                } else {
                    System.err.println("L'exécution " + i + " a échoué (Code d'erreur: " + exitCode + ").");
                }
            }

            System.out.println("\nToutes les exécutions sont terminées !");
            System.out.println("Les résultats ont été ajoutés dans le fichier : " + fichierSortie);

        } catch (IOException | InterruptedException e) {
            System.err.println("Une erreur est survenue lors de l'exécution ou de l'écriture.");
            e.printStackTrace();
        }
    }
}
