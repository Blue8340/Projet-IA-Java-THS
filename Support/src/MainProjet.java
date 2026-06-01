package src;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MainProjet {

    public static void main(String[] args) {
        String dossierTrain = "dataset_groupe_8/train/";
        String dossierTest = "dataset_groupe_8/test/";

        // ========================================================
        // 1. CHARGEMENT ET PRÉPARATION (Avec le Image.java d'origine)
        // ========================================================
        System.out.println("Chargement et labellisation des données d'entraînement...");
        List<String> cheminsTrain = Image.listeFichiers(dossierTrain);
        List<Image> imagesTrain = new ArrayList<>();

        int tailleReference = -1;

        for (String chemin : cheminsTrain) {
            int label = -1;
            // Respect STRICT de la consigne "actif = chat" (donc label = 1)
            if (chemin.toLowerCase().contains("cat")) {
                label = 1; 
            } else if (chemin.toLowerCase().contains("dog")) {
                label = 0; // "inactif = pas chat"
            }

            if (label != -1) {
                Image img = new Image(chemin, label, true); // true = Niveaux de gris
                
                if (tailleReference == -1) tailleReference = img.taille();
                
                // Sécurité dimensionnelle
                if (img.taille() == tailleReference) {
                    imagesTrain.add(img);
                }
            }
        }

        // Mélange robuste via la bibliothèque standard Java
        Collections.shuffle(imagesTrain);

        // ========================================================
        // 2. CONNEXION ET NORMALISATION (Ce que les profs attendent)
        // ========================================================
        int nbImages = imagesTrain.size();
        float[][] entreesTrain = new float[nbImages][tailleReference];
        float[] resultatsTrain = new float[nbImages];

        for (int i = 0; i < nbImages; i++) {
            Image img = imagesTrain.get(i);
            resultatsTrain[i] = (float) img.label();
            
            // On fait le pont entre le int[] de Image et le float[] du Neurone
            int[] pixelsBruts = img.donnees();
            for (int j = 0; j < tailleReference; j++) {
                entreesTrain[i][j] = pixelsBruts[j] / 255.0f; // Normalisation
            }
        }

        // ========================================================
        // 3. APPRENTISSAGE
        // ========================================================
        System.out.println("Début de l'entraînement...");
        //iNeurone neurone = new NeuroneSigmoide(tailleReference);
        //iNeurone neurone = new NeuroneHeavyside(tailleReference);
        iNeurone neurone = new NeuroneReLU(tailleReference);
        Neurone.fixeCoefApprentissage(0.001f);
        neurone.apprentissage(entreesTrain, resultatsTrain, 0.05f);

        // ========================================================
        // 4. TEST SUR DONNÉES INCONNUES
        // ========================================================
        System.out.println("\nDébut du Test...");
        List<String> cheminsTest = Image.listeFichiers(dossierTest);
        
        int correct = 0;
        int totalTest = 0;

        for (String chemin : cheminsTest) {
            int vraiLabel = -1;
            if (chemin.toLowerCase().contains("cat")) vraiLabel = 1;
            else if (chemin.toLowerCase().contains("dog")) vraiLabel = 0;

            if (vraiLabel != -1) {
                Image imgTest = new Image(chemin, vraiLabel, true);
                if (imgTest.taille() != tailleReference) continue;

                // Normalisation à la volée pour le test
                float[] entreeTest = new float[tailleReference];
                int[] pixelsTest = imgTest.donnees();
                for (int j = 0; j < tailleReference; j++) {
                    entreeTest[j] = pixelsTest[j] / 255.0f;
                }

                // Prédiction
                neurone.metAJour(entreeTest);
                int prediction = (neurone.sortie() >= 0.5f) ? 1 : 0;
                
                if (prediction == vraiLabel) correct++;
                totalTest++;
            }
        }

        float taux = ((float) correct / totalTest) * 100f;
        System.out.printf("Résultat : %d/%d (%.2f%% de précision)\n", correct, totalTest, taux);
    }
}