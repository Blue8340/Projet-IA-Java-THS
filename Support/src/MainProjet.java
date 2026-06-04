package src;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class MainProjet {

    public static void main(String[] args) {
        String dossierTrain = "dataset_groupe_8/train/";
        String dossierTest = "dataset_groupe_8/test/";

        final int LABEL_CHAT = 0;
        final int LABEL_CHIEN = 1;
        final int LABEL_WILD = 2;

        // ========================================================
        // 1. CHARGEMENT + FILTRE DE DIMENSION
        // ========================================================
        System.out.println("Chargement des données d'entraînement...");
        List<String> cheminsTrain = Image.listeFichiers(dossierTrain);
        List<Image> imagesTrain = new ArrayList<>();
        int tailleBrute = -1;

        if (cheminsTrain != null) {
            for (String chemin : cheminsTrain) {
                int label = -1;
                String c = chemin.toLowerCase();
                if (c.contains("/dog/") || c.contains("\\dog\\")) label = LABEL_CHIEN;
                else if (c.contains("/cat/") || c.contains("\\cat\\")) label = LABEL_CHAT;
                else if (c.contains("/wild/") || c.contains("\\wild\\")) label = LABEL_WILD;
                if (label != -1) {
                    Image img = new Image(chemin, label, false);
                    if (tailleBrute == -1) tailleBrute = img.taille();
                    if (img.taille() == tailleBrute) imagesTrain.add(img);
                }
            }
            Collections.shuffle(imagesTrain);
        }

        // ========================================================
        // 2. HOG + AUGMENTATION (les 5 pistes). Le TEST n'est PAS augmenté.
        // ========================================================
        int nbBase = imagesTrain.size();
        List<float[]> listeEntrees = new ArrayList<>();
        List<Integer> listeLabels = new ArrayList<>();

        System.out.println("Calcul des HOG + augmentations (cela peut etre long)...");
        for (int i = 0; i < nbBase; i++) {
            Image img = imagesTrain.get(i);
            int lab = img.label();
            for (float[] v : augmenter(img)) {   // toutes les versions de cette image
                listeEntrees.add(v);
                listeLabels.add(lab);            // même label pour toutes
            }
        }

        int nbAug = listeEntrees.size();
        int tailleReference = listeEntrees.get(0).length;
        System.out.println("Entrées de base : " + nbBase + "  ->  après augmentation : " + nbAug
                + "  (x" + (nbAug / Math.max(1, nbBase)) + ")");
        System.out.println("Taille d'une entrée (HOG) : " + tailleReference);

        // Tableaux + cibles one-vs-all
        float[][] entreesTrain = new float[nbAug][];
        float[] objChien = new float[nbAug], objChat = new float[nbAug], objWild = new float[nbAug];
        for (int k = 0; k < nbAug; k++) {
            entreesTrain[k] = listeEntrees.get(k);
            int lab = listeLabels.get(k);
            objChien[k] = (lab == LABEL_CHIEN) ? 1f : 0f;
            objChat[k] = (lab == LABEL_CHAT) ? 1f : 0f;
            objWild[k] = (lab == LABEL_WILD) ? 1f : 0f;
        }
        listeEntrees = null;
        listeLabels = null; // libère la mémoire

        // Re-mélange (Fisher-Yates parallèle sur les 4 tableaux)
        Random rnd = new Random();
        for (int a = nbAug - 1; a > 0; a--) {
            int b = rnd.nextInt(a + 1);
            float[] te = entreesTrain[a];
            entreesTrain[a] = entreesTrain[b];
            entreesTrain[b] = te;
            float t;
            t = objChien[a];
            objChien[a] = objChien[b];
            objChien[b] = t;
            t = objChat[a];
            objChat[a] = objChat[b];
            objChat[b] = t;
            t = objWild[a];
            objWild[a] = objWild[b];
            objWild[b] = t;
        }

        // ========================================================
        // 3. ENTRAÎNEMENT DES 3 EXPERTS
        // ========================================================
        Neurone.fixeCoefApprentissage(0.0001f);
        // ATTENTION : avec beaucoup d'augmentations, baissez ce nombre (sinon très long).
        final int ITERATIONS = 800;

        System.out.println("\n>>> Expert CHIEN...");
        iNeurone expertChien = new NeuroneSigmoide(tailleReference);
        ((Neurone) expertChien).apprentissage(entreesTrain, objChien, 0f, ITERATIONS);
        System.out.println("\n>>> Expert CHAT...");
        iNeurone expertChat = new NeuroneSigmoide(tailleReference);
        ((Neurone) expertChat).apprentissage(entreesTrain, objChat, 0f, ITERATIONS);
        System.out.println("\n>>> Expert WILD...");
        iNeurone expertWild = new NeuroneSigmoide(tailleReference);
        ((Neurone) expertWild).apprentissage(entreesTrain, objWild, 0f, ITERATIONS);
        System.out.println("\nApprentissage terminé !");

        // ========================================================
        // 4. TESTS (test NON augmenté)
        // ========================================================
        System.out.println("\n=== TRAIN ===");
        testerLeModele(dossierTrain, expertChien, expertChat, expertWild, tailleReference, LABEL_CHAT, LABEL_CHIEN, LABEL_WILD);
        System.out.println("\n=== TEST (données inconnues) ===");
        testerLeModele(dossierTest, expertChien, expertChat, expertWild, tailleReference, LABEL_CHAT, LABEL_CHIEN, LABEL_WILD);
    }

    // ========================================================
    // AUGMENTATION : renvoie toutes les versions HOG d'une image.
    // Commentez les lignes pour tester l'effet de chaque piste séparément.
    // ========================================================
    private static List<float[]> augmenter(Image img) {
        List<float[]> v = new ArrayList<>();
        v.add(img.donneesHOG());                  // original
        v.add(img.donneesHOGMiroir());            // miroir
        // Piste 5 : plusieurs angles de rotation
        v.add(img.donneesHOGRotation(10));
        v.add(img.donneesHOGRotation(-10));
        v.add(img.donneesHOGRotation(20));
        v.add(img.donneesHOGRotation(-20));
        // Piste 1 : translations de quelques pixels
        v.add(img.donneesHOGTranslation(3, 0));
        v.add(img.donneesHOGTranslation(-3, 0));
        v.add(img.donneesHOGTranslation(0, 3));
        v.add(img.donneesHOGTranslation(0, -3));
        // Piste 3 : zoom avant / arrière
        v.add(img.donneesHOGZoom(1.1));
        v.add(img.donneesHOGZoom(0.9));
        // Piste 4 : bruit gaussien léger
        v.add(img.donneesHOGBruit(8.0));
        // Piste 2 : combinaison (miroir + rotation)
        v.add(img.donneesHOGMiroirRotation(10));
        return v;
    }

    // === Remplace la méthode testerLeModele dans MainProjet ===
    private static void testerLeModele(String dossier, iNeurone expertChien, iNeurone expertChat, iNeurone expertWild,
                                       int tailleReference, int LABEL_CHAT, int LABEL_CHIEN, int LABEL_WILD) {
        List<String> cheminsTest = Image.listeFichiers(dossier);
        int correct = 0, totalTest = 0;
        // conf[vrai][predit] : lignes = vraie classe, colonnes = classe prédite
        // index : 0 = CHAT, 1 = CHIEN, 2 = WILD
        int[][] conf = new int[3][3];
        String[] noms = {"CHAT", "CHIEN", "WILD"};

        if (cheminsTest != null) {
            for (String chemin : cheminsTest) {
                int vraiLabel = -1;
                String c = chemin.toLowerCase();
                if (c.contains("/dog/") || c.contains("\\dog\\")) vraiLabel = LABEL_CHIEN;
                else if (c.contains("/cat/") || c.contains("\\cat\\")) vraiLabel = LABEL_CHAT;
                else if (c.contains("/wild/") || c.contains("\\wild\\")) vraiLabel = LABEL_WILD;

                if (vraiLabel != -1) {
                    Image imgTest = new Image(chemin, vraiLabel, false);
                    float[] feats = imgTest.donneesHOG();
                    if (feats.length != tailleReference) continue;

                    expertChien.metAJour(feats);
                    expertChat.metAJour(feats);
                    expertWild.metAJour(feats);
                    float sC = expertChien.sortie(), sCh = expertChat.sortie(), sW = expertWild.sortie();
                    int prediction = LABEL_CHIEN;
                    float maxScore = sC;
                    if (sCh > maxScore) {
                        prediction = LABEL_CHAT;
                        maxScore = sCh;
                    }
                    if (sW > maxScore) {
                        prediction = LABEL_WILD;
                        maxScore = sW;
                    }

                    conf[vraiLabel][prediction]++;
                    if (prediction == vraiLabel) correct++;
                    totalTest++;
                }
            }
        }

        if (totalTest == 0) {
            System.out.println("Aucune image valide.");
            return;
        }

        System.out.printf("RÉSULTAT GLOBAL : %d/%d (%.2f%% de précision)%n", correct, totalTest, 100f * correct / totalTest);

        // Matrice de confusion
        System.out.println("\nMatrice de confusion (lignes = vraie classe, colonnes = prédiction) :");
        System.out.printf("%-8s %8s %8s %8s%n", "", "CHAT", "CHIEN", "WILD");
        for (int v = 0; v < 3; v++)
            System.out.printf("%-8s %8d %8d %8d%n", noms[v], conf[v][0], conf[v][1], conf[v][2]);

        // Précision (predit=c correct / total predit=c) et rappel (vrai=c correct / total vrai=c) par classe
        System.out.println("\nPar classe :");
        for (int c2 = 0; c2 < 3; c2++) {
            int vp = conf[c2][c2];
            int totalPredit = conf[0][c2] + conf[1][c2] + conf[2][c2];
            int totalVrai = conf[c2][0] + conf[c2][1] + conf[c2][2];
            float precision = totalPredit == 0 ? 0 : 100f * vp / totalPredit;
            float rappel = totalVrai == 0 ? 0 : 100f * vp / totalVrai;
            System.out.printf("  %-6s : précision=%.2f%%  rappel=%.2f%%%n", noms[c2], precision, rappel);
        }
        System.out.println("---------------------------------------------------");
    }
}
