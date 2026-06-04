package src;

import java.util.ArrayList;
import java.util.List;

// Classifieur de référence (baseline) : k plus proches voisins sur features HOG.
// Permet de contextualiser la performance du neurone. Aucun apprentissage :
// on compare chaque image de test à toutes les images de train (distance euclidienne).
public class TestKNN {

    static final int K = 5;  // nombre de voisins

    public static void main(String[] args) {
        String dossierTrain = "dataset_groupe_8/train/";
        String dossierTest  = "dataset_groupe_8/test/";

        // 1) Chargement TRAIN (features HOG + labels)
        List<float[]> trainX = new ArrayList<>();
        List<Integer> trainY = new ArrayList<>();
        int taille = -1;
        for (String chemin : Image.listeFichiers(dossierTrain)) {
            int lab = labelOf(chemin);
            if (lab < 0) continue;
            Image img = new Image(chemin, lab, false);
            float[] f = img.donneesHOG();
            if (taille == -1) taille = f.length;
            if (f.length != taille) continue;
            trainX.add(f); trainY.add(lab);
        }
        System.out.println("Train chargé : " + trainX.size() + " images.");

        // 2) TEST + matrice de confusion
        int[][] conf = new int[3][3];
        int correct = 0, total = 0;
        for (String chemin : Image.listeFichiers(dossierTest)) {
            int vrai = labelOf(chemin);
            if (vrai < 0) continue;
            Image img = new Image(chemin, vrai, false);
            float[] x = img.donneesHOG();
            if (x.length != taille) continue;
            int pred = classerKNN(x, trainX, trainY);
            conf[vrai][pred]++;
            if (pred == vrai) correct++;
            total++;
        }

        System.out.printf("%nkNN (k=%d) : %d/%d (%.2f%% de précision)%n", K, correct, total, 100f*correct/total);
        String[] noms = {"CHAT", "CHIEN", "WILD"};
        System.out.println("\nMatrice de confusion (lignes = vrai, colonnes = prédiction) :");
        System.out.printf("%-8s %8s %8s %8s%n", "", "CHAT", "CHIEN", "WILD");
        for (int v = 0; v < 3; v++)
            System.out.printf("%-8s %8d %8d %8d%n", noms[v], conf[v][0], conf[v][1], conf[v][2]);
    }

    // cat=0, dog=1, wild=2
    static int labelOf(String chemin) {
        String c = chemin.toLowerCase();
        if (c.contains("/cat/")  || c.contains("\\cat\\"))  return 0;
        if (c.contains("/dog/")  || c.contains("\\dog\\"))  return 1;
        if (c.contains("/wild/") || c.contains("\\wild\\")) return 2;
        return -1;
    }

    // Renvoie la classe majoritaire parmi les K plus proches voisins
    static int classerKNN(float[] x, List<float[]> trainX, List<Integer> trainY) {
        final int n = trainX.size();
        double[] meilleuresDist = new double[K];
        int[] meilleursLab = new int[K];
        for (int i = 0; i < K; i++) { meilleuresDist[i] = Double.MAX_VALUE; meilleursLab[i] = -1; }

        for (int t = 0; t < n; t++) {
            final float[] v = trainX.get(t);
            double d = 0;
            for (int j = 0; j < x.length; j++) { double e = x[j] - v[j]; d += e*e; }
            // insère si meilleur qu'un des K courants
            int pireIdx = 0;
            for (int i = 1; i < K; i++) if (meilleuresDist[i] > meilleuresDist[pireIdx]) pireIdx = i;
            if (d < meilleuresDist[pireIdx]) { meilleuresDist[pireIdx] = d; meilleursLab[pireIdx] = trainY.get(t); }
        }
        int[] votes = new int[3];
        for (int i = 0; i < K; i++) if (meilleursLab[i] >= 0) votes[meilleursLab[i]]++;
        int best = 0;
        for (int c = 1; c < 3; c++) if (votes[c] > votes[best]) best = c;
        return best;
    }
}
