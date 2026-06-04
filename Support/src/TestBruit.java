package src;

import java.util.Random;

// Extension E5 : robustesse d'un neurone face à des entrées bruitées.
// Démarche : on apprend une fonction logique PARFAITE sur des données propres,
// puis on alimente le neurone avec les mêmes cas mais bruités, et on mesure
// la précision en fonction de l'amplitude du bruit (-> notion de signal/bruit).
public class TestBruit
{
    // Fonction cible : true = ET, false = OU
    static final boolean APPRENDRE_ET = true;

    public static void main(String[] args)
    {
        final float[][] entrees   = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        final float[]   resultats = APPRENDRE_ET ? new float[]{0, 0, 0, 1}   // ET
                                                  : new float[]{0, 1, 1, 1};  // OU

        // 1) APPRENTISSAGE SUR DONNÉES PROPRES (neurone "parfait")
        // Changez ici le type de neurone à tester :
        // final iNeurone n = new NeuroneHeavyside(2);
         final iNeurone n = new NeuroneReLu(2);
        // final iNeurone n = new NeuroneSigmoide(2);
        n.apprentissage(entrees, resultats, 0.001f);
        System.out.println("Apprentissage terminé (fonction " + (APPRENDRE_ET ? "ET" : "OU") + ")");

        // 1bis) VÉRIFICATION : le neurone a-t-il VRAIMENT appris la fonction propre ?
        // Indispensable : l'expérience de bruit n'a de sens que si le neurone
        // est "parfait" sur les données propres (sinon on mesure une erreur, pas du bruit).
        System.out.println("Vérification sur données propres :");
        boolean parfait = true;
        for (int i = 0; i < 4; i++)
        {
            n.metAJour(entrees[i]);
            final int prediction = (n.sortie() >= 0.5f) ? 1 : 0;
            final int attendu = (int) resultats[i];
            if (prediction != attendu) parfait = false;
            System.out.printf("  cas {%.0f,%.0f} : prediction=%d (attendu %d)%n",
                    entrees[i][0], entrees[i][1], prediction, attendu);
        }
        if (!parfait)
            System.out.println("  /!\\ ATTENTION : apprentissage NON parfait -> le test de bruit ci-dessous n'est pas fiable.");
        System.out.println();

        // 2) TEST DE ROBUSTESSE : on bruite les entrées
        final int     nbEssais   = 10000;  // tirages par niveau de bruit
        final float[] amplitudes = {0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.7f, 1.0f};
        final Random  rand       = new Random();

        System.out.printf("%-12s %-12s %-12s%n", "Amplitude", "SNR (1/amp)", "Précision");
        for (final float amp : amplitudes)
        {
            int correct = 0;
            for (int essai = 0; essai < nbEssais; essai++)
            {
                final int cas = rand.nextInt(4);          // un des 4 cas au hasard
                final float[] bruite = new float[2];
                for (int j = 0; j < 2; j++)
                {
                    // bruit uniforme dans [-amp, +amp] ajouté autour de 0 ou 1
                    final float bruit = (rand.nextFloat() * 2f - 1f) * amp;
                    bruite[j] = entrees[cas][j] + bruit;
                }
                n.metAJour(bruite);
                // On seuille la sortie PUIS on la compare à la réponse PROPRE attendue pour ce cas
                final int prediction = (n.sortie() >= 0.5f) ? 1 : 0;
                if (prediction == (int) resultats[cas]) correct++;
            }
            final float precision = 100f * correct / nbEssais;
            final String snr = (amp == 0f) ? "inf" : String.format("%.2f", 1f / amp);
            System.out.printf("%-12.2f %-12s %-10.2f%%%n", amp, snr, precision);
        }
    }
}
