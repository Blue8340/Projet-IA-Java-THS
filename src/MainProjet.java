import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.io.File;
import javax.swing.*;
import javax.swing.border.EmptyBorder;
import java.awt.*;

public class MainProjet {

    // Constantes des classes
    private static final int LABEL_CHAT = 0;
    private static final int LABEL_CHIEN = 1;
    private static final int LABEL_WILD = 2;

    // Variables globales pour retenir les dimensions de l'image de référence
    private static int tailleReference = -1;
    private static int largeurRef = -1;
    private static int hauteurRef = -1;

    // Structure pour lier une image à son label
    static class DonneeEntrainement {
        float[] pixels;
        int label;
        DonneeEntrainement(float[] p, int l) {
            this.pixels = p;
            this.label = l;
        }
    }

    // ========================================================
    // 1. LE CHEF D'ORCHESTRE (MAIN)
    // ========================================================
    public static void main(String[] args) {
        String dossierTrain = "dataset_groupe_8/train/";
        String dossierTest = "dataset_groupe_8/test/";

        System.out.println("=== DÉMARRAGE DU PIPELINE IA (CLEAN CODE) ===");

        // Étape 1 : Chargement et préparation des données
        List<DonneeEntrainement> dataset = chargerEtAugmenterDonnees(dossierTrain);
        
        // Étape 2 : Entraînement des 3 experts (One-vs-All)
        System.out.println("\n=== ENTRAÎNEMENT DES EXPERTS ===");
        Neurone.fixeCoefApprentissage(0.01f); 
        float mseCible = 0.05f; 
        
        iNeurone expertChien = entrainerUnExpert(dataset, LABEL_CHIEN, "CHIEN", mseCible);
        iNeurone expertChat  = entrainerUnExpert(dataset, LABEL_CHAT, "CHAT", mseCible);
        iNeurone expertWild  = entrainerUnExpert(dataset, LABEL_WILD, "WILD", mseCible);

        // Étape 3 : Évaluations en console
        System.out.println("\n=== ÉVALUATIONS STATISTIQUES ===");
        testerLeModele(dossierTrain, expertChien, expertChat, expertWild, "TRAIN (Données connues)");
        testerLeModele(dossierTest, expertChien, expertChat, expertWild, "TEST (Données inconnues)");

        // Étape 4 : Lancement de l'interface graphique
        System.out.println("\n>>> Lancement de l'interface utilisateur...");
        SwingUtilities.invokeLater(() -> {
            creerEtAfficherInterface(expertChien, expertChat, expertWild);
        });
    }

    // ========================================================
    // 2. SOUS-MÉTHODE : CHARGEMENT ET AUGMENTATION (TSL + MIROIR)
    // ========================================================
    private static List<DonneeEntrainement> chargerEtAugmenterDonnees(String dossier) {
        List<String> chemins = Image.listeFichiers(dossier);
        List<DonneeEntrainement> dataset = new ArrayList<>();

        if (chemins == null) return dataset;

        for (String chemin : chemins) {
            int vraiLabel = -1;
            String c = chemin.toLowerCase();
            if (c.contains("/dog/") || c.contains("\\dog\\")) vraiLabel = LABEL_CHIEN;
            else if (c.contains("/cat/") || c.contains("\\cat\\")) vraiLabel = LABEL_CHAT;
            else if (c.contains("/wild/") || c.contains("\\wild\\")) vraiLabel = LABEL_WILD;

            if (vraiLabel != -1) {
                Image img = new Image(chemin, vraiLabel, false); // false = Couleur
                
                if (tailleReference == -1) {
                    tailleReference = img.taille();
                    largeurRef = img.largeur();
                    hauteurRef = img.hauteur();
                }
                
                if (img.taille() == tailleReference) {
                    float[] tslNormal = new float[tailleReference];
                    float[] tslMiroir = new float[tailleReference];
                    int[] pixelsBruts = img.donnees();

                    for (int y = 0; y < hauteurRef; y++) {
                        for (int x = 0; x < largeurRef; x++) {
                            int idxOrigine = (y * largeurRef + x) * 3;
                            int idxMiroir = (y * largeurRef + (largeurRef - 1 - x)) * 3;

                            float[] tsl = rgbVersTsl(pixelsBruts[idxOrigine], pixelsBruts[idxOrigine + 1], pixelsBruts[idxOrigine + 2]);

                            tslNormal[idxOrigine] = tsl[0]; tslNormal[idxOrigine + 1] = tsl[1]; tslNormal[idxOrigine + 2] = tsl[2];
                            tslMiroir[idxMiroir] = tsl[0]; tslMiroir[idxMiroir + 1] = tsl[1]; tslMiroir[idxMiroir + 2] = tsl[2];
                        }
                    }
                    dataset.add(new DonneeEntrainement(tslNormal, vraiLabel));
                    dataset.add(new DonneeEntrainement(tslMiroir, vraiLabel));
                }
            }
        }
        Collections.shuffle(dataset);
        System.out.println("Dataset chargé et doublé (Miroir) : " + dataset.size() + " images.");
        return dataset;
    }

    // ========================================================
    // 3. SOUS-MÉTHODE : ENTRAÎNEMENT D'UN SEUL EXPERT
    // ========================================================
    private static iNeurone entrainerUnExpert(List<DonneeEntrainement> dataset, int labelCible, String nom, float mseCible) {
        System.out.println(">>> Entraînement de l'Expert " + nom + "...");
        
        int nbImages = dataset.size();
        float[][] entrees = new float[nbImages][tailleReference];
        float[] objectifs = new float[nbImages];

        for (int i = 0; i < nbImages; i++) {
            entrees[i] = dataset.get(i).pixels;
            objectifs[i] = (dataset.get(i).label == labelCible) ? 1.0f : 0.0f;
        }

        iNeurone expert = new NeuroneSigmoide(tailleReference);
        expert.apprentissage(entrees, objectifs, mseCible);
        return expert;
    }

    // ========================================================
    // 4. SOUS-MÉTHODE : TEST ET SYSTÈME DE VOTE
    // ========================================================
    private static void testerLeModele(String dossier, iNeurone expertChien, iNeurone expertChat, iNeurone expertWild, String nomTest) {
        List<String> cheminsTest = Image.listeFichiers(dossier);
        int correct = 0, totalTest = 0;
        int compteChien = 0, compteChat = 0, compteWild = 0;

        if (cheminsTest != null) {
            for (String chemin : cheminsTest) {
                int vraiLabel = -1;
                String c = chemin.toLowerCase();
                if (c.contains("/dog/") || c.contains("\\dog\\")) vraiLabel = LABEL_CHIEN;
                else if (c.contains("/cat/") || c.contains("\\cat\\")) vraiLabel = LABEL_CHAT;
                else if (c.contains("/wild/") || c.contains("\\wild\\")) vraiLabel = LABEL_WILD;

                if (vraiLabel != -1) {
                    Image imgTest = new Image(chemin, vraiLabel, false);
                    if (imgTest.taille() != tailleReference) continue;

                    float[] tslTest = new float[tailleReference];
                    int[] pixelsBruts = imgTest.donnees();

                    for (int y = 0; y < hauteurRef; y++) {
                        for (int x = 0; x < largeurRef; x++) {
                            int idx = (y * largeurRef + x) * 3;
                            float[] tsl = rgbVersTsl(pixelsBruts[idx], pixelsBruts[idx+1], pixelsBruts[idx+2]);
                            tslTest[idx] = tsl[0]; tslTest[idx+1] = tsl[1]; tslTest[idx+2] = tsl[2];
                        }
                    }

                    expertChien.metAJour(tslTest); expertChat.metAJour(tslTest); expertWild.metAJour(tslTest);

                    int prediction = LABEL_CHIEN; 
                    float maxScore = expertChien.sortie();
                    if (expertChat.sortie() > maxScore) { prediction = LABEL_CHAT; maxScore = expertChat.sortie(); }
                    if (expertWild.sortie() > maxScore) { prediction = LABEL_WILD; maxScore = expertWild.sortie(); }

                    if (prediction == vraiLabel) correct++;
                    totalTest++;
                    
                    if (prediction == LABEL_CHIEN) compteChien++;
                    else if (prediction == LABEL_CHAT) compteChat++;
                    else if (prediction == LABEL_WILD) compteWild++;
                }
            }
        }

        if (totalTest > 0) {
            float taux = ((float) correct / totalTest) * 100f;
            System.out.printf("-> Score sur %s : %d/%d (%.2f%% de précision)\n", nomTest, correct, totalTest, taux);
        }
    }

    // ========================================================
    // 5. SOUS-MÉTHODE : OUTIL MATHÉMATIQUE (RGB vers TSL)
    // ========================================================
    public static float[] rgbVersTsl(int r, int g, int b) {
        float rf = r / 255.0f; float gf = g / 255.0f; float bf = b / 255.0f;
        float cmax = Math.max(rf, Math.max(gf, bf));
        float cmin = Math.min(rf, Math.min(gf, bf));
        float delta = cmax - cmin;
        float l = (cmax + cmin) / 2.0f;
        float s = (delta == 0) ? 0 : delta / (1.0f - Math.abs(2.0f * l - 1.0f));
        float h = 0;
        if (delta != 0) {
            if (cmax == rf) h = 60 * (((gf - bf) / delta) % 6);
            else if (cmax == gf) h = 60 * (((bf - rf) / delta) + 2);
            else if (cmax == bf) h = 60 * (((rf - gf) / delta) + 4);
        }
        if (h < 0) h += 360;
        return new float[]{ h / 360.0f, s, l };
    }

    // ========================================================
    // 6. SOUS-MÉTHODE : INTERFACE GRAPHIQUE (UI)
    // ========================================================
    private static void creerEtAfficherInterface(iNeurone expertChien, iNeurone expertChat, iNeurone expertWild) {
        try { UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName()); } catch (Exception e) {}

        JFrame frame = new JFrame("IA Groupe 8 - Scanner Animaux");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(650, 550);
        frame.setLayout(new BorderLayout(15, 15));
        frame.getContentPane().setBackground(new Color(240, 242, 245));

        JPanel panelHaut = new JPanel(new BorderLayout());
        panelHaut.setOpaque(false);
        panelHaut.setBorder(new EmptyBorder(20, 20, 10, 20));
        JLabel labelTitre = new JLabel("Module de Classification IA", JLabel.CENTER);
        labelTitre.setFont(new Font("SansSerif", Font.BOLD, 24));
        JLabel labelSousTitre = new JLabel("Réseau de neurones One-vs-All (TSL)", JLabel.CENTER);
        labelSousTitre.setFont(new Font("SansSerif", Font.PLAIN, 14));
        panelHaut.add(labelTitre, BorderLayout.NORTH);
        panelHaut.add(labelSousTitre, BorderLayout.SOUTH);
        frame.add(panelHaut, BorderLayout.NORTH);

        JPanel panelImageFond = new JPanel(new BorderLayout());
        panelImageFond.setBackground(Color.WHITE);
        panelImageFond.setBorder(BorderFactory.createCompoundBorder(new EmptyBorder(10, 30, 10, 30), BorderFactory.createLineBorder(new Color(200, 200, 200), 1, true)));
        JLabel labelImage = new JLabel("Cliquez en bas pour charger une image", JLabel.CENTER);
        panelImageFond.add(labelImage, BorderLayout.CENTER);
        frame.add(panelImageFond, BorderLayout.CENTER);

        JPanel panelBas = new JPanel(new BorderLayout(10, 15));
        panelBas.setOpaque(false);
        panelBas.setBorder(new EmptyBorder(10, 30, 30, 30));
        JButton boutonOuvrir = new JButton("Sélectionner une photo...");
        boutonOuvrir.setFont(new Font("SansSerif", Font.BOLD, 15));
        
        JPanel panelResultat = new JPanel(new BorderLayout());
        panelResultat.setBackground(new Color(220, 220, 220));
        panelResultat.setBorder(new EmptyBorder(15, 10, 15, 10));
        JLabel labelResultat = new JLabel("En attente d'analyse...", JLabel.CENTER);
        labelResultat.setFont(new Font("SansSerif", Font.BOLD, 16));
        panelResultat.add(labelResultat, BorderLayout.CENTER);

        panelBas.add(boutonOuvrir, BorderLayout.NORTH);
        panelBas.add(panelResultat, BorderLayout.SOUTH);
        frame.add(panelBas, BorderLayout.SOUTH);

        boutonOuvrir.addActionListener(e -> {
            JFileChooser selecteur = new JFileChooser("dataset_groupe_8/test/");
            if (selecteur.showOpenDialog(frame) == JFileChooser.APPROVE_OPTION) {
                String chemin = selecteur.getSelectedFile().getAbsolutePath();
                java.awt.Image imgRedimensionnee = new ImageIcon(chemin).getImage().getScaledInstance(260, 260, java.awt.Image.SCALE_SMOOTH);
                labelImage.setIcon(new ImageIcon(imgRedimensionnee));
                labelImage.setText(""); 

                Image imgIA = new Image(chemin, -1, false);
                if (imgIA.taille() != tailleReference) {
                    labelResultat.setText("Format d'image non supporté");
                    panelResultat.setBackground(new Color(255, 200, 200));
                    return;
                }

                float[] tslFenetre = new float[tailleReference];
                int[] pixelsBruts = imgIA.donnees();
                for (int y = 0; y < hauteurRef; y++) {
                    for (int x = 0; x < largeurRef; x++) {
                        int idx = (y * largeurRef + x) * 3;
                        float[] tsl = rgbVersTsl(pixelsBruts[idx], pixelsBruts[idx+1], pixelsBruts[idx+2]);
                        tslFenetre[idx] = tsl[0]; tslFenetre[idx+1] = tsl[1]; tslFenetre[idx+2] = tsl[2];
                    }
                }

                expertChien.metAJour(tslFenetre); expertChat.metAJour(tslFenetre); expertWild.metAJour(tslFenetre);
                
                float sChien = expertChien.sortie(), sChat = expertChat.sortie(), sWild = expertWild.sortie();
                String verdict = "CHIEN"; float maxScore = sChien;
                Color bgColor = new Color(212, 245, 212); Color textColor = new Color(0, 100, 0);

                if (sChat > maxScore) { verdict = "CHAT"; maxScore = sChat; bgColor = new Color(255, 235, 204); textColor = new Color(200, 100, 0); }
                if (sWild > maxScore) { verdict = "WILD"; maxScore = sWild; bgColor = new Color(255, 215, 215); textColor = new Color(180, 0, 0); }

                labelResultat.setText(String.format("VERDICT : %s  (Précision max : %.1f%%)", verdict, maxScore * 100f));
                panelResultat.setBackground(bgColor); labelResultat.setForeground(textColor);
            }
        });

        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
}
