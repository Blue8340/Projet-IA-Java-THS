import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.plaf.basic.BasicComboBoxUI;
import java.awt.*;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;

/**
 * Classe principale d'orchestration pour le système de classification multi-classes.
 * Implémente un pipeline complet : Extraction de caractéristiques (HOG, FFT, TSL),
 * Augmentation de données géométrique, et Apprentissage supervisé One-vs-All.
 * * @author Groupe 8
 * @version 1.0 (Production-ready)
 */
public class MainProjet {

    // ========================================================
    // [CONSTANTES & PARAMÈTRES GLOBAUX]
    // ========================================================
    private static final int LABEL_CHAT = 0;
    private static final int LABEL_CHIEN = 1;
    private static final int LABEL_WILD = 2;

    private static int tailleReferenceVector = -1;
    private static int largeurReference = -1;
    private static int hauteurReference = -1;

    // Experts d'inférence (Architecture One-vs-All)
    private static iNeurone expertChien = null;
    private static iNeurone expertChat = null;
    private static iNeurone expertWild = null;

    // État de la configuration active
    private static String configurationCouranteSignal = "HOG";
    private static boolean configurationCouranteNorm = true;

    /**
     * Structure de données encapsulant le vecteur caractéristique et son étiquette.
     */
    static class DonneeEntrainement {
        final float[] features;
        final int etiquette;

        DonneeEntrainement(float[] features, int etiquette) {
            this.features = features;
            this.etiquette = etiquette;
        }
    }

    public static void main(String[] args) {
        System.out.println("======================================================");
        System.out.println("=== INITIALISATION DU PIPELINE NEURONAL (V1.0)     ===");
        System.out.println("======================================================");
        SwingUtilities.invokeLater(() -> creerEtAfficherInterface());
    }

    // ========================================================
    // [MODULE 1] : MOTEUR DE TRAITEMENT DU SIGNAL & MATHÉMATIQUES
    // Centralise l'extraction de features (HOG, FFT, TSL) et la Data Augmentation
    // ========================================================
    static class MoteurTraitementSignal {

        /**
         * Extrait la matrice de luminance d'un objet Image.
         */
        private static double[][] extraireLuminance(Image img) {
            int h = img.hauteur(), w = img.largeur();
            int[] donneesBrutes = img.donnees();
            boolean estMonochrome = img.estEnNiveauxDeGris();
            double[][] grille = new double[h][w];
            
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    int idx = i * w + j;
                    if (estMonochrome) {
                        grille[i][j] = donneesBrutes[idx];
                    } else {
                        // Pondération standard ITU-R BT.709 pour la luminance
                        grille[i][j] = 0.2125 * donneesBrutes[3*idx] + 0.7154 * donneesBrutes[3*idx+1] + 0.0721 * donneesBrutes[3*idx+2];
                    }
                }
            }
            return grille;
        }

        private static double interpolationBilineaire(double[][] g, double sx, double sy) {
            final int H = g.length, W = g[0].length;
            int x0 = (int)Math.floor(sx), y0 = (int)Math.floor(sy);
            final double fx = sx - x0, fy = sy - y0;
            int x1 = Math.max(0, Math.min(W-1, x0 + 1)), y1 = Math.max(0, Math.min(H-1, y0 + 1));
            x0 = Math.max(0, Math.min(W-1, x0)); y0 = Math.max(0, Math.min(H-1, y0));
            
            final double haut = g[y0][x0] + (g[y0][x1] - g[y0][x0]) * fx;
            final double bas = g[y1][x0] + (g[y1][x1] - g[y1][x0]) * fx;
            return haut + (bas - haut) * fy;
        }

        /* --- MÉTHODES D'AUGMENTATION DE DONNÉES --- */
        
        private static double[][] appliquerMiroirHorizontal(double[][] g) {
            final int H = g.length, W = g[0].length;
            final double[][] resultat = new double[H][W];
            for (int i = 0; i < H; i++) 
                for (int j = 0; j < W; j++) 
                    resultat[i][j] = g[i][W-1-j];
            return resultat;
        }

        private static double[][] appliquerRotation(double[][] g, double angleDegres) {
            final int H = g.length, W = g[0].length;
            final double rad = Math.toRadians(angleDegres), cos = Math.cos(rad), sin = Math.sin(rad);
            final double cx = (W-1)/2.0, cy = (H-1)/2.0;
            final double[][] resultat = new double[H][W];
            for (int i = 0; i < H; i++) 
                for (int j = 0; j < W; j++) {
                    final double dx = j - cx, dy = i - cy;
                    resultat[i][j] = interpolationBilineaire(g, cos*dx + sin*dy + cx, -sin*dx + cos*dy + cy);
                }
            return resultat;
        }

        private static double[][] appliquerTranslation(double[][] g, int dx, int dy) {
            final int H = g.length, W = g[0].length;
            final double[][] resultat = new double[H][W];
            for (int i = 0; i < H; i++) 
                for (int j = 0; j < W; j++) {
                    int srcI = Math.max(0, Math.min(H-1, i - dy));
                    int srcJ = Math.max(0, Math.min(W-1, j - dx));
                    resultat[i][j] = g[srcI][srcJ];
                }
            return resultat;
        }

        private static double[][] appliquerZoom(double[][] g, double facteur) {
            final int H = g.length, W = g[0].length;
            final double cx = (W-1)/2.0, cy = (H-1)/2.0;
            final double[][] resultat = new double[H][W];
            for (int i = 0; i < H; i++) 
                for (int j = 0; j < W; j++)
                    resultat[i][j] = interpolationBilineaire(g, cx + (j - cx)/facteur, cy + (i - cy)/facteur);
            return resultat;
        }

        private static double[][] appliquerBruitGaussien(double[][] g, double amplitude) {
            final int H = g.length, W = g[0].length;
            final Random generateurAleatoire = new Random();
            final double[][] resultat = new double[H][W];
            for (int i = 0; i < H; i++) 
                for (int j = 0; j < W; j++)
                    resultat[i][j] = Math.max(0, Math.min(255, g[i][j] + generateurAleatoire.nextGaussian() * amplitude));
            return resultat;
        }

        /* --- ALGORITHMES D'EXTRACTION (HOG, FFT, TSL) --- */

        /**
         * Calcule l'Histogramme des Gradients Orientés (HOG) pour capturer la morphologie.
         */
        private static float[] extraireHOG(double[][] g) {
            final int H = g.length, W = g[0].length;
            final int TAILLE_CELLULE = 8, NB_BINS = 9;
            final double[][] magnitude = new double[H][W], orientation = new double[H][W];
            
            // Calcul des gradients (Sobel)
            for (int i = 0; i < H; i++) {
                for (int j = 0; j < W; j++) {
                    double gradX = (j == 0) ? g[i][j+1]-g[i][j] : (j == W-1) ? g[i][j]-g[i][j-1] : g[i][j+1]-g[i][j-1];
                    double gradY = (i == 0) ? g[i+1][j]-g[i][j] : (i == H-1) ? g[i][j]-g[i-1][j] : g[i+1][j]-g[i-1][j];
                    
                    magnitude[i][j] = Math.sqrt(gradX*gradX + gradY*gradY);
                    double angle = Math.toDegrees(Math.atan2(gradY, gradX));
                    if (angle < 0) angle += 180.0; 
                    if (angle >= 180.0) angle -= 180.0;
                    orientation[i][j] = angle;
                }
            }
            
            // Construction des histogrammes locaux
            final int cellulesY = H / TAILLE_CELLULE, cellulesX = W / TAILLE_CELLULE;
            final float[] vecteurCaracteristique = new float[cellulesY * cellulesX * NB_BINS];
            final double largeurBin = 180.0 / NB_BINS; 
            int index = 0;
            
            for (int cy = 0; cy < cellulesY; cy++) {
                for (int cx = 0; cx < cellulesX; cx++) {
                    final double[] histogrammeLocal = new double[NB_BINS];
                    for (int di = 0; di < TAILLE_CELLULE; di++) {
                        for (int dj = 0; dj < TAILLE_CELLULE; dj++) {
                            final int i = cy*TAILLE_CELLULE+di, j = cx*TAILLE_CELLULE+dj;
                            int bin = (int)(orientation[i][j] / largeurBin); 
                            if (bin >= NB_BINS) bin = NB_BINS - 1;
                            histogrammeLocal[bin] += magnitude[i][j];
                        }
                    }
                    // Normalisation L2 (Block Normalization)
                    double norme = 0; 
                    for (double val : histogrammeLocal) norme += val * val;
                    norme = Math.sqrt(norme) + 1e-6;
                    for (int b = 0; b < NB_BINS; b++) {
                        vecteurCaracteristique[index++] = (float)(histogrammeLocal[b] / norme);
                    }
                }
            }
            return vecteurCaracteristique;
        }

        public static float[] convertRGBtoTSL(int r, int g, int b) {
            float rf = r / 255.0f, gf = g / 255.0f, bf = b / 255.0f;
            float max = Math.max(rf, Math.max(gf, bf)), min = Math.min(rf, Math.min(gf, bf));
            float delta = max - min, luminance = (max + min) / 2.0f;
            float saturation = (delta == 0) ? 0 : delta / (1.0f - Math.abs(2.0f * luminance - 1.0f));
            float teinte = 0;
            if (delta != 0) {
                if (max == rf) teinte = 60 * (((gf - bf) / delta) % 6);
                else if (max == gf) teinte = 60 * (((bf - rf) / delta) + 2);
                else teinte = 60 * (((rf - gf) / delta) + 4);
            }
            if (teinte < 0) teinte += 360;
            return new float[]{ teinte / 360.0f, saturation, luminance };
        }

        // Interfaces publiques d'accès
        public static float[] executerExtractionHOG(Image img) { return extraireHOG(extraireLuminance(img)); }
        public static float[] executerHOGMiroir(Image img) { return extraireHOG(appliquerMiroirHorizontal(extraireLuminance(img))); }
        public static float[] executerHOGRotation(Image img, double a) { return extraireHOG(appliquerRotation(extraireLuminance(img), a)); }
        public static float[] executerHOGTranslation(Image img, int x, int y) { return extraireHOG(appliquerTranslation(extraireLuminance(img), x, y)); }
        public static float[] executerHOGZoom(Image img, double f) { return extraireHOG(appliquerZoom(extraireLuminance(img), f)); }
        public static float[] executerHOGBruit(Image img, double a) { return extraireHOG(appliquerBruitGaussien(extraireLuminance(img), a)); }
    }

    // ========================================================
    // [MODULE 2] : ORCHESTRATEUR D'APPRENTISSAGE (MACHINE LEARNING)
    // ========================================================
    
    private static float[] extraireVecteur(Image img, String modalite, boolean normaliser) {
        if (modalite.equals("HOG")) {
            return MoteurTraitementSignal.executerExtractionHOG(img); 
        } 
        // Gestion des pixels bruts et transformations spatiales
        float[] features = new float[img.taille()];
        int[] pixels = img.donnees();
        float facteurNorm = normaliser ? 255.0f : 1.0f;
        
        if (modalite.equals("Niveaux de gris")) {
            for (int j = 0; j < img.taille(); j++) features[j] = pixels[j] / facteurNorm;
        } else {
            for (int y = 0; y < img.hauteur(); y++) {
                for (int x = 0; x < img.largeur(); x++) {
                    int idx = (y * img.largeur() + x) * 3;
                    if (modalite.equals("TSL")) {
                        float[] tsl = MoteurTraitementSignal.convertRGBtoTSL(pixels[idx], pixels[idx+1], pixels[idx+2]);
                        features[idx] = tsl[0]; features[idx+1] = tsl[1]; features[idx+2] = tsl[2];
                        if (!normaliser) { features[idx]*=255; features[idx+1]*=255; features[idx+2]*=255; }
                    } else { // RGB Classique
                        features[idx] = pixels[idx]/facteurNorm;
                        features[idx+1] = pixels[idx+1]/facteurNorm;
                        features[idx+2] = pixels[idx+2]/facteurNorm;
                    }
                }
            }
        }
        return features;
    }

    /**
     * Pilote le cycle d'apprentissage du modèle : Chargement, Prétraitement, Augmentation, Descente de gradient.
     */
    private static String executerPipelineEntrainement(String signal, String activation, boolean utiliserShuffle, 
                                                       boolean normaliser, boolean miroir, boolean rot, boolean trans, 
                                                       boolean zoom, boolean bruit, float tauxApprentissage, float mseCible, 
                                                       java.util.function.Consumer<String> observateurLog) {
        
        observateurLog.accept("> Initialisation des tenseurs (" + signal + ")...");
        List<String> cheminsFichiers = Image.listeFichiers("dataset_groupe_8/train/");
        List<DonneeEntrainement> corpus = new ArrayList<>();

        if (cheminsFichiers == null || cheminsFichiers.isEmpty()) return "Échec : Dataset introuvable.";

        tailleReferenceVector = -1;
        boolean modeMonochrome = signal.equals("Niveaux de gris");
        
        for (String chemin : cheminsFichiers) {
            int etiquetteAttendue = -1;
            String nomNormalise = chemin.toLowerCase();
            if (nomNormalise.contains("/dog/") || nomNormalise.contains("\\dog\\")) etiquetteAttendue = LABEL_CHIEN;
            else if (nomNormalise.contains("/cat/") || nomNormalise.contains("\\cat\\")) etiquetteAttendue = LABEL_CHAT;
            else if (nomNormalise.contains("/wild/") || nomNormalise.contains("\\wild\\")) etiquetteAttendue = LABEL_WILD;

            if (etiquetteAttendue != -1) {
                Image instance = new Image(chemin, etiquetteAttendue, modeMonochrome);
                float[] vecteurInitial = extraireVecteur(instance, signal, normaliser);
                
                if (tailleReferenceVector == -1) {
                    tailleReferenceVector = vecteurInitial.length; largeurReference = instance.largeur(); hauteurReference = instance.hauteur();
                }
                
                if (vecteurInitial.length == tailleReferenceVector) {
                    corpus.add(new DonneeEntrainement(vecteurInitial, etiquetteAttendue));

                    // Pipeline d'augmentation des données synthétiques
                    if (signal.equals("HOG")) {
                        if (miroir) corpus.add(new DonneeEntrainement(MoteurTraitementSignal.executerHOGMiroir(instance), etiquetteAttendue));
                        if (rot) {
                            corpus.add(new DonneeEntrainement(MoteurTraitementSignal.executerHOGRotation(instance, 10), etiquetteAttendue));
                            corpus.add(new DonneeEntrainement(MoteurTraitementSignal.executerHOGRotation(instance, -10), etiquetteAttendue));
                        }
                        if (trans) {
                            corpus.add(new DonneeEntrainement(MoteurTraitementSignal.executerHOGTranslation(instance, 3, 0), etiquetteAttendue));
                            corpus.add(new DonneeEntrainement(MoteurTraitementSignal.executerHOGTranslation(instance, -3, 0), etiquetteAttendue));
                        }
                        if (zoom) {
                            corpus.add(new DonneeEntrainement(MoteurTraitementSignal.executerHOGZoom(instance, 1.1), etiquetteAttendue));
                            corpus.add(new DonneeEntrainement(MoteurTraitementSignal.executerHOGZoom(instance, 0.9), etiquetteAttendue));
                        }
                        if (bruit) corpus.add(new DonneeEntrainement(MoteurTraitementSignal.executerHOGBruit(instance, 8.0), etiquetteAttendue));
                    }
                }
            }
        }

        if (utiliserShuffle) Collections.shuffle(corpus); 

        int volumeDonnees = corpus.size();
        float[][] tenseurEntrees = new float[volumeDonnees][tailleReferenceVector];
        float[][] tenseurObjectifs = new float[3][volumeDonnees];

        for (int i = 0; i < volumeDonnees; i++) {
            tenseurEntrees[i] = corpus.get(i).features;
            tenseurObjectifs[0][i] = (corpus.get(i).etiquette == LABEL_CHIEN) ? 1.0f : 0.0f;
            tenseurObjectifs[1][i] = (corpus.get(i).etiquette == LABEL_CHAT)  ? 1.0f : 0.0f;
            tenseurObjectifs[2][i] = (corpus.get(i).etiquette == LABEL_WILD)  ? 1.0f : 0.0f;
        }

        Neurone.fixeCoefApprentissage(tauxApprentissage); 

        // Injection des dépendances d'activation
        if (activation.equals("Sigmoïde")) {
            expertChien = new NeuroneSigmoide(tailleReferenceVector); expertChat = new NeuroneSigmoide(tailleReferenceVector); expertWild = new NeuroneSigmoide(tailleReferenceVector);
        } else if (activation.equals("ReLU")) {
            expertChien = new NeuroneReLU(tailleReferenceVector); expertChat = new NeuroneReLU(tailleReferenceVector); expertWild = new NeuroneReLU(tailleReferenceVector);
        } else {
            expertChien = new NeuroneHeavyside(tailleReferenceVector); expertChat = new NeuroneHeavyside(tailleReferenceVector); expertWild = new NeuroneHeavyside(tailleReferenceVector);
        }

        // Exécution de l'algorithme d'apprentissage
        observateurLog.accept(String.format("> Ajustement des poids [EXPERT CHIEN] (Taille du lot : %d)...", volumeDonnees));
        expertChien.apprentissage(tenseurEntrees, tenseurObjectifs[0], mseCible);
        
        observateurLog.accept("> Ajustement des poids [EXPERT CHAT]...");
        expertChat.apprentissage(tenseurEntrees, tenseurObjectifs[1], mseCible);
        
        observateurLog.accept("> Ajustement des poids [EXPERT SAUVAGE]...");
        expertWild.apprentissage(tenseurEntrees, tenseurObjectifs[2], mseCible);

        configurationCouranteSignal = signal; configurationCouranteNorm = normaliser;
        return String.format("Architecture neuronale compilée. [Activation : %s | Mode : %s]", activation, signal);
    }

    // ========================================================
    // [MODULE 3] : ARCHITECTURE DU DESIGN SYSTEM UI
    // Isolant les surcharges Swing pour garantir la clarté du code métier.
    // ========================================================
    static class DesignSystem {
        static final Color THEME_FOND = new Color(34, 56, 43);      
        static final Color THEME_PANNEAU = new Color(55, 80, 60);     
        static final Color THEME_TEXTE = new Color(240, 235, 225);   
        static final Color THEME_ACCENT = new Color(210, 175, 130);
        static final Color THEME_TEXTE_BOUTON = new Color(60, 40, 20);      
        static final Color THEME_SUCCES = new Color(150, 200, 120);
        static final Color THEME_ERREUR = new Color(200, 100, 80);   

        static class PanneauArrondi extends JPanel {
            private int rayon; private Color couleurFond;
            public PanneauArrondi(LayoutManager layout, int r, Color c) { super(layout); rayon = r; couleurFond = c; setOpaque(false); }
            @Override protected void paintComponent(Graphics g) {
                super.paintComponent(g); Graphics2D g2 = (Graphics2D) g;
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g2.setColor(couleurFond); g2.fillRoundRect(0, 0, getWidth(), getHeight(), rayon, rayon);
            }
        }

        static class SaisieArrondie extends JTextField {
            public SaisieArrondie(String texte, int colonnes) {
                super(texte, colonnes); setOpaque(false); setBackground(THEME_PANNEAU.brighter()); 
                setForeground(THEME_TEXTE); setCaretColor(THEME_ACCENT); setBorder(new EmptyBorder(5, 10, 5, 10));
            }
            @Override protected void paintComponent(Graphics g) {
                Graphics2D g2 = (Graphics2D) g; g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g2.setColor(getBackground()); g2.fillRoundRect(0, 0, getWidth(), getHeight(), 20, 20); super.paintComponent(g);
            }
        }

        static class BoutonArrondi extends JButton {
            public BoutonArrondi(String etiquette) {
                super(etiquette); setContentAreaFilled(false); setFocusPainted(false); setBorderPainted(false); setOpaque(false);
            }
            @Override protected void paintComponent(Graphics g) {
                Graphics2D g2 = (Graphics2D) g.create(); g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                if (getModel().isArmed()) g2.setColor(getBackground().darker());
                else if (getModel().isRollover()) g2.setColor(getBackground().brighter());
                else g2.setColor(getBackground());
                g2.fillRoundRect(0, 0, getWidth(), getHeight(), 30, 30);
                FontMetrics m = g2.getFontMetrics(getFont());
                int x = (getWidth() - m.stringWidth(getText())) / 2, y = ((getHeight() - m.getHeight()) / 2) + m.getAscent();
                g2.setColor(getForeground()); g2.setFont(getFont()); g2.drawString(getText(), x, y); g2.dispose();
            }
        }

        static class SelecteurArrondi<E> extends JComboBox<E> {
            public SelecteurArrondi(E[] items) {
                super(items); setOpaque(false); setBackground(THEME_PANNEAU.brighter()); setForeground(THEME_TEXTE);
                setUI(new BasicComboBoxUI() { @Override protected JButton createArrowButton() { JButton b = super.createArrowButton(); b.setContentAreaFilled(false); b.setBorder(null); return b; }});
            }
            @Override protected void paintComponent(Graphics g) {
                Graphics2D g2 = (Graphics2D) g; g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g2.setColor(getBackground()); g2.fillRoundRect(0, 0, getWidth(), getHeight(), 20, 20); super.paintComponent(g);
            }
        }

        static void appliquerStyle(JComponent c) {
            c.setBackground(THEME_PANNEAU.brighter()); c.setForeground(THEME_TEXTE); c.setFont(new Font("SansSerif", Font.PLAIN, 14));
        }
    }

    private static ImageIcon adapterRenduImage(java.awt.Image img, int maxW, int maxH) {
        if (img == null || maxW <= 0 || maxH <= 0) return null;
        double ratio = Math.min((double) maxW / img.getWidth(null), (double) maxH / img.getHeight(null));
        return new ImageIcon(img.getScaledInstance((int) (img.getWidth(null) * ratio), (int) (img.getHeight(null) * ratio), java.awt.Image.SCALE_SMOOTH));
    }

    // ========================================================
    // CONTRÔLEUR D'INTERFACE ET POINT D'ENTRÉE GRAPHIQUE
    // ========================================================
    private static void creerEtAfficherInterface() {
        try { UIManager.setLookAndFeel(UIManager.getCrossPlatformLookAndFeelClassName()); } catch (Exception e) {}

        JFrame conteneur = new JFrame("Intelligence Artificielle - Inférence Avancée");
        conteneur.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        conteneur.setSize(1100, 900);
        conteneur.setMinimumSize(new Dimension(1000, 700));
        conteneur.getContentPane().setBackground(DesignSystem.THEME_FOND);
        conteneur.setLayout(new BorderLayout(20, 20));
        ((JPanel)conteneur.getContentPane()).setBorder(new EmptyBorder(20, 20, 20, 20));

        JLabel enTete = new JLabel("Module de Classification", JLabel.CENTER);
        enTete.setFont(new Font("SansSerif", Font.BOLD, 26)); enTete.setForeground(DesignSystem.THEME_ACCENT);
        conteneur.add(enTete, BorderLayout.NORTH);

        JPanel panneauCentral = new JPanel(new BorderLayout(15, 15)); panneauCentral.setOpaque(false);
        DesignSystem.PanneauArrondi blocConfiguration = new DesignSystem.PanneauArrondi(new GridLayout(3, 1, 10, 10), 30, DesignSystem.THEME_PANNEAU);
        blocConfiguration.setBorder(new EmptyBorder(15, 15, 15, 15));

        /* --- Configuration des Hyperparamètres --- */
        JPanel ligneOptions = new JPanel(new FlowLayout(FlowLayout.CENTER, 20, 5)); ligneOptions.setOpaque(false);
        DesignSystem.SelecteurArrondi<String> modSignal = new DesignSystem.SelecteurArrondi<>(new String[]{"Niveaux de gris", "RGB", "TSL", "HOG"});
        modSignal.setSelectedItem("HOG"); DesignSystem.appliquerStyle(modSignal);
        DesignSystem.SelecteurArrondi<String> modActivation = new DesignSystem.SelecteurArrondi<>(new String[]{"Sigmoïde", "ReLU", "Heaviside"});
        modActivation.setSelectedItem("Sigmoïde"); DesignSystem.appliquerStyle(modActivation);
        
        JCheckBox optMelange = new JCheckBox("Mélange des tenseurs", true); DesignSystem.appliquerStyle(optMelange); optMelange.setOpaque(false);
        JCheckBox optNorm = new JCheckBox("Normalisation L2", true); DesignSystem.appliquerStyle(optNorm); optNorm.setOpaque(false);

        ligneOptions.add(new JLabel("Signal d'entrée:") {{ setForeground(DesignSystem.THEME_TEXTE); }}); ligneOptions.add(modSignal);
        ligneOptions.add(new JLabel("Fonction cible:") {{ setForeground(DesignSystem.THEME_TEXTE); }}); ligneOptions.add(modActivation);
        ligneOptions.add(optMelange); ligneOptions.add(optNorm);

        /* --- Moteur d'Augmentation Synthétique --- */
        JPanel ligneAugmentation = new JPanel(new FlowLayout(FlowLayout.CENTER, 15, 5)); ligneAugmentation.setOpaque(false);
        JCheckBox augMiroir = new JCheckBox("Miroir", false); DesignSystem.appliquerStyle(augMiroir); augMiroir.setOpaque(false);
        JCheckBox augRotation = new JCheckBox("Rotation", false); DesignSystem.appliquerStyle(augRotation); augRotation.setOpaque(false);
        JCheckBox augTrans = new JCheckBox("Translation", false); DesignSystem.appliquerStyle(augTrans); augTrans.setOpaque(false);
        JCheckBox augZoom = new JCheckBox("Zoom", false); DesignSystem.appliquerStyle(augZoom); augZoom.setOpaque(false);
        JCheckBox augBruit = new JCheckBox("Bruit Additif", false); DesignSystem.appliquerStyle(augBruit); augBruit.setOpaque(false);
        
        ligneAugmentation.add(new JLabel("Augmentation Synthétique:") {{ setForeground(DesignSystem.THEME_ACCENT); }});
        ligneAugmentation.add(augMiroir); ligneAugmentation.add(augRotation); ligneAugmentation.add(augTrans); ligneAugmentation.add(augZoom); ligneAugmentation.add(augBruit);

        // Les transformations avancées nécessitent HOG (Traitement spatial)
        modSignal.addActionListener(e -> {
            boolean modeAvanceHog = modSignal.getSelectedItem().equals("HOG");
            augRotation.setEnabled(modeAvanceHog); augTrans.setEnabled(modeAvanceHog); augZoom.setEnabled(modeAvanceHog); augBruit.setEnabled(modeAvanceHog);
            if (!modeAvanceHog) { augRotation.setSelected(false); augTrans.setSelected(false); augZoom.setSelected(false); augBruit.setSelected(false); }
        });

        /* --- Contrôles de Convergence & Persistance --- */
        JPanel ligneActions = new JPanel(new FlowLayout(FlowLayout.CENTER, 15, 5)); ligneActions.setOpaque(false);
        DesignSystem.SaisieArrondie valEta = new DesignSystem.SaisieArrondie("0.0001", 6); DesignSystem.appliquerStyle(valEta); 
        DesignSystem.SaisieArrondie valMse = new DesignSystem.SaisieArrondie("0.15", 4); DesignSystem.appliquerStyle(valMse); 
        
        DesignSystem.BoutonArrondi cmdEntrainer = new DesignSystem.BoutonArrondi("Initier l'Apprentissage");
        cmdEntrainer.setBackground(DesignSystem.THEME_ACCENT); cmdEntrainer.setForeground(DesignSystem.THEME_TEXTE_BOUTON);
        cmdEntrainer.setFont(new Font("SansSerif", Font.BOLD, 14)); cmdEntrainer.setPreferredSize(new Dimension(200, 35));

        DesignSystem.BoutonArrondi cmdSauvegarder = new DesignSystem.BoutonArrondi("Exporter Modèle");
        cmdSauvegarder.setBackground(DesignSystem.THEME_PANNEAU.brighter()); cmdSauvegarder.setForeground(DesignSystem.THEME_TEXTE);
        cmdSauvegarder.setFont(new Font("SansSerif", Font.BOLD, 12)); cmdSauvegarder.setPreferredSize(new Dimension(160, 35));

        DesignSystem.BoutonArrondi cmdCharger = new DesignSystem.BoutonArrondi("Importer Modèle");
        cmdCharger.setBackground(DesignSystem.THEME_PANNEAU.brighter()); cmdCharger.setForeground(DesignSystem.THEME_TEXTE);
        cmdCharger.setFont(new Font("SansSerif", Font.BOLD, 12)); cmdCharger.setPreferredSize(new Dimension(140, 35));

        ligneActions.add(new JLabel("Taux d'apprentissage:") {{ setForeground(DesignSystem.THEME_TEXTE); }}); ligneActions.add(valEta);
        ligneActions.add(new JLabel("Seuil de convergence (MSE):") {{ setForeground(DesignSystem.THEME_TEXTE); }}); ligneActions.add(valMse);
        ligneActions.add(cmdEntrainer); ligneActions.add(cmdSauvegarder); ligneActions.add(cmdCharger);

        blocConfiguration.add(ligneOptions); blocConfiguration.add(ligneAugmentation); blocConfiguration.add(ligneActions);
        panneauCentral.add(blocConfiguration, BorderLayout.NORTH);

        /* --- Interface Visuelle --- */
        DesignSystem.PanneauArrondi blocRenduVisuel = new DesignSystem.PanneauArrondi(new BorderLayout(), 30, DesignSystem.THEME_PANNEAU);
        blocRenduVisuel.setBorder(new EmptyBorder(20, 20, 20, 20));
        JLabel afficheurImage = new JLabel("En attente de paramètres...", JLabel.CENTER);
        afficheurImage.setFont(new Font("SansSerif", Font.ITALIC, 16)); afficheurImage.setForeground(DesignSystem.THEME_TEXTE.darker());
        blocRenduVisuel.add(afficheurImage, BorderLayout.CENTER); panneauCentral.add(blocRenduVisuel, BorderLayout.CENTER);
        conteneur.add(panneauCentral, BorderLayout.CENTER);

        JPanel blocTerminal = new JPanel(new BorderLayout(15, 15)); blocTerminal.setOpaque(false); blocTerminal.setBorder(new EmptyBorder(10, 0, 0, 0));
        DesignSystem.BoutonArrondi cmdInference = new DesignSystem.BoutonArrondi("Demander une prédiction (Inférence)...");
        cmdInference.setBackground(DesignSystem.THEME_TEXTE); cmdInference.setForeground(DesignSystem.THEME_FOND);
        cmdInference.setFont(new Font("SansSerif", Font.BOLD, 16)); cmdInference.setPreferredSize(new Dimension(0, 50)); cmdInference.setEnabled(false); 
        
        DesignSystem.PanneauArrondi afficheurTerminal = new DesignSystem.PanneauArrondi(new BorderLayout(), 20, DesignSystem.THEME_FOND.darker());
        afficheurTerminal.setBorder(new EmptyBorder(15, 20, 15, 20));
        JLabel journalExecution = new JLabel("> Console Système en attente d'instructions...", JLabel.LEFT);
        journalExecution.setFont(new Font("Monospaced", Font.BOLD, 14)); journalExecution.setForeground(DesignSystem.THEME_TEXTE);
        afficheurTerminal.add(journalExecution, BorderLayout.CENTER);

        blocTerminal.add(cmdInference, BorderLayout.NORTH); blocTerminal.add(afficheurTerminal, BorderLayout.SOUTH);
        conteneur.add(blocTerminal, BorderLayout.SOUTH);

        final java.awt.Image[] imageAffichee = new java.awt.Image[1];
        afficheurImage.addComponentListener(new ComponentAdapter() {
            @Override public void componentResized(ComponentEvent e) {
                if (imageAffichee[0] != null) {
                    afficheurImage.setIcon(adapterRenduImage(imageAffichee[0], afficheurImage.getWidth(), afficheurImage.getHeight()));
                }
            }
        });

        /* --- LOGIQUE ÉVÈNEMENTIELLE (CONTRÔLEURS) --- */

        cmdEntrainer.addActionListener(e -> {
            String modaliteSignal = (String) modSignal.getSelectedItem();
            String modaliteActiv = (String) modActivation.getSelectedItem();
            boolean utiliseShuffle = optMelange.isSelected(), utiliseNorm = optNorm.isSelected();
            boolean mMi = augMiroir.isSelected(), mRot = augRotation.isSelected(), mTr = augTrans.isSelected(), mZm = augZoom.isSelected(), mBr = augBruit.isSelected();
            
            float pEta = 0.0001f, pMse = 0.15f;
            try { 
                pEta = Float.parseFloat(valEta.getText()); 
                pMse = Float.parseFloat(valMse.getText()); 
            } catch (NumberFormatException ex) {
                journalExecution.setText("> Exception levée : Paramètres numériques invalides.");
                journalExecution.setForeground(DesignSystem.THEME_ERREUR); return;
            }

            journalExecution.setForeground(DesignSystem.THEME_ACCENT); journalExecution.setText("> Descente de gradient en cours. Processus alloué sur un thread secondaire...");
            cmdEntrainer.setEnabled(false); cmdInference.setEnabled(false); 

            final float fEta = pEta; final float fMse = pMse;

            // Déportation du calcul lourd pour préserver la réactivité de l'UI
            new Thread(() -> {
                try {
                    String resultat = executerPipelineEntrainement(modaliteSignal, modaliteActiv, utiliseShuffle, utiliseNorm, 
                            mMi, mRot, mTr, mZm, mBr, fEta, fMse, msg -> SwingUtilities.invokeLater(() -> journalExecution.setText(msg)));

                    SwingUtilities.invokeLater(() -> {
                        journalExecution.setText("> " + resultat); journalExecution.setForeground(DesignSystem.THEME_SUCCES);
                        cmdInference.setEnabled(true); cmdEntrainer.setEnabled(true);
                        imageAffichee[0] = null; afficheurImage.setIcon(null); afficheurImage.setText("Réseau compilé. Prêt pour l'inférence visuelle.");
                    });
                } catch (Exception ex) {
                    ex.printStackTrace();
                    SwingUtilities.invokeLater(() -> {
                        journalExecution.setText("> Exception Critique : Échec de la rétropropagation."); journalExecution.setForeground(DesignSystem.THEME_ERREUR); cmdEntrainer.setEnabled(true);
                    });
                }
            }).start();
        });

        cmdSauvegarder.addActionListener(e -> {
            if (expertChien == null) { journalExecution.setText("> Impossible d'exporter un modèle non initialisé."); journalExecution.setForeground(DesignSystem.THEME_ERREUR); return; }
            try {
                expertChien.sauvegarde("modele_chien.txt"); expertChat.sauvegarde("modele_chat.txt"); expertWild.sauvegarde("modele_wild.txt");
                journalExecution.setText("> Données synaptiques persistées avec succès."); journalExecution.setForeground(DesignSystem.THEME_SUCCES);
            } catch (Exception ex) {}
        });

        cmdCharger.addActionListener(e -> {
            try {
                if (tailleReferenceVector == -1) {
                    int compteLignes = 0; BufferedReader liseur = new BufferedReader(new FileReader(new File("modele_chien.txt")));
                    while (liseur.readLine() != null) compteLignes++; liseur.close();
                    tailleReferenceVector = compteLignes - 1; 
                }
                String activationVoulue = (String) modActivation.getSelectedItem();
                if (activationVoulue.equals("Sigmoïde")) { expertChien = new NeuroneSigmoide(tailleReferenceVector); expertChat = new NeuroneSigmoide(tailleReferenceVector); expertWild = new NeuroneSigmoide(tailleReferenceVector); }
                else if (activationVoulue.equals("ReLU")) { expertChien = new NeuroneReLU(tailleReferenceVector); expertChat = new NeuroneReLU(tailleReferenceVector); expertWild = new NeuroneReLU(tailleReferenceVector); }
                else { expertChien = new NeuroneHeavyside(tailleReferenceVector); expertChat = new NeuroneHeavyside(tailleReferenceVector); expertWild = new NeuroneHeavyside(tailleReferenceVector); }

                expertChien.chargement("modele_chien.txt"); expertChat.chargement("modele_chat.txt"); expertWild.chargement("modele_wild.txt");
                configurationCouranteSignal = (String) modSignal.getSelectedItem(); configurationCouranteNorm = optNorm.isSelected();
                
                journalExecution.setText("> Architecture restaurée à partir de la mémoire morte."); journalExecution.setForeground(DesignSystem.THEME_SUCCES);
                cmdInference.setEnabled(true); afficheurImage.setIcon(null); afficheurImage.setText("Modèle restauré.");
            } catch (Exception ex) { journalExecution.setText("> Erreur d'I/O lors de la restauration."); journalExecution.setForeground(DesignSystem.THEME_ERREUR); }
        });

        cmdInference.addActionListener(e -> {
            JFileChooser selecteurFichier = new JFileChooser("dataset_groupe_8/test/");
            if (selecteurFichier.showOpenDialog(conteneur) == JFileChooser.APPROVE_OPTION) {
                String cibleChemin = selecteurFichier.getSelectedFile().getAbsolutePath();
                imageAffichee[0] = new ImageIcon(cibleChemin).getImage();
                afficheurImage.setIcon(adapterRenduImage(imageAffichee[0], afficheurImage.getWidth(), afficheurImage.getHeight()));
                afficheurImage.setText(""); 

                boolean necessiteMonochrome = configurationCouranteSignal.equals("Niveaux de gris");
                Image echantillonDeTest = new Image(cibleChemin, -1, necessiteMonochrome);
                float[] vecteurEntree = extraireVecteur(echantillonDeTest, configurationCouranteSignal, configurationCouranteNorm);

                if (vecteurEntree.length != tailleReferenceVector) {
                    journalExecution.setText("> Disparité des tenseurs : L'échantillon ne correspond pas aux dimensions du réseau."); journalExecution.setForeground(DesignSystem.THEME_ERREUR); return;
                }

                expertChien.metAJour(vecteurEntree); expertChat.metAJour(vecteurEntree); expertWild.metAJour(vecteurEntree);
                float indiceChien = expertChien.sortie(), indiceChat = expertChat.sortie(), indiceWild = expertWild.sortie();
                float partitionGraphes = indiceChien + indiceChat + indiceWild; 
                if (partitionGraphes == 0) partitionGraphes = 1; 
                
                String diagnostic = "CHIEN"; float plusHautScore = indiceChien; Color indicateurCouleur = DesignSystem.THEME_SUCCES;
                if (indiceChat > plusHautScore) { diagnostic = "CHAT"; plusHautScore = indiceChat; indicateurCouleur = DesignSystem.THEME_ACCENT; } 
                if (indiceWild > plusHautScore) { diagnostic = "SAUVAGE"; plusHautScore = indiceWild; indicateurCouleur = DesignSystem.THEME_ERREUR; }

                journalExecution.setText(String.format("> Évaluation Heuristique : %s  (Certitude relative -> Chien : %.0f%% | Chat : %.0f%% | Sauvage : %.0f%%)", 
                    diagnostic, (indiceChien/partitionGraphes)*100, (indiceChat/partitionGraphes)*100, (indiceWild/partitionGraphes)*100));
                journalExecution.setForeground(indicateurCouleur);
            }
        });

        conteneur.setLocationRelativeTo(null);
        conteneur.setVisible(true);
    }
}
