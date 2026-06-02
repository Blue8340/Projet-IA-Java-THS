import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.io.File;
import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.plaf.basic.BasicComboBoxUI;
import javax.swing.plaf.basic.BasicComboPopup;
import javax.swing.plaf.basic.ComboPopup;
import java.awt.*;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
import java.awt.geom.RoundRectangle2D;

public class MainProjet {

    private static final int LABEL_CHAT = 0;
    private static final int LABEL_CHIEN = 1;
    private static final int LABEL_WILD = 2;

    private static int tailleReference = -1;
    private static int largeurRef = -1;
    private static int hauteurRef = -1;

    private static iNeurone expertChien = null;
    private static iNeurone expertChat = null;
    private static iNeurone expertWild = null;

    private static String activeCouleur = "TSL";
    private static boolean activeNorm = true;

    // --- COULEURS DU THÈME GRAPHIQUE ---
    private static final Color BG_DARK = new Color(34, 56, 43);      // Fond principal
    private static final Color BG_PANEL = new Color(55, 80, 60);     // Arrière-plan des panneaux
    private static final Color FG_TEXT = new Color(240, 235, 225);   // Texte principal
    private static final Color ACCENT_BTN = new Color(210, 175, 130);// Accents et boutons d'action
    private static final Color TXT_BTN = new Color(60, 40, 20);      // Texte sur les boutons d'action
    private static final Color NEON_GREEN = new Color(150, 200, 120);// Succès / Validation
    private static final Color NEON_RED = new Color(200, 100, 80);   // Erreur / Avertissement

    static class DonneeEntrainement {
        float[] pixels;
        int label;
        DonneeEntrainement(float[] p, int l) {
            this.pixels = p;
            this.label = l;
        }
    }

    public static void main(String[] args) {
        System.out.println("=================================================");
        System.out.println("=== PLATEFORME DE CLASSIFICATION NEURONALE    ===");
        System.out.println("=================================================");
        SwingUtilities.invokeLater(() -> {
            creerEtAfficherInterface();
        });
    }

    // ========================================================
    // PIPELINE DYNAMIQUE D'ENTRAÎNEMENT (E1 -> E10)
    // ========================================================
    private static String executerPipelineEntrainement(String modeCouleur, String modeActivation, boolean activerShuffle, 
                                                       boolean normaliser, boolean miroir, float eta, float mse, 
                                                       java.util.function.Consumer<String> logger) {
        
        logger.accept("> Prétraitement et chargement des données (" + modeCouleur + ")...");
        
        String dossierTrain = "dataset_groupe_8/train/";
        List<String> chemins = Image.listeFichiers(dossierTrain);
        List<DonneeEntrainement> dataset = new ArrayList<>();

        if (chemins == null || chemins.isEmpty()) return "Erreur : Répertoire d'entraînement introuvable.";

        tailleReference = -1;
        boolean modeGrisDentre = modeCouleur.equals("Niveaux de gris") || modeCouleur.equals("FFT 2D");
        float divNorm = normaliser ? 255.0f : 1.0f; 
        
        for (String chemin : chemins) {
            int vraiLabel = -1;
            String c = chemin.toLowerCase();
            if (c.contains("/dog/") || c.contains("\\dog\\")) vraiLabel = LABEL_CHIEN;
            else if (c.contains("/cat/") || c.contains("\\cat\\")) vraiLabel = LABEL_CHAT;
            else if (c.contains("/wild/") || c.contains("\\wild\\")) vraiLabel = LABEL_WILD;

            if (vraiLabel != -1) {
                Image img = new Image(chemin, vraiLabel, modeGrisDentre);
                if (tailleReference == -1) {
                    tailleReference = img.taille(); largeurRef = img.largeur(); hauteurRef = img.hauteur();
                }
                
                if (img.taille() == tailleReference) {
                    int[] pixelsBruts = img.donnees();
                    float[] featuresImage = new float[tailleReference];
                    
                    if (modeCouleur.equals("FFT 2D")) {
                        featuresImage = calculerFFT2D(pixelsBruts, largeurRef, hauteurRef, normaliser); 
                    } 
                    else if (modeGrisDentre) {
                        for (int j = 0; j < tailleReference; j++) featuresImage[j] = pixelsBruts[j] / divNorm;
                    } else {
                        for (int y = 0; y < hauteurRef; y++) {
                            for (int x = 0; x < largeurRef; x++) {
                                int idx = (y * largeurRef + x) * 3;
                                int r = pixelsBruts[idx], g = pixelsBruts[idx + 1], b = pixelsBruts[idx + 2];
                                if (modeCouleur.equals("TSL")) {
                                    float[] tsl = rgbVersTsl(r, g, b); 
                                    featuresImage[idx] = tsl[0]; featuresImage[idx+1] = tsl[1]; featuresImage[idx+2] = tsl[2];
                                    if (!normaliser) { featuresImage[idx] *= 255; featuresImage[idx+1] *= 255; featuresImage[idx+2] *= 255; }
                                } else { 
                                    featuresImage[idx] = r/divNorm; featuresImage[idx+1] = g/divNorm; featuresImage[idx+2] = b/divNorm;
                                }
                            }
                        }
                    }

                    dataset.add(new DonneeEntrainement(featuresImage, vraiLabel));

                    // Augmentation de données (Symétrie horizontale)
                    if (miroir) {
                        float[] featuresMiroir = new float[tailleReference];
                        int channels = (modeGrisDentre && !modeCouleur.equals("FFT 2D")) ? 1 : (modeCouleur.equals("FFT 2D") ? 1 : 3);
                        for (int y = 0; y < hauteurRef; y++) {
                            for (int x = 0; x < largeurRef; x++) {
                                for (int ch = 0; ch < channels; ch++) {
                                    int idxOrig = (y * largeurRef + x) * channels + ch;
                                    int idxMir  = (y * largeurRef + (largeurRef - 1 - x)) * channels + ch;
                                    featuresMiroir[idxMir] = featuresImage[idxOrig];
                                }
                            }
                        }
                        dataset.add(new DonneeEntrainement(featuresMiroir, vraiLabel));
                    }
                }
            }
        }

        if (activerShuffle) Collections.shuffle(dataset); 

        int nbImages = dataset.size();
        float[][] entrees = new float[nbImages][tailleReference];
        float[][] objectifs = new float[3][nbImages];

        for (int i = 0; i < nbImages; i++) {
            entrees[i] = dataset.get(i).pixels;
            objectifs[0][i] = (dataset.get(i).label == LABEL_CHIEN) ? 1.0f : 0.0f;
            objectifs[1][i] = (dataset.get(i).label == LABEL_CHAT)  ? 1.0f : 0.0f;
            objectifs[2][i] = (dataset.get(i).label == LABEL_WILD)  ? 1.0f : 0.0f;
        }

        Neurone.fixeCoefApprentissage(eta); 

        if (modeActivation.equals("Sigmoïde")) {
            expertChien = new NeuroneSigmoide(tailleReference); expertChat = new NeuroneSigmoide(tailleReference); expertWild = new NeuroneSigmoide(tailleReference);
        } else if (modeActivation.equals("ReLU")) {
            expertChien = new NeuroneReLU(tailleReference); expertChat = new NeuroneReLU(tailleReference); expertWild = new NeuroneReLU(tailleReference);
        } else {
            expertChien = new NeuroneHeavyside(tailleReference); expertChat = new NeuroneHeavyside(tailleReference); expertWild = new NeuroneHeavyside(tailleReference);
        }

        logger.accept(String.format("> Entraînement de l'expert [CHIEN] (Échantillons : %d, Eta : %.3f)...", nbImages, eta));
        expertChien.apprentissage(entrees, objectifs[0], mse);
        
        logger.accept("> Entraînement de l'expert [CHAT]...");
        expertChat.apprentissage(entrees, objectifs[1], mse);
        
        logger.accept("> Entraînement de l'expert [WILD]...");
        expertWild.apprentissage(entrees, objectifs[2], mse);
        
        logger.accept("> Synchronisation terminée. Le modèle est opérationnel.");

        activeCouleur = modeCouleur;
        activeNorm = normaliser;

        return String.format("MODÈLE PRÊT. (Activation : %s | Signal : %s)", modeActivation, modeCouleur);
    }

    // ========================================================
    // TESTS EN CONSOLE (MÉTRIQUES)
    // ========================================================
    private static void testerLeModeleConsole(String dossier, String nomTest, String modeCouleur, boolean normaliser) {
        List<String> cheminsTest = Image.listeFichiers(dossier);
        int correct = 0, totalTest = 0;

        if (cheminsTest != null) {
            for (String chemin : cheminsTest) {
                int vraiLabel = -1;
                String c = chemin.toLowerCase();
                if (c.contains("/dog/") || c.contains("\\dog\\")) vraiLabel = LABEL_CHIEN;
                else if (c.contains("/cat/") || c.contains("\\cat\\")) vraiLabel = LABEL_CHAT;
                else if (c.contains("/wild/") || c.contains("\\wild\\")) vraiLabel = LABEL_WILD;

                if (vraiLabel != -1) {
                    boolean modeGris = modeCouleur.equals("Niveaux de gris") || modeCouleur.equals("FFT 2D");
                    Image imgTest = new Image(chemin, vraiLabel, modeGris);
                    if (imgTest.taille() != tailleReference) continue;

                    float[] entreesTest = new float[tailleReference];
                    int[] pixelsBruts = imgTest.donnees();
                    float divNorm = normaliser ? 255.0f : 1.0f;

                    if (modeCouleur.equals("FFT 2D")) {
                        entreesTest = calculerFFT2D(pixelsBruts, largeurRef, hauteurRef, normaliser);
                    }
                    else if (modeGris) {
                        for (int j = 0; j < tailleReference; j++) entreesTest[j] = pixelsBruts[j] / divNorm;
                    } else {
                        for (int y = 0; y < hauteurRef; y++) {
                            for (int x = 0; x < largeurRef; x++) {
                                int idx = (y * largeurRef + x) * 3;
                                if (modeCouleur.equals("TSL")) {
                                    float[] tsl = rgbVersTsl(pixelsBruts[idx], pixelsBruts[idx+1], pixelsBruts[idx+2]);
                                    entreesTest[idx] = tsl[0]; entreesTest[idx+1] = tsl[1]; entreesTest[idx+2] = tsl[2];
                                    if (!normaliser) { entreesTest[idx] *= 255; entreesTest[idx+1] *= 255; entreesTest[idx+2] *= 255; }
                                } else {
                                    entreesTest[idx] = pixelsBruts[idx]/divNorm; entreesTest[idx+1] = pixelsBruts[idx+1]/divNorm; entreesTest[idx+2] = pixelsBruts[idx+2]/divNorm;
                                }
                            }
                        }
                    }

                    expertChien.metAJour(entreesTest); expertChat.metAJour(entreesTest); expertWild.metAJour(entreesTest);

                    int prediction = LABEL_CHIEN; 
                    float maxScore = expertChien.sortie();
                    if (expertChat.sortie() > maxScore) { prediction = LABEL_CHAT; maxScore = expertChat.sortie(); }
                    if (expertWild.sortie() > maxScore) { prediction = LABEL_WILD; maxScore = expertWild.sortie(); }

                    if (prediction == vraiLabel) correct++;
                    totalTest++;
                }
            }
        }
        if (totalTest > 0) {
            System.out.printf("   [ÉVALUATION] Précision sur l'ensemble %s : %d/%d (%.2f%%)\n", nomTest, correct, totalTest, ((float)correct/totalTest)*100f);
        }
    }

    // ========================================================
    // OUTILS MATHÉMATIQUES (TSL & FFT 2D)
    // ========================================================
    public static float[] rgbVersTsl(int r, int g, int b) {
        float rf = r / 255.0f; float gf = g / 255.0f; float bf = b / 255.0f;
        float cmax = Math.max(rf, Math.max(gf, bf)); float cmin = Math.min(rf, Math.min(gf, bf));
        float delta = cmax - cmin; float l = (cmax + cmin) / 2.0f;
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

    private static void fft1D(double[] reel, double[] imag) {
        int n = reel.length;
        if (n <= 1) return;
        int decalage = 32 - Integer.numberOfLeadingZeros(n - 1);
        for (int i = 0; i < n; i++) {
            int j = Integer.reverse(i) >>> (32 - decalage);
            if (j > i) {
                double temp = reel[i]; reel[i] = reel[j]; reel[j] = temp;
                temp = imag[i]; imag[i] = imag[j]; imag[j] = temp;
            }
        }
        for (int taille = 2; taille <= n; taille *= 2) {
            int demiTaille = taille / 2;
            double angle = -2 * Math.PI / taille;
            double wReelBase = Math.cos(angle), wImagBase = Math.sin(angle);
            for (int i = 0; i < n; i += taille) {
                double wReel = 1, wImag = 0;
                for (int j = 0; j < demiTaille; j++) {
                    int k = i + j, l = i + j + demiTaille;
                    double tReel = wReel * reel[l] - wImag * imag[l];
                    double tImag = wReel * imag[l] + wImag * reel[l];
                    reel[l] = reel[k] - tReel; imag[l] = imag[k] - tImag;
                    reel[k] += tReel; imag[k] += tImag;
                    double prochainWReel = wReel * wReelBase - wImag * wImagBase;
                    wImag = wReel * wImagBase + wImag * wReelBase;
                    wReel = prochainWReel;
                }
            }
        }
    }

    public static float[] calculerFFT2D(int[] pixelsBruts, int largeur, int hauteur, boolean normaliser) {
        double[][] reel = new double[hauteur][largeur];
        double[][] imag = new double[hauteur][largeur];
        for(int y = 0; y < hauteur; y++){
            for(int x = 0; x < largeur; x++){
                reel[y][x] = pixelsBruts[y * largeur + x] / 255.0; imag[y][x] = 0;
            }
        }
        for(int y = 0; y < hauteur; y++) fft1D(reel[y], imag[y]);
        double[] colReel = new double[hauteur], colImag = new double[hauteur];
        for(int x = 0; x < largeur; x++){
            for(int y = 0; y < hauteur; y++) { colReel[y] = reel[y][x]; colImag[y] = imag[y][x]; }
            fft1D(colReel, colImag);
            for(int y = 0; y < hauteur; y++) { reel[y][x] = colReel[y]; imag[y][x] = colImag[y]; }
        }
        float[] magnitude = new float[largeur * hauteur];
        float maxMag = 0;
        for(int y = 0; y < hauteur; y++){
            for(int x = 0; x < largeur; x++){
                int shiftX = (x + largeur / 2) % largeur; int shiftY = (y + hauteur / 2) % hauteur;
                double magDouble = Math.sqrt(reel[y][x]*reel[y][x] + imag[y][x]*imag[y][x]);
                float val = (float) Math.log(1 + magDouble);
                magnitude[shiftY * largeur + shiftX] = val;
                if (val > maxMag) maxMag = val;
            }
        }
        if (normaliser && maxMag > 0) {
            for (int i = 0; i < magnitude.length; i++) magnitude[i] /= maxMag;
        }
        return magnitude;
    }

    // ========================================================
    // COMPOSANTS GRAPHIQUES PERSONNALISÉS (BORDS ARRONDIS)
    // ========================================================
    static class RoundedPanel extends JPanel {
        private Color backgroundColor;
        private int cornerRadius = 30;

        public RoundedPanel(LayoutManager layout, int radius, Color bgColor) {
            super(layout);
            this.cornerRadius = radius;
            this.backgroundColor = bgColor;
            setOpaque(false);
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g;
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setColor(backgroundColor);
            g2.fillRoundRect(0, 0, getWidth(), getHeight(), cornerRadius, cornerRadius);
        }
    }

    static class RoundedTextField extends JTextField {
        public RoundedTextField(String text, int cols) {
            super(text, cols);
            setOpaque(false);
            setBackground(BG_PANEL.brighter());
            setForeground(FG_TEXT);
            setCaretColor(ACCENT_BTN);
            setBorder(new EmptyBorder(5, 10, 5, 10));
        }
        @Override protected void paintComponent(Graphics g) {
            Graphics2D g2 = (Graphics2D) g;
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setColor(getBackground());
            g2.fillRoundRect(0, 0, getWidth(), getHeight(), 20, 20);
            super.paintComponent(g);
        }
    }

    static class RoundedButton extends JButton {
        public RoundedButton(String label) {
            super(label);
            setContentAreaFilled(false);
            setFocusPainted(false);
            setBorderPainted(false);
            setOpaque(false);
        }

        @Override
        protected void paintComponent(Graphics g) {
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            if (getModel().isArmed()) {
                g2.setColor(getBackground().darker());
            } else if (getModel().isRollover()) {
                g2.setColor(getBackground().brighter());
            } else {
                g2.setColor(getBackground());
            }
            g2.fillRoundRect(0, 0, getWidth(), getHeight(), 30, 30);
            
            FontMetrics metrics = g2.getFontMetrics(getFont());
            int x = (getWidth() - metrics.stringWidth(getText())) / 2;
            int y = ((getHeight() - metrics.getHeight()) / 2) + metrics.getAscent();
            g2.setColor(getForeground());
            g2.setFont(getFont());
            g2.drawString(getText(), x, y);
            
            g2.dispose();
        }
    }

    static class RoundedComboBox<E> extends JComboBox<E> {
        public RoundedComboBox(E[] items) {
            super(items);
            setOpaque(false);
            setBackground(BG_PANEL.brighter());
            setForeground(FG_TEXT);
            setUI(new BasicComboBoxUI() {
                @Override protected JButton createArrowButton() {
                    JButton btn = super.createArrowButton();
                    btn.setContentAreaFilled(false); btn.setBorder(null);
                    return btn;
                }
            });
        }
        @Override protected void paintComponent(Graphics g) {
            Graphics2D g2 = (Graphics2D) g;
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setColor(getBackground());
            g2.fillRoundRect(0, 0, getWidth(), getHeight(), 20, 20);
            super.paintComponent(g);
        }
    }

    private static void styleComponent(JComponent c) {
        c.setBackground(BG_PANEL.brighter());
        c.setForeground(FG_TEXT);
        c.setFont(new Font("SansSerif", Font.PLAIN, 14));
    }

    private static ImageIcon redimensionnerImageResponsive(java.awt.Image img, int maxWidth, int maxHeight) {
        if (img == null || maxWidth <= 0 || maxHeight <= 0) return null;
        int imgW = img.getWidth(null), imgH = img.getHeight(null);
        double ratio = Math.min((double) maxWidth / imgW, (double) maxHeight / imgH);
        return new ImageIcon(img.getScaledInstance((int) (imgW * ratio), (int) (imgH * ratio), java.awt.Image.SCALE_SMOOTH));
    }

    // ========================================================
    // INTERFACE GRAPHIQUE
    // ========================================================
    private static void creerEtAfficherInterface() {
        try { UIManager.setLookAndFeel(UIManager.getCrossPlatformLookAndFeelClassName()); } catch (Exception e) {}

        JFrame frame = new JFrame("IA Groupe 8 - Classification d'Images");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(950, 800);
        frame.setMinimumSize(new Dimension(850, 600));
        frame.getContentPane().setBackground(BG_DARK);
        frame.setLayout(new BorderLayout(20, 20));

        ((JPanel)frame.getContentPane()).setBorder(new EmptyBorder(20, 20, 20, 20));

        // --- EN TÊTE ---
        JLabel labelTitre = new JLabel("Module de Classification", JLabel.CENTER);
        labelTitre.setFont(new Font("SansSerif", Font.BOLD, 26));
        labelTitre.setForeground(ACCENT_BTN);
        frame.add(labelTitre, BorderLayout.NORTH);

        // --- ZONE CENTRALE ---
        JPanel panelCentre = new JPanel(new BorderLayout(15, 15));
        panelCentre.setOpaque(false);

        // Paramétrage
        RoundedPanel panelConfig = new RoundedPanel(new GridLayout(2, 1, 10, 10), 30, BG_PANEL);
        panelConfig.setBorder(new EmptyBorder(15, 15, 15, 15));

        JPanel row1 = new JPanel(new FlowLayout(FlowLayout.CENTER, 20, 5));
        row1.setOpaque(false);
        RoundedComboBox<String> comboCouleur = new RoundedComboBox<>(new String[]{"Niveaux de gris", "RGB", "TSL", "FFT 2D"});
        comboCouleur.setSelectedItem("TSL"); styleComponent(comboCouleur);
        RoundedComboBox<String> comboActivation = new RoundedComboBox<>(new String[]{"Sigmoïde", "ReLU", "Heaviside"});
        comboActivation.setSelectedItem("Sigmoïde"); styleComponent(comboActivation);
        
        JCheckBox checkShuffle = new JCheckBox("Mélange aléatoire", true); styleComponent(checkShuffle); checkShuffle.setOpaque(false);
        JCheckBox checkNorm = new JCheckBox("Normalisation", true); styleComponent(checkNorm); checkNorm.setOpaque(false);

        JLabel lblSignal = new JLabel("Signal:"); lblSignal.setForeground(FG_TEXT);
        JLabel lblAct = new JLabel("Activation:"); lblAct.setForeground(FG_TEXT);
        
        row1.add(lblSignal); row1.add(comboCouleur);
        row1.add(lblAct); row1.add(comboActivation);
        row1.add(checkShuffle); row1.add(checkNorm);

        JPanel row2 = new JPanel(new FlowLayout(FlowLayout.CENTER, 20, 5));
        row2.setOpaque(false);
        JCheckBox checkMiroir = new JCheckBox("Symétrie Miroir", false); styleComponent(checkMiroir); checkMiroir.setOpaque(false);
        
        RoundedTextField textEta = new RoundedTextField("0.01", 4); styleComponent(textEta); 
        RoundedTextField textMse = new RoundedTextField("0.15", 4); styleComponent(textMse); 
        
        RoundedButton boutonEntrainer = new RoundedButton("Lancer l'apprentissage");
        boutonEntrainer.setBackground(ACCENT_BTN);
        boutonEntrainer.setForeground(TXT_BTN);
        boutonEntrainer.setFont(new Font("SansSerif", Font.BOLD, 15));
        boutonEntrainer.setPreferredSize(new Dimension(280, 40));

        JLabel lblEta = new JLabel("Taux (Eta):"); lblEta.setForeground(FG_TEXT);
        JLabel lblMse = new JLabel("Marge d'erreur:"); lblMse.setForeground(FG_TEXT);

        row2.add(checkMiroir);
        row2.add(lblEta); row2.add(textEta);
        row2.add(lblMse); row2.add(textMse);
        row2.add(boutonEntrainer);

        panelConfig.add(row1); panelConfig.add(row2);
        panelCentre.add(panelConfig, BorderLayout.NORTH);

        // Affichage de l'image
        RoundedPanel panelImageFond = new RoundedPanel(new BorderLayout(), 30, BG_PANEL);
        panelImageFond.setBorder(new EmptyBorder(20, 20, 20, 20));
        
        JLabel labelImage = new JLabel("Aucun modèle chargé. Veuillez lancer l'apprentissage.", JLabel.CENTER);
        labelImage.setFont(new Font("SansSerif", Font.ITALIC, 16));
        labelImage.setForeground(FG_TEXT.darker());
        panelImageFond.add(labelImage, BorderLayout.CENTER);
        panelCentre.add(panelImageFond, BorderLayout.CENTER);

        frame.add(panelCentre, BorderLayout.CENTER);

        // --- ZONE INFÉRIEURE (ACTIONS ET STATUT) ---
        JPanel panelBas = new JPanel(new BorderLayout(15, 15));
        panelBas.setOpaque(false);
        panelBas.setBorder(new EmptyBorder(10, 0, 0, 0));

        RoundedButton boutonOuvrir = new RoundedButton("Sélectionner une image pour l'inférence...");
        boutonOuvrir.setBackground(FG_TEXT);
        boutonOuvrir.setForeground(BG_DARK);
        boutonOuvrir.setFont(new Font("SansSerif", Font.BOLD, 16));
        boutonOuvrir.setPreferredSize(new Dimension(0, 50));
        boutonOuvrir.setEnabled(false); 
        
        RoundedPanel panelResultat = new RoundedPanel(new BorderLayout(), 20, BG_DARK.darker());
        panelResultat.setBorder(new EmptyBorder(15, 20, 15, 20));
        
        JLabel labelResultat = new JLabel("> Système en attente d'initialisation...", JLabel.LEFT);
        labelResultat.setFont(new Font("SansSerif", Font.BOLD, 15));
        labelResultat.setForeground(FG_TEXT);
        panelResultat.add(labelResultat, BorderLayout.CENTER);

        panelBas.add(boutonOuvrir, BorderLayout.NORTH);
        panelBas.add(panelResultat, BorderLayout.SOUTH);
        frame.add(panelBas, BorderLayout.SOUTH);

        // Rendu responsive de l'image
        final java.awt.Image[] imageOriginaleMemoire = new java.awt.Image[1];
        labelImage.addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                if (imageOriginaleMemoire[0] != null) {
                    int maxW = labelImage.getWidth(); int maxH = labelImage.getHeight();
                    if (maxW > 0 && maxH > 0) labelImage.setIcon(redimensionnerImageResponsive(imageOriginaleMemoire[0], maxW, maxH));
                }
            }
        });

        // ACTION : LANCER L'APPRENTISSAGE
        boutonEntrainer.addActionListener(e -> {
            String couleurChoisie = (String) comboCouleur.getSelectedItem();
            String activationChoisie = (String) comboActivation.getSelectedItem();
            boolean shuffleActif = checkShuffle.isSelected();
            boolean normActif = checkNorm.isSelected();
            boolean miroirActif = checkMiroir.isSelected();
            
            float eta = 0.01f, mse = 0.15f;
            try { eta = Float.parseFloat(textEta.getText()); mse = Float.parseFloat(textMse.getText()); } 
            catch (Exception ex) { textEta.setText("0.01"); textMse.setText("0.15"); }

            labelResultat.setForeground(ACCENT_BTN);
            labelResultat.setText("> Optimisation des poids synaptiques en cours...");
            boutonEntrainer.setEnabled(false);
            boutonOuvrir.setEnabled(false); 

            final float finalEta = eta;
            final float finalMse = mse;

            new Thread(() -> {
                try {
                    String messageResultat = executerPipelineEntrainement(couleurChoisie, activationChoisie, shuffleActif, normActif, miroirActif, finalEta, finalMse, message -> {
                        SwingUtilities.invokeLater(() -> labelResultat.setText(message)); 
                    });

                    testerLeModeleConsole("dataset_groupe_8/train/", "TRAIN", couleurChoisie, normActif);
                    testerLeModeleConsole("dataset_groupe_8/test/", "TEST", couleurChoisie, normActif);

                    SwingUtilities.invokeLater(() -> {
                        labelResultat.setText("> " + messageResultat);
                        labelResultat.setForeground(NEON_GREEN);
                        boutonOuvrir.setEnabled(true); 
                        boutonEntrainer.setEnabled(true);
                        imageOriginaleMemoire[0] = null;
                        labelImage.setIcon(null);
                        labelImage.setText("Modèle entraîné avec succès. En attente d'une image.");
                    });
                } catch (Exception ex) {
                    SwingUtilities.invokeLater(() -> {
                        labelResultat.setText("> Erreur critique lors de l'apprentissage.");
                        labelResultat.setForeground(NEON_RED);
                        boutonEntrainer.setEnabled(true);
                    });
                }
            }).start();
        });

        // ACTION : INFÉRENCE SUR IMAGE
        boutonOuvrir.addActionListener(e -> {
            JFileChooser selecteur = new JFileChooser("dataset_groupe_8/test/");
            if (selecteur.showOpenDialog(frame) == JFileChooser.APPROVE_OPTION) {
                String chemin = selecteur.getSelectedFile().getAbsolutePath();
                
                imageOriginaleMemoire[0] = new ImageIcon(chemin).getImage();
                labelImage.setIcon(redimensionnerImageResponsive(imageOriginaleMemoire[0], labelImage.getWidth(), labelImage.getHeight()));
                labelImage.setText(""); 

                boolean modeGris = activeCouleur.equals("Niveaux de gris") || activeCouleur.equals("FFT 2D");
                Image imgIA = new Image(chemin, -1, modeGris);

                if (imgIA.taille() != tailleReference) {
                    labelResultat.setText("> Erreur : Dimensions de l'image incompatibles.");
                    labelResultat.setForeground(NEON_RED);
                    return;
                }

                float[] entreesTest = new float[tailleReference];
                int[] pixelsBruts = imgIA.donnees();
                float divNorm = activeNorm ? 255.0f : 1.0f;

                if (activeCouleur.equals("FFT 2D")) {
                    entreesTest = calculerFFT2D(pixelsBruts, largeurRef, hauteurRef, activeNorm);
                }
                else if (modeGris) {
                    for (int j = 0; j < tailleReference; j++) entreesTest[j] = pixelsBruts[j] / divNorm;
                } else {
                    for (int y = 0; y < hauteurRef; y++) {
                        for (int x = 0; x < largeurRef; x++) {
                            int idx = (y * largeurRef + x) * 3;
                            if (activeCouleur.equals("TSL")) {
                                float[] tsl = rgbVersTsl(pixelsBruts[idx], pixelsBruts[idx+1], pixelsBruts[idx+2]);
                                entreesTest[idx] = tsl[0]; entreesTest[idx+1] = tsl[1]; entreesTest[idx+2] = tsl[2];
                                if (!activeNorm) { entreesTest[idx] *= 255; entreesTest[idx+1] *= 255; entreesTest[idx+2] *= 255; }
                            } else {
                                entreesTest[idx] = pixelsBruts[idx]/divNorm; entreesTest[idx+1] = pixelsBruts[idx+1]/divNorm; entreesTest[idx+2] = pixelsBruts[idx+2]/divNorm;
                            }
                        }
                    }
                }

                expertChien.metAJour(entreesTest); expertChat.metAJour(entreesTest); expertWild.metAJour(entreesTest);
                
                float sChien = expertChien.sortie(), sChat = expertChat.sortie(), sWild = expertWild.sortie();
                float sommeScores = sChien + sChat + sWild;
                if (sommeScores == 0) sommeScores = 1; 
                
                float pChien = (sChien / sommeScores) * 100f;
                float pChat  = (sChat / sommeScores) * 100f;
                float pWild  = (sWild / sommeScores) * 100f;

                String verdict = "CHIEN"; float maxScore = sChien;
                Color textColor = NEON_GREEN;

                if (sChat > maxScore) { verdict = "CHAT"; maxScore = sChat; textColor = ACCENT_BTN; } 
                if (sWild > maxScore) { verdict = "SAUVAGE"; maxScore = sWild; textColor = NEON_RED; }

                labelResultat.setText(String.format("> Classe Prédite : %s  (Confiance : Chien %.0f%% | Chat %.0f%% | Sauvage %.0f%%)", verdict, pChien, pChat, pWild));
                labelResultat.setForeground(textColor);
            }
        });

        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
}
