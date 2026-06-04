package src;

import java.io.*;
import java.util.*;
import javax.imageio.*;
import java.awt.image.*;
import java.nio.file.*;
import java.util.stream.*;

public class Image
{
	static private int LabelChat = 0;
	static private int LabelChien = 1;
	static private int LabelWild = 2;
	static private int LabelInconnu = 3;
	private int label = -1;
	private int largeur = 0;
	private int hauteur = 0;
	private int[] donnees = null; // image applatie en concaténant les lignes les unes après les autres

	public int label() {return label;}
	public int largeur() {return largeur;}
	public int hauteur() {return hauteur;}
	public int taille() {return donnees.length;} // nombre de pixels: hauteur*largeur ou 3*hauteur*largeur pour une image RGB
	public int[] donnees() {return donnees;}

	public boolean estEnNiveauxDeGris() {return taille() == largeur() * hauteur();}

	public void afficheMetadonnees() {
		String type = estEnNiveauxDeGris() ? "grayscale" : " couleurs";
		System.out.printf("Image (%s): label=%d, largeur=%d, hauteur=%d, taille=%d\n",
			type, label(), largeur(), hauteur(), taille());
	}

	public Image(final String cheminImage, int label, boolean niveauxDeGris) {
		try {
			final BufferedImage img = ImageIO.read(new File(cheminImage));
			this.label = label;
			largeur = img.getWidth(null);
			hauteur = img.getHeight(null);
			final int taille = niveauxDeGris ? hauteur*largeur : 3*hauteur*largeur;
			donnees = new int[taille];
			for (int i = 0; i < hauteur; ++i) {
				for (int j = 0; j < largeur; ++j) {
					final long rgb = img.getRGB(j, i);
					final int r = (int)((rgb>>16)&255);	// Isoler la composante rouge
					final int g = (int)((rgb>>8)&255);	// Isoler la composante verte
					final int b = (int)((rgb)&255);		// Isoler la composante bleue
					final int index = i * largeur + j;
					if (niveauxDeGris) {
						final float gris = 0.2125f * r + 0.7154f * g + 0.0721f * b; // RGB -> niveaux de gris
						donnees[index] = (int) Math.max(0, Math.min(255, gris));
					}
					else {
						donnees[3*index+0] = r;
						donnees[3*index+1] = g;
						donnees[3*index+2] = b;
					}
				}
			}
		}
		catch (Exception e) {
			e.printStackTrace();
			System.err.printf("Image non trouvée ou non lisible: %s\n", cheminImage);
		}
	}

	public static List<String> listeFichiers(String repertoire) {
		List<String> cheminsFichiers = null;
		try {
			// La syntaxe qui suit enchaîne plusieurs méthodes d'affilée
			cheminsFichiers = Files.walk(Paths.get(repertoire))	// Récupère les chemins
				.filter(Files::isRegularFile)					// filtre uniquement les fichiers
				.map(Path::toAbsolutePath)						// convertit le chemin en chemin absolu
				.map(Path::toString)							// convertit le chemin en chaine de caractères
				.collect(Collectors.toList());					// crée une collection à partir de ces chaînes
		} catch (Exception e) {
			e.printStackTrace();
		}
		return cheminsFichiers;
	}

    // Conversion des pixels RGB en représentation TSL/HSL (Teinte, Saturation, Luminosité).
// L'image doit avoir été chargée EN COULEUR (niveauxDeGris = false).
// Renvoie un tableau déjà normalisé dans [0,1] (pas de /255 à refaire après).
    public float[] donneesHSL() {
        final int nbPixels = largeur * hauteur;
        final float[] hsl = new float[3 * nbPixels];
        for (int p = 0; p < nbPixels; p++) {
            final float r = donnees[3*p + 0] / 255.0f;
            final float g = donnees[3*p + 1] / 255.0f;
            final float b = donnees[3*p + 2] / 255.0f;

            final float max = Math.max(r, Math.max(g, b));
            final float min = Math.min(r, Math.min(g, b));
            final float l = (max + min) / 2.0f;            // Luminosité

            float h, s;
            if (max == min) {            // pixel gris : teinte/saturation indéfinies -> 0
                h = 0.0f;
                s = 0.0f;
            } else {
                final float d = max - min;
                s = (l > 0.5f) ? d / (2.0f - max - min) : d / (max + min);   // Saturation
                if (max == r)      h = (g - b) / d + (g < b ? 6.0f : 0.0f);
                else if (max == g) h = (b - r) / d + 2.0f;
                else               h = (r - g) / d + 4.0f;
                h /= 6.0f;               // Teinte ramenée dans [0,1]
            }
            hsl[3*p + 0] = h;
            hsl[3*p + 1] = s;
            hsl[3*p + 2] = l;
        }
        return hsl;
    }

    // --- Extension E10 : FFT 2D ---
// Renvoie le spectre d'amplitude (log-magnitude) normalisé dans [0,1], aplati en 1D.
// Travaille sur la luminance (niveaux de gris) de l'image.
    public float[] donneesFFT() {
        // 1) Matrice de luminance à partir des données stockées (gris OU RGB)
        final boolean gris1canal = estEnNiveauxDeGris();
        final double[][] gris = new double[hauteur][largeur];
        for (int i = 0; i < hauteur; i++)
            for (int j = 0; j < largeur; j++) {
                final int idx = i * largeur + j;
                if (gris1canal) gris[i][j] = donnees[idx];
                else gris[i][j] = 0.2125*donnees[3*idx] + 0.7154*donnees[3*idx+1] + 0.0721*donnees[3*idx+2];
            }

        // 2) Dimensions complétées aux puissances de 2 (padding par des zéros)
        final int H = puissanceDe2SupOuEgale(hauteur);
        final int W = puissanceDe2SupOuEgale(largeur);
        final double[][] re = new double[H][W];
        final double[][] im = new double[H][W]; // partie imaginaire à 0
        for (int i = 0; i < hauteur; i++)
            for (int j = 0; j < largeur; j++)
                re[i][j] = gris[i][j];

        // 3) FFT des lignes, puis des colonnes (FFT 2D séparable)
        for (int i = 0; i < H; i++) fft1D(re[i], im[i]);
        final double[] colR = new double[H], colI = new double[H];
        for (int j = 0; j < W; j++) {
            for (int i = 0; i < H; i++) { colR[i] = re[i][j]; colI[i] = im[i][j]; }
            fft1D(colR, colI);
            for (int i = 0; i < H; i++) { re[i][j] = colR[i]; im[i][j] = colI[i]; }
        }

        // 4) Spectre d'amplitude en échelle log + normalisation [0,1]
        final float[] spectre = new float[H * W];
        final double[] tmp = new double[H * W];
        double maxLog = 0.0;
        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++) {
                final double mag = Math.sqrt(re[i][j]*re[i][j] + im[i][j]*im[i][j]);
                final double v = Math.log(1.0 + mag); // compression : la composante continue est énorme
                tmp[i*W + j] = v;
                if (v > maxLog) maxLog = v;
            }
        if (maxLog == 0.0) maxLog = 1.0;
        for (int k = 0; k < tmp.length; k++) spectre[k] = (float)(tmp[k] / maxLog);
        return spectre;
    }

    // Plus petite puissance de 2 >= x
    private static int puissanceDe2SupOuEgale(int x) {
        int p = 1;
        while (p < x) p <<= 1;
        return p;
    }

    // FFT 1D en place (Cooley-Tukey radix-2). n = re.length DOIT être une puissance de 2.
    private static void fft1D(final double[] re, final double[] im) {
        final int n = re.length;
        // inversion de bits
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) {
                double t = re[i]; re[i] = re[j]; re[j] = t;
                t = im[i]; im[i] = im[j]; im[j] = t;
            }
        }
        // papillons
        for (int len = 2; len <= n; len <<= 1) {
            final double ang = -2.0 * Math.PI / len;
            final double wlenR = Math.cos(ang), wlenI = Math.sin(ang);
            for (int i = 0; i < n; i += len) {
                double wR = 1.0, wI = 0.0;
                for (int k = 0; k < len/2; k++) {
                    final int a = i + k, b = i + k + len/2;
                    final double vR = re[b]*wR - im[b]*wI;
                    final double vI = re[b]*wI + im[b]*wR;
                    re[b] = re[a] - vR; im[b] = im[a] - vI;
                    re[a] += vR;        im[a] += vI;
                    final double nwR = wR*wlenR - wI*wlenI;
                    wI = wR*wlenI + wI*wlenR; wR = nwR;
                }
            }
        }
    }
    // Extension E9 : retourne une COPIE miroir horizontal des données brutes.
// Gère gris (1 canal) et couleur (3 canaux). N'altère pas l'image d'origine.
    public float[] donneesMiroir() {
        final int canaux = estEnNiveauxDeGris() ? 1 : 3;
        final float[] m = new float[donnees.length];
        for (int i = 0; i < hauteur; i++)
            for (int j = 0; j < largeur; j++) {
                final int jm = largeur - 1 - j;          // colonne miroir
                for (int c = 0; c < canaux; c++)
                    m[(i*largeur + j)*canaux + c] = donnees[(i*largeur + jm)*canaux + c];
            }
        return m;
    }

    // ============================================================
// BLOC HOG + AUGMENTATION DE DONNÉES
// Remplace les anciennes donneesHOG / donneesHOGMiroir / donneesHOGRotation.
// (nécessite "import java.util.Random;" ou "import java.util.*;" en tête de fichier)
// ============================================================

    // ---- Luminance (grille hauteur x largeur), gère gris et couleur ----
    private double[][] luminance() {
        final int canaux = estEnNiveauxDeGris() ? 1 : 3;
        final double[][] g = new double[hauteur][largeur];
        for (int i = 0; i < hauteur; i++)
            for (int j = 0; j < largeur; j++) {
                final int idx = i * largeur + j;
                if (canaux == 1) g[i][j] = donnees[idx];
                else g[i][j] = 0.2125*donnees[3*idx] + 0.7154*donnees[3*idx+1] + 0.0721*donnees[3*idx+2];
            }
        return g;
    }

    // ---- Interpolation bilinéaire avec bords répliqués ----
    private static double bilineaire(double[][] g, double sx, double sy) {
        final int H = g.length, W = g[0].length;
        int x0 = (int)Math.floor(sx), y0 = (int)Math.floor(sy);
        final double fx = sx - x0, fy = sy - y0;
        int x1 = x0 + 1, y1 = y0 + 1;
        x0 = Math.max(0, Math.min(W-1, x0)); x1 = Math.max(0, Math.min(W-1, x1));
        y0 = Math.max(0, Math.min(H-1, y0)); y1 = Math.max(0, Math.min(H-1, y1));
        final double top = g[y0][x0] + (g[y0][x1] - g[y0][x0]) * fx;
        final double bot = g[y1][x0] + (g[y1][x1] - g[y1][x0]) * fx;
        return top + (bot - top) * fy;
    }

    // ---- TRANSFORMATIONS (chacune renvoie une nouvelle grille de luminance) ----
    private static double[][] miroirH(double[][] g) {
        final int H = g.length, W = g[0].length;
        final double[][] r = new double[H][W];
        for (int i = 0; i < H; i++) for (int j = 0; j < W; j++) r[i][j] = g[i][W-1-j];
        return r;
    }
    private static double[][] rotation(double[][] g, double angleDeg) {
        final int H = g.length, W = g[0].length;
        final double rad = Math.toRadians(angleDeg), cos = Math.cos(rad), sin = Math.sin(rad);
        final double cx = (W-1)/2.0, cy = (H-1)/2.0;
        final double[][] r = new double[H][W];
        for (int i = 0; i < H; i++) for (int j = 0; j < W; j++) {
            final double dx = j - cx, dy = i - cy;
            r[i][j] = bilineaire(g, cos*dx + sin*dy + cx, -sin*dx + cos*dy + cy);
        }
        return r;
    }
    private static double[][] translation(double[][] g, int dx, int dy) {
        final int H = g.length, W = g[0].length;
        final double[][] r = new double[H][W];
        for (int i = 0; i < H; i++) for (int j = 0; j < W; j++) {
            int si = Math.max(0, Math.min(H-1, i - dy));
            int sj = Math.max(0, Math.min(W-1, j - dx));
            r[i][j] = g[si][sj];
        }
        return r;
    }
    private static double[][] zoom(double[][] g, double facteur) {
        final int H = g.length, W = g[0].length;
        final double cx = (W-1)/2.0, cy = (H-1)/2.0;
        final double[][] r = new double[H][W];
        for (int i = 0; i < H; i++) for (int j = 0; j < W; j++)
            r[i][j] = bilineaire(g, cx + (j - cx)/facteur, cy + (i - cy)/facteur);
        return r;
    }
    private static double[][] bruit(double[][] g, double amplitude) {
        final int H = g.length, W = g[0].length;
        final Random rnd = new Random();
        final double[][] r = new double[H][W];
        for (int i = 0; i < H; i++) for (int j = 0; j < W; j++)
            r[i][j] = Math.max(0, Math.min(255, g[i][j] + rnd.nextGaussian()*amplitude));
        return r;
    }

    // ---- CALCUL HOG (sur n'importe quelle grille de luminance) ----
    private static float[] hog(double[][] g) {
        final int H = g.length, W = g[0].length;
        final int CELL = 8, BINS = 9;
        final double[][] mag = new double[H][W];
        final double[][] ori = new double[H][W];
        for (int i = 0; i < H; i++)
            for (int j = 0; j < W; j++) {
                double gx, gy;
                if (j == 0) gx = g[i][j+1]-g[i][j];
                else if (j == W-1) gx = g[i][j]-g[i][j-1];
                else gx = g[i][j+1]-g[i][j-1];
                if (i == 0) gy = g[i+1][j]-g[i][j];
                else if (i == H-1) gy = g[i][j]-g[i-1][j];
                else gy = g[i+1][j]-g[i-1][j];
                mag[i][j] = Math.sqrt(gx*gx + gy*gy);
                double a = Math.toDegrees(Math.atan2(gy, gx));
                if (a < 0) a += 180.0; if (a >= 180.0) a -= 180.0;
                ori[i][j] = a;
            }
        final int nCellY = H/CELL, nCellX = W/CELL;
        final float[] feats = new float[nCellY*nCellX*BINS];
        final double binW = 180.0/BINS; int f = 0;
        for (int cy = 0; cy < nCellY; cy++)
            for (int cx = 0; cx < nCellX; cx++) {
                final double[] h = new double[BINS];
                for (int di = 0; di < CELL; di++)
                    for (int dj = 0; dj < CELL; dj++) {
                        final int i = cy*CELL+di, j = cx*CELL+dj;
                        int bin = (int)(ori[i][j]/binW); if (bin >= BINS) bin = BINS-1;
                        h[bin] += mag[i][j];
                    }
                double norm = 0; for (double v : h) norm += v*v;
                norm = Math.sqrt(norm) + 1e-6;
                for (int b = 0; b < BINS; b++) feats[f++] = (float)(h[b]/norm);
            }
        return feats;
    }

    // ---- MÉTHODES PUBLIQUES : HOG de l'original et de chaque augmentation ----
    public float[] donneesHOG()                         { return hog(luminance()); }
    public float[] donneesHOGMiroir()                   { return hog(miroirH(luminance())); }
    public float[] donneesHOGRotation(double angle)     { return hog(rotation(luminance(), angle)); }
    public float[] donneesHOGTranslation(int dx, int dy){ return hog(translation(luminance(), dx, dy)); }
    public float[] donneesHOGZoom(double facteur)       { return hog(zoom(luminance(), facteur)); }
    public float[] donneesHOGBruit(double amplitude)    { return hog(bruit(luminance(), amplitude)); }
    // Combinaison (piste 2) : miroir PUIS rotation
    public float[] donneesHOGMiroirRotation(double angle){ return hog(rotation(miroirH(luminance()), angle)); }

	public static void main (String[] args)
	{
		List<String> cheminsFichiers = listeFichiers("dataset_animaux/");
		for (String chemin : cheminsFichiers) {
			System.out.println(chemin);
		}

		final String chemin = "dataset_animaux/train/dog/010552.jpg";
		final int labelImage = chemin.indexOf("dog") != -1 ? LabelChien : LabelInconnu;
		Image im1 = new Image(chemin, labelImage, false);
		Image im2 = new Image(chemin, labelImage, true);
		im1.afficheMetadonnees();
		im2.afficheMetadonnees();
	}
}
