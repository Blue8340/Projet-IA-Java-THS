

import java.util.Random;

public class TraitementHOG {

    private static double[][] luminance(Image img) {
        int hauteur = img.hauteur();
        int largeur = img.largeur();
        int[] donnees = img.donnees();
        boolean estGris = img.estEnNiveauxDeGris();
        
        double[][] g = new double[hauteur][largeur];
        for (int i = 0; i < hauteur; i++) {
            for (int j = 0; j < largeur; j++) {
                int idx = i * largeur + j;
                if (estGris) {
                    g[i][j] = donnees[idx];
                } else {
                    g[i][j] = 0.2125 * donnees[3 * idx] + 0.7154 * donnees[3 * idx + 1] + 0.0721 * donnees[3 * idx + 2];
                }
            }
        }
        return g;
    }

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

    public static float[] donneesHOG(Image img)                         { return hog(luminance(img)); }
    public static float[] donneesHOGMiroir(Image img)                   { return hog(miroirH(luminance(img))); }
    public static float[] donneesHOGRotation(Image img, double angle)   { return hog(rotation(luminance(img), angle)); }
    public static float[] donneesHOGTranslation(Image img, int dx, int dy){ return hog(translation(luminance(img), dx, dy)); }
    public static float[] donneesHOGZoom(Image img, double facteur)     { return hog(zoom(luminance(img), facteur)); }
    public static float[] donneesHOGBruit(Image img, double amplitude)  { return hog(bruit(luminance(img), amplitude)); }
}
