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
	private float label = -1;
	private int largeur = 0;
	private int hauteur = 0;
	private float[] donnees = null; // image applatie en concaténant les lignes les unes après les autres

	public float label() {return label;}
	public int largeur() {return largeur;}
	public int hauteur() {return hauteur;}
	public int taille() {return donnees.length;} // nombre de pixels: hauteur*largeur ou 3*hauteur*largeur pour une image RGB
	public float[] donnees() {return donnees;}

	public boolean estEnNiveauxDeGris() {return taille() == largeur() * hauteur();}

	public void afficheMetadonnees() {
		String type = estEnNiveauxDeGris() ? "grayscale" : " couleurs";
		System.out.printf("Image (%s): label=%d, largeur=%d, hauteur=%d, taille=%d\n",
			type, label(), largeur(), hauteur(), taille());
	}

	public Image(final String cheminImage, float label, boolean niveauxDeGris) {
		try {
			final BufferedImage img = ImageIO.read(new File(cheminImage));
			this.label = (int) label;
			largeur = img.getWidth(null);
			hauteur = img.getHeight(null);
			final int taille = niveauxDeGris ? hauteur*largeur : 3*hauteur*largeur;
			donnees = new float[taille];
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

	public static float[] Labelliser(List <String> cheminsFichiers) {
		float[] labels = new float[cheminsFichiers.size()];
		for (int i = 0; i < cheminsFichiers.size(); i++) {
			String chemin = cheminsFichiers.get(i);
			labels[i] = chemin.indexOf("cat") != -1 ? LabelChat : 
					   chemin.indexOf("dog") != -1 ? LabelChien : 
					   chemin.indexOf("wild") != -1 ? LabelWild : LabelInconnu;
		}
		return labels;
	}

	public static void Normaliser(float[] donnees) {
		for (int i = 0; i < donnees.length; i++) {
			donnees[i]/= 255.0f;
		}
	}

	public static void Mélanger(List<String> chemingsFichiers, float[] labels){
		Random rand=new Random();
		for (int i = 0; i < chemingsFichiers.size(); i++) {
			int j = rand.nextInt(chemingsFichiers.size());
			
			String tempChemin = chemingsFichiers.get(i);
			chemingsFichiers.set(i, chemingsFichiers.get(j));
			chemingsFichiers.set(j, tempChemin);

			float tempLabel = labels[i];
			labels[i] = labels[j];
			labels[j] = tempLabel;
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

	public static void main (String[] args)
	{
		String chemin ="C:\\Users\\yahya\\Desktop\\Projet JAVA THS\\Projet-IA-Java-THS\\dataset_groupe_8\\train";
		List<String> cheminsFichiers = listeFichiers(chemin);
		float[] Resultats = Labelliser(cheminsFichiers);
		Mélanger(cheminsFichiers, Resultats);
		float[][] Entrées =new float[cheminsFichiers.size()][];
		for (int i=0;i<cheminsFichiers.size();i++){
			Entrées[i]=new Image(cheminsFichiers.get(i), Resultats[i], true).donnees();
			Normaliser(Entrées[i]);
			System.out.println(cheminsFichiers.get(i));
		}
		float MSElimite = 0.001f;
		final iNeurone n = new NeuroneHeavyside(Entrées[0].length);
		n.apprentissage(Entrées, Resultats, MSElimite);

		final Neurone vueNeurone = (Neurone)n;
		System.out.print("Synapses : ");
		for (final float f : vueNeurone.synapses())
			System.out.print(f+" ");
		System.out.print("\nBiais : ");
		System.out.println(vueNeurone.biais());
		
		// On affiche chaque cas appris
		for (int i = 0; i < Entrées.length; ++i)
		{
			// Pour une entrée donnée
			final float[] entree = Entrées[i];
			// On met à jour la sortie du neurone
			n.metAJour(entree);
			// On affiche cette sortie
			System.out.println("Entree "+i+" : "+n.sortie());
		}
	}
}
