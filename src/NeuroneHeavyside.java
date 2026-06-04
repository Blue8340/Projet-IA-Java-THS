public class NeuroneHeavyside extends Neurone
{
	// Fonction d'activation Heaviside : sortie = {0 si x<0, 1 si x>=0}
	@Override
	protected float activation(final float valeur) {return valeur >= 0 ? 1.f : 0.f;}
	
	// Constructeur
	public NeuroneHeavyside(final int nbEntrees) {super(nbEntrees);}
}
