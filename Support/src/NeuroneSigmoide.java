package src;

public class NeuroneSigmoide extends Neurone
{
	// Constructeur
	public NeuroneSigmoide(final int nbEntrees) {
		super(nbEntrees);
	}

	// Fonction d'activation Sigmoïde : 1 / (1 + e^(-x))
	@Override
	protected float activation(final float valeur) {
		return (float) (1.0 / (1.0 + Math.exp(-valeur)));
	}
}