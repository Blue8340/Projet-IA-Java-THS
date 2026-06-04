public class NeuroneReLU extends Neurone
{
    // Fonction d'activation ReLU : sortie = max(0, x)
    @Override
    protected float activation(final float valeur)
    {
        return Math.max(0f, valeur);
    }

    //Constructeur
    public NeuroneReLU(final int nbEntrees)
    {
        super(nbEntrees);
    }

}
