package src;

public class NeuroneReLu extends Neurone
{
    
    public NeuroneReLu(final int nbEntrees)
    {
        super(nbEntrees);
    }
 
    // Fonction d'activation ReLU : sortie = max(0, x)
    // ReLU(x) = x si x > 0, sinon 0
    @Override
    protected float activation(final float valeur)
    {
        return Math.max(0f, valeur);
    }
}
