implementing "../brdf.slang";

public struct GlassBRDF
{
    public float3 albedo;
    public float ior;

    public float3 evaluate()
    {
        return albedo;
    }

    public float3 sample(float3 rayOut, float3 normal)
    {

        float3 sampleDirection = rayOut;
        float3x3 worldToLocalMatrix = getWorldToLocalMatrix(normal, rayOut);
        float3x3 localToWorldMatrix = transposeMatrix(worldToLocalMatrix);
        
        // TODO: Student implementation starts here.

        float3 rayOutLocal = normalize(mul(worldToLocalMatrix, rayOut));
        float3 nLocal = float3(0.0, 1.0, 0.0);

        bool entering = rayOutLocal.y > 0.0;
        float eta = entering ? (1.0 / ior) : ior;
        if (!entering)
        {
            nLocal = -nLocal;
        }

        float cosTheta1 = max(dot(rayOutLocal, nLocal), 0.0);
        float k = 1.0 - eta * eta * (1.0 - cosTheta1 * cosTheta1);

        float3 rayInLocal;
        if (k < 0.0)
        {
            // Total internal reflection: fallback to perfect mirror reflection.
            rayInLocal = float3(-rayOutLocal.x, rayOutLocal.y, -rayOutLocal.z);
        }
        else
        {
            rayInLocal = -eta * rayOutLocal + (eta * cosTheta1 - sqrt(k)) * nLocal;
        }

        sampleDirection = normalize(mul(localToWorldMatrix, rayInLocal));

        // TODO: Student implementation ends here.

        return sampleDirection;
    }

    public float pdf(float3 rayIn, float3 rayOut, float3 normal)
    {
        // For dielectric materials, the PDF is 1.0 for the single direction
        // (either refracted or reflected due to total internal reflection)
        return 1.0;
    }
}
