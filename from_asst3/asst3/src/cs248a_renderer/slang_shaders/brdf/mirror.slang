implementing "../brdf.slang";

public struct MirrorBRDF
{
    public float3 albedo;

    public float3 evaluate()
    {
        return albedo;
    }

    public float3 sample(float3 rayOut, float3 normal)
    {
        float3 sampleDirection = rayOut;
        float3x3 worldToLocalMatrix = getWorldToLocalMatrix(normal, rayOut);
        float3x3 localToWorldMatrix = transposeMatrix(worldToLocalMatrix);

        // HINT: First convert the reflected ray direction to the local space,
        // then compute the incident ray direction in the local space,
        // and finally convert it to the world space.

        // TODO: Student implementation starts here.

        float3 n = normalize(normal);
        float3 wo = normalize(rayOut);
        if (dot(wo, n) < 0.0)
        {
            n = -n;
        }
        sampleDirection = normalize(-wo + 2.0 * dot(wo, n) * n);

        // TODO: Student implementation ends here.

        return sampleDirection;
    }

    public float pdf(float3 rayIn, float3 rayOut, float3 normal)
    {
        const float EPSILON = 1e-6;
        float pdf = all(abs(rayIn - rayOut) < EPSILON) ? 1.0 : 0.0;
        float3x3 worldToLocalMatrix = getWorldToLocalMatrix(normal, rayOut);

        // TODO: Student implementation starts here.

        float3 n = normalize(normal);
        float3 wo = normalize(rayOut);
        if (dot(wo, n) < 0.0)
        {
            n = -n;
        }
        float3 expectedRayIn = normalize(-wo + 2.0 * dot(wo, n) * n);
        pdf = all(abs(normalize(rayIn) - expectedRayIn) < EPSILON) ? 1.0 : 0.0;

        // TODO: Student implementation ends here.

        return pdf;
    }
}
