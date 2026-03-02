implementing "../brdf.slang";

import utils;

public struct LambertianBRDF
{
    public float3 albedo;
    public bool useCosineWeightedSampling;

    public float3 evaluate()
    {
        float3 fr = this.albedo;

        fr = this.albedo * INV_PI;

        return fr;
    }

    public float3 sample(float3 rayOut, float3 normal, float2 randomEpsilon)
    {
        float3 sampleDirection = rayOut;
        float3x3 worldToLocalMatrix = getWorldToLocalMatrix(normal, rayOut);
        float3x3 localToWorldMatrix = transposeMatrix(worldToLocalMatrix);

        // HINT: First convert the reflected ray direction to the local space,
        // then compute the incident ray direction in the local space,
        // and finally convert it to the world space.

        if (this.useCosineWeightedSampling)
        {
            // TODO: Student implementation starts here.

            float3 n = normalize(normal);
            float3 wo = normalize(rayOut);
            if (dot(n, wo) < 0.0)
            {
                n = -n;
            }
            float3 helper = abs(n.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
            float3 t = normalize(cross(helper, n));
            float3 b = normalize(cross(n, t));

            float2 xi = randomEpsilon;
            float r = sqrt(xi.x);
            float phi = TWO_PI * xi.y;
            float x = r * cos(phi);
            float z = r * sin(phi);
            float y = sqrt(max(0.0, 1.0 - xi.x));
            sampleDirection = normalize(x * t + y * n + z * b);

            // TODO: Student implementation ends here.
        }
        else // uniform sampling, use uniform sampling by default
        {
            // TODO: Student implementation starts here.

            float3 n = normalize(normal);
            float3 wo = normalize(rayOut);
            if (dot(n, wo) < 0.0)
            {
                n = -n;
            }
            float3 helper = abs(n.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
            float3 t = normalize(cross(helper, n));
            float3 b = normalize(cross(n, t));

            float2 xi = randomEpsilon;
            float y = xi.x;
            float phi = TWO_PI * xi.y;
            float sinTheta = sqrt(max(0.0, 1.0 - y * y));
            float x = sinTheta * cos(phi);
            float z = sinTheta * sin(phi);
            sampleDirection = normalize(x * t + y * n + z * b);

            // TODO: Student implementation ends here.
        }

        return sampleDirection;
    }

    public float pdf(float3 rayIn, float3 rayOut, float3 normal)
    {
        float EPSILON = 1e-6;
        float pdf = all(abs(rayIn - rayOut) < EPSILON) ? 1.0 : 0.0;

        if (this.useCosineWeightedSampling)
        {
            // TODO: Student implementation starts here.

            // float cosTheta = max(dot(normalize(normal), normalize(rayIn)), 0.0);
            float3 n = normalize(normal);
            float3 wo = normalize(rayOut);
            if (dot(n, wo) < 0.0)
            {
                n = -n;
            }
            float cosTheta = max(dot(n, normalize(rayIn)), 0.0);
            pdf = cosTheta * INV_PI;

            // TODO: Student implementation ends here.
        }
        else // uniform sampling, use uniform sampling by default
        {
            // TODO: Student implementation starts here.

            // float cosTheta = dot(normalize(normal), normalize(rayIn));
            float3 n = normalize(normal);
            float3 wo = normalize(rayOut);
            if (dot(n, wo) < 0.0)
            {
                n = -n;
            }
            float cosTheta = dot(n, normalize(rayIn));
            pdf = cosTheta > 0.0 ? INV_TWO_PI : 0.0;

            // TODO: Student implementation ends here.
        }

        return pdf;
    }
}
