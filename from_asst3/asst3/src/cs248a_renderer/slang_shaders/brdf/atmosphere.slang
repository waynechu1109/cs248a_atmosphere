implementing "../brdf.slang";

import utils;

public struct AtmosphereBRDF
{
    public float3 surfaceAlbedo;
    public float3 scatteringColor;
    public float3 absorptionColor;
    public float densityFalloff;
    public float scatteringStrength;
    public float phaseG;
    public float planetRadius;
    public float atmosphereRadius;

    public float getShellThickness()
    {
        float shellThickness = max(atmosphereRadius - planetRadius, 0.0);
        return shellThickness;
    }

    public float getPhaseValue(float cosTheta)
    {
        float g = clamp(phaseG, -0.99, 0.99);
        float denom = pow(max(1.0 + g * g - 2.0 * g * cosTheta, 1e-4), 1.5);
        return (1.0 - g * g) / (4.0 * PI * denom);
    }

    public float3 evaluate(float3 rayOut, float3 normal)
    {
        float3 n = normalize(normal);
        float3 wo = normalize(rayOut);
        if (dot(n, wo) < 0.0)
        {
            n = -n;
        }

        float shellDensity = 1.0 - exp(-max(densityFalloff, 0.0) * max(getShellThickness(), 1e-4));
        float viewAltitude = 1.0 - clamp(dot(n, wo), 0.0, 1.0);
        float phase = getPhaseValue(clamp(dot(wo, n), -1.0, 1.0));
        float3 transmittance = exp(-absorptionColor * shellDensity);
        float3 scattering = scatteringColor * scatteringStrength * shellDensity;
        float horizonBoost = 0.15 + 0.85 * viewAltitude;

        return surfaceAlbedo * transmittance + scattering * horizonBoost * (0.5 + phase);
    }

    public float3 sample(float3 rayOut, float3 normal)
    {
        float3 n = normalize(normal);
        float3 wo = normalize(rayOut);
        if (dot(n, wo) < 0.0)
        {
            n = -n;
        }

        float forwardBias = clamp(0.5 * (phaseG + 1.0), 0.0, 1.0);
        float3 sampleDirection = normalize((1.0 - forwardBias) * n + forwardBias * wo);

        if (length(sampleDirection) < 1e-6)
        {
            return n;
        }

        return sampleDirection;
    }

    public float pdf(float3 rayIn, float3 rayOut, float3 normal)
    {
        return 1.0;
    }
}
