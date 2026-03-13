implementing "../brdf.slang";

import utils;

public struct AtmosphereBRDF
{
    public float3 surfaceAlbedo;
    public float3 rayleighScatteringColor;
    public float3 mieScatteringColor;
    public float3 absorptionColor;
    public float rayleighDensityFalloff;
    public float mieDensityFalloff;
    public float rayleighStrength;
    public float mieStrength;
    public float phaseG;
    public float planetRadius;
    public float atmosphereRadius;

    public float getShellThickness()   // get the thickness of atmosphere
    {
        float shellThickness = max(atmosphereRadius - planetRadius, 0.0);
        return shellThickness;
    }

    public float getRayleighPhase(float cosTheta)
    {
        float c = clamp(cosTheta, -1.0, 1.0);
        return (3.0 / (16.0 * PI)) * (1.0 + c * c);
    }

    public float getMiePhase(float cosTheta)
    {
        float c = clamp(cosTheta, -1.0, 1.0);
        float g = clamp(phaseG, -0.999, 0.999);
        float gg = g * g;
        float denom = max(1.0 + gg - 2.0 * g * c, 1e-6);

        return (1.0 - gg) / (4.0 * PI * pow(denom, 1.5));
    }

    public float getPhaseValue(float cosTheta)   // compute the scattering direction distribution
    {
        float rayleigh = getRayleighPhase(cosTheta);
        float mie = getMiePhase(cosTheta);

        return rayleighStrength * rayleigh + mieStrength * mie;
    }

    public float3 getScatteringCoefficient(float cosTheta)
    {
        float rayleigh = getRayleighPhase(cosTheta);
        float mie = getMiePhase(cosTheta);
        return rayleighScatteringColor * rayleighStrength * rayleigh +
            mieScatteringColor * mieStrength * mie;
    }

    public float3 getScatteringCoefficient(float cosTheta, float rayleighDensity, float mieDensity)
    {
        float rayleigh = getRayleighPhase(cosTheta);
        float mie = getMiePhase(cosTheta);
        return rayleighScatteringColor * rayleighStrength * rayleigh * rayleighDensity +
            mieScatteringColor * mieStrength * mie * mieDensity;
    }

    public float getRayleighDensity(float normalizedHeight)
    {
        return exp(-normalizedHeight * max(rayleighDensityFalloff, 0.0));
    }

    public float getMieDensity(float normalizedHeight)
    {
        float baseDensity = exp(-normalizedHeight * max(mieDensityFalloff, 0.0));
        float groundWeight = pow(saturate(1.0 - normalizedHeight), 3.0);
        return baseDensity * groundWeight;
    }

    public float3 getExtinctionCoefficient(float rayleighDensity, float mieDensity)
    {
        float mediumDensity = max(rayleighDensity, mieDensity);
        return absorptionColor * mediumDensity +
            rayleighScatteringColor * rayleighStrength * rayleighDensity +
            mieScatteringColor * mieStrength * mieDensity;
    }

    public float getPathLength(float3 direction, float3 normal)
    {
        float3 n = normalize(normal);
        float3 w = normalize(direction);
        float cosTheta = max(abs(dot(n, w)), 1e-3);
        float shellThickness = max(getShellThickness(), 1e-4);
        return shellThickness / cosTheta;
    }

    public float3 getTransmittance(float pathLength)
    {
        float opticalDepth = max(pathLength, 0.0);
        return exp(-getExtinctionCoefficient(1.0, 1.0) * opticalDepth);
    }

    public float3 getViewTransmittance(float3 rayOut, float3 normal)
    {
        return getTransmittance(getPathLength(rayOut, normal));
    }

    public float3 evaluate(float3 rayOut, float3 normal, float3 lightDirection)
    {
        float3 wo = normalize(rayOut);
        float3 wi = normalize(lightDirection);

        float viewPathLength = getPathLength(wo, normal);
        float lightPathLength = getPathLength(wi, normal);
        float viewOpticalDepth = viewPathLength;
        float lightOpticalDepth = lightPathLength;
        float combinedOpticalDepth = viewOpticalDepth + lightOpticalDepth;

        float scatterAmount = 1.0 - exp(-combinedOpticalDepth);

        float3 lightTransmittance = exp(-getExtinctionCoefficient(1.0, 1.0) * lightOpticalDepth);
        float3 inScatteredLight = getScatteringCoefficient(dot(wo, wi), 1.0, 1.0) * scatterAmount * lightTransmittance;

        return inScatteredLight;
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
