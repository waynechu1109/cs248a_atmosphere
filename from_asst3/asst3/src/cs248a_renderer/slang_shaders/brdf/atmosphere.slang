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
        float rayleighWeight = 0.6;
        float mieWeight = 0.4;

        float rayleigh = getRayleighPhase(cosTheta);
        float mie = getMiePhase(cosTheta);

        return rayleighWeight * rayleigh + mieWeight * mie;
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
        float extinction = max(densityFalloff, 0.0);
        float opticalDepth = extinction * max(pathLength, 0.0);
        return exp(-absorptionColor * opticalDepth);
    }

    public float3 getViewTransmittance(float3 rayOut, float3 normal)
    {
        return getTransmittance(getPathLength(rayOut, normal));
    }

    public float3 evaluate(float3 rayOut, float3 normal, float3 lightDirection)
    {
        float3 wo = normalize(rayOut);
        float3 wi = normalize(lightDirection);

        float extinction = max(densityFalloff, 0.0);
        float viewPathLength = getPathLength(wo, normal);
        float lightPathLength = getPathLength(wi, normal);
        float viewOpticalDepth = extinction * viewPathLength;
        float lightOpticalDepth = extinction * lightPathLength;
        float combinedOpticalDepth = viewOpticalDepth + lightOpticalDepth;

        float phase = getPhaseValue(dot(wo, wi));
        float scatterAmount = 1.0 - exp(-combinedOpticalDepth);

        float3 lightTransmittance = exp(-absorptionColor * lightOpticalDepth);
        float3 inScatteredLight =
            scatteringColor * scatteringStrength * phase * scatterAmount * lightTransmittance;

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
