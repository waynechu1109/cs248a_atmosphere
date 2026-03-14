implementing "../renderer.slang";

import brdf;
import light;

public struct SampleBRDFResult
{
    public float3 fr;
    public float3 sampledDirection;
    public float sampledPdf;
}

public struct AtmosphereMarchResult
{
    public float3 inscatteredRadiance;
    public float3 transmittance;
}

public struct FilteredLightResult
{
    public float3 transmittance;
    public bool occluded;
}

public float3 getSampleIrradiance(LightSample lightSample, float3 normal, bool isLambertian, float fallOffFactor = 1.0)
{
    const float EPSILON = 1e-2;
    float3 lightDirection = lightSample.lightDirection;
    if (isLambertian)
    {
        float cosTheta = max(dot(normalize(normal), normalize(lightDirection)), 0.0);
        return lightSample.radiance * cosTheta / (fallOffFactor + EPSILON);
    }
    else
    {
        return lightSample.radiance / (fallOffFactor + EPSILON);
    }
}

public Ray getShadowRay(LightSample lightSample, float3 hitPoint)
{
    const float EPSILON = 1e-2;
    Ray shadowRay = Ray(float3(0.0), float3(0.0), float2(EPSILON, lightSample.lightDistance - EPSILON));

    // TODO: Student implementation starts here.

    shadowRay.direction = normalize(lightSample.lightDirection);
    shadowRay.origin = hitPoint + shadowRay.direction * EPSILON;

    // TODO: Student implementation ends here.

    return shadowRay;
}

public bool isLightOccluded(LightSample lightSample, float3 hitPoint, RendererUniform uniforms)
{
    bool occluded = false;

    // TODO: Student implementation starts here.

    // Step 1: Generate the shadow ray
    Ray shadowRay = getShadowRay(lightSample, hitPoint);

    // Step 2: Check if the shadow ray intersects with any rectangular light or mesh.
    // HINT: Check the modified signature of `rayMeshIntersection` function in ray_mesh_intersection.slang.
    // Can you modify it to make it faster for this particular use case?
    // Also, make use of `rayRectangularLightIntersection` function. Check its usage in `mesh_renderer.slang`.
    Optional<RectangularLight.LightHitResult> lightHit = rayRectangularLightIntersection(shadowRay, uniforms);
    Optional<RayMeshIntersectionResult> meshHit = rayMeshIntersection(shadowRay, uniforms, true);

    // if there is mesh and nearer than the light -> occluded
    if (meshHit.hasValue)
    {
        if (!lightHit.hasValue || meshHit.value.closestHit.t < lightHit.value.t)
        {
            occluded = true;
        }
    }

    // TODO: Student implementation ends here.

    return occluded;
}

public float3 evaluateFlashLightContribution(float3 hitPoint, float3 fr, float3 normal, RendererUniform uniforms, float3 rayOut, bool lambertianScattering = false)
{
    float3 outgoingRadiance = float3(0.0);
    float fallOffFactor = 1.0;  // Never falls off with distance

    // We have only one flash light source, so we can simply sample it.
    LightSample lightSample = FlashLight().sample(rayOut);

    // Unlike other light sources, a flash light is never occluded because it
    // always points at the closest hit from camera, and returning back to it.
    // Therefore, ideally we would not need to check for occlusion.
    // Yet, we do it anyway to be consistent with the function bodies of other light sources.
    bool occluded = isLightOccluded(lightSample, hitPoint, uniforms);
    if (occluded)
    {
        return outgoingRadiance;
    }
    FilteredLightResult filteredLight = filterLightThroughAtmospheres(
        hitPoint,
        lightSample.lightDirection,
        lightSample.lightDistance,
        uniforms
    );
    if (filteredLight.occluded)
    {
        return outgoingRadiance;
    }

    float3 sampleIrradiance = getSampleIrradiance(lightSample, normal, lambertianScattering, fallOffFactor);
    // PDF is used for Monte-Carlo integration. But it is only needed for light 
    // sources that have an area, for eg. rectangular light. Therefore, we are 
    // using `lightSample.pdf=1` here.
    float3 irradiance = (sampleIrradiance / lightSample.pdf) * filteredLight.transmittance;
    outgoingRadiance = fr * irradiance;

    return outgoingRadiance;
}

public float3 evaluatePointLightContribution(StructuredBuffer<PointLight> pointLightBuf, uint pointLightCount, float3 hitPoint, float3 fr, float3 normal, RendererUniform uniforms, bool lambertianScattering = false)
{
    float3 outgoingRadiance = float3(0.0);
    float fallOffFactor = 1.0;  // Point light falls off with distance. Calculate it below.
    for (int i = 0; i < pointLightCount; i++)
    {
        LightSample lightSample = pointLightBuf[i].sample(hitPoint);
        bool occluded = isLightOccluded(lightSample, hitPoint, uniforms);
        if (occluded)
        {
            continue;
        }
        FilteredLightResult filteredLight = filterLightThroughAtmospheres(
            hitPoint,
            lightSample.lightDirection,
            lightSample.lightDistance,
            uniforms
        );
        if (filteredLight.occluded)
        {
            continue;
        }

        // TODO: Student implementation starts here.

        // Calculate the fall off factor for point light.
        fallOffFactor = lightSample.lightDistance * lightSample.lightDistance;

        // TODO: Student implementation ends here.

        float3 sampleIrradiance = getSampleIrradiance(lightSample, normal, lambertianScattering, fallOffFactor);
        // PDF is used for Monte-Carlo integration. But it is only needed for light
        // sources that have an area, for eg. rectangular light. Therefore, we are
        // using `lightSample.pdf=1` here.
        float3 irradiance = (sampleIrradiance / lightSample.pdf) * filteredLight.transmittance;
        outgoingRadiance += fr * irradiance;
    }
    return outgoingRadiance;
}

public float3 evaluateDirectionalLightContribution(StructuredBuffer<DirectionalLight> directionalLightBuf, uint directionalLightCount, float3 hitPoint, float3 fr, float3 normal, RendererUniform uniforms, bool lambertianScattering = false)
{
    float3 outgoingRadiance = float3(0.0);
    float fallOffFactor = 1.0; // Never falls off with distance
    for (int i = 0; i < directionalLightCount; i++)
    {
        LightSample lightSample = directionalLightBuf[i].sample();
        bool occluded = isLightOccluded(lightSample, hitPoint, uniforms);
        if (occluded)
        {
            continue;
        }
        FilteredLightResult filteredLight = filterLightThroughAtmospheres(
            hitPoint,
            lightSample.lightDirection,
            lightSample.lightDistance,
            uniforms
        );
        if (filteredLight.occluded)
        {
            continue;
        }
        float3 sampleIrradiance = getSampleIrradiance(lightSample, normal, lambertianScattering, fallOffFactor);
        // PDF is used for Monte-Carlo integration. But it is only needed for light
        // sources that have an area, for eg. rectangular light. Therefore, we are
        // using `lightSample.pdf=1` here.
        float3 irradiance = (sampleIrradiance / lightSample.pdf) * filteredLight.transmittance;
        outgoingRadiance += fr * irradiance;
    }
    return outgoingRadiance;
}

public float3 evaluateRectangularLightContribution(StructuredBuffer<RectangularLight> rectangularLightBuf, uint rectangularLightCount, float3 hitPoint, float3 fr, float3 normal, RendererUniform uniforms, inout RNG rng, bool lambertianScattering = false)
{
    float3 outgoingRadiance = float3(0.0);
    float fallOffFactor = 1.0;  // Never falls off with distance
    for (int i = 0; i < rectangularLightCount; i++)
    {
        // Monte Carlo Estimator of the irradiance from the rectangular light source.
        // It is integral of the BRDF over the rectangular light source.
        float3 accumulatedIrradiance = float3(0.0);
        for (int j = 0; j < uniforms.numRectangularLightSamples; j++)
        {
            float2 randomEpsilon = rng.next_2d();
            LightSample lightSample = rectangularLightBuf[i].sample(hitPoint, randomEpsilon);
            bool occluded = isLightOccluded(lightSample, hitPoint, uniforms);
            if (occluded)
            {
                continue;
            }
            FilteredLightResult filteredLight = filterLightThroughAtmospheres(
                hitPoint,
                lightSample.lightDirection,
                lightSample.lightDistance,
                uniforms
            );
            if (filteredLight.occluded)
            {
                continue;
            }
            float3 sampleIrradiance = getSampleIrradiance(lightSample, normal, lambertianScattering, fallOffFactor);
            accumulatedIrradiance += (sampleIrradiance / lightSample.pdf) * filteredLight.transmittance;
        }
        float3 irradiance = accumulatedIrradiance / uniforms.numRectangularLightSamples;
        // Monte-Carlo Integration over.

        outgoingRadiance += fr * irradiance;
    }
    return outgoingRadiance;
}

//
public float2 intersectSphere(float3 rayOrigin, float3 rayDirection, float3 sphereCenter, float sphereRadius)
{
    // compute the intersection(s) between ray and sphere
    float3 oc = rayOrigin - sphereCenter;
    float a = dot(rayDirection, rayDirection);
    float b = 2.0 * dot(oc, rayDirection);
    float c = dot(oc, oc) - sphereRadius * sphereRadius;
    float discriminant = b * b - 4.0 * a * c;   // check the number of solutions
    if (discriminant < 0.0)
    {
        // no solution
        return float2(float.maxValue, -float.maxValue);
    }

    float sqrtDiscriminant = sqrt(discriminant);
    float invDenom = 0.5 / max(a, 1e-6);
    float t0 = (-b - sqrtDiscriminant) * invDenom;
    float t1 = (-b + sqrtDiscriminant) * invDenom;
    return float2(min(t0, t1), max(t0, t1));
}

public float2 densityAtPoint(AtmosphereBRDF brdf, float3 sphereCenter, float3 samplePoint)
{
    float shellThickness = max(brdf.getShellThickness(), 1e-4);
    float radius = length(samplePoint - sphereCenter);   // sphereCenter: center of the planet
    if (radius <= brdf.planetRadius || radius >= brdf.atmosphereRadius)
    {
        // make sure in the atmosphere
        return float2(0.0);
    }

    // transer the height form the ground to [0, 1]
    float normalizedHeight = saturate((radius - brdf.planetRadius) / shellThickness);
    return float2(
        brdf.getRayleighDensity(normalizedHeight),   // the density of air molecules
        brdf.getMieDensity(normalizedHeight)         // the density of air particles
    );
}

// helper function to get stable perpendicular unit vector
public float3 getStablePerpendicular(float3 direction)
{
    float3 axis = abs(direction.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    return normalize(cross(direction, axis));
}

public float2 opticalDepth(
    AtmosphereBRDF brdf,
    float3 sphereCenter,
    float3 rayOrigin,
    float3 rayDirection,
    float tStart,
    float tEnd,
    uint stepCount)
{
    if (tEnd <= tStart || stepCount == 0)
    {
        return float2(0.0);
    }

    float shellThickness = max(brdf.getShellThickness(), 1e-4);
    float stepLength = (tEnd - tStart) / stepCount;
    float rayleighOpticalDepth = 0.0;
    float mieOpticalDepth = 0.0;

    // integrate: the density Reyleigh go through / the density Mie go through
    for (uint i = 0; i < stepCount; ++i)
    {
        float t = tStart + (float(i) + 0.5) * stepLength;
        float3 position = rayOrigin + rayDirection * t;
        float2 densities = densityAtPoint(brdf, sphereCenter, position);
        rayleighOpticalDepth += densities.x * (stepLength / shellThickness);
        mieOpticalDepth      += densities.y * (stepLength / shellThickness);
    }

    return float2(rayleighOpticalDepth, mieOpticalDepth);
}

// Estimate how much of the light that would otherwise be blocked by the planet
// can still reach the sample point after skimming through the atmosphere.
//
// This is an intentionally cheap approximation rather than a full curved-ray
// refraction solve:
// - First, test whether the light ray passes behind the planet from the sample's
//   point of view.
// - If it does, measure how deeply that straight ray would penetrate into the
//   planet's silhouette.
// - Use that penetration depth to place a "virtual" tangent path inside the
//   atmosphere, representing a plausible refracted route around the limb.
// - Integrate optical depth along that virtual path and convert it to
//   transmittance with Beer-Lambert extinction.
//
// The result is a soft, colored leakage term near the planetary limb or horizon,
// where some sunlight should still appear even though the straight shadow ray is
// geometrically occluded by the planet.
public float3 getApproximateRefractedLightTransmittance(
    AtmosphereBRDF brdf,
    float3 sphereCenter,
    float3 samplePoint,
    float3 lightDirection,
    float maxDistance = float.maxValue)
{
    // determines how many samples are used to estimate the optical depth 
    // when approximately refracted light passes through the atmosphere.
    const uint REFRACTION_STEPS = 16;
    float3 direction = normalize(lightDirection);

    // Start slightly away from the sample to avoid self-intersection
    float3 rayStart = samplePoint + direction * 1e-3;
    float3 toCenter = sphereCenter - rayStart;

    // Project the vector from the ray start to the planet center onto the light
    // direction. This gives the distance to the closest approach point on the
    // straight light ray.
    float projectedDistance = dot(toCenter, direction);
    if (projectedDistance <= 0.0 || projectedDistance >= maxDistance)
    {
        // The closest approach is behind the ray start or beyond the valid light
        // segment, so there is no relevant near-limb refracted contribution.
        return float3(0.0);
    }

    float3 closestPoint = rayStart + direction * projectedDistance;
    float3 radialOffset = closestPoint - sphereCenter;

    // The impact parameter is the minimum distance from the straight light ray
    // to the planet center.
    float impactParameter = length(radialOffset);
    if (impactParameter >= brdf.planetRadius || impactParameter >= brdf.atmosphereRadius)
    {
        // If the ray misses the solid planet, it should be handled by the normal
        // atmospheric transmittance path instead of this refracted-shadow path.
        // If it even misses the atmosphere, there is also nothing to refract.
        return float3(0.0);
    }

    float shellThickness = max(brdf.getShellThickness(), 1e-4);

    // Only shallow shadow penetration is allowed to leak around the limb. Once
    // the straight ray cuts too deeply through the planet, this approximation
    // assumes atmospheric refraction can no longer recover any light.
    float maxRefractionReach = shellThickness * 0.6;   // 0.6: emperical
    float shadowPenetration = brdf.planetRadius - impactParameter;
    if (shadowPenetration >= maxRefractionReach)
    {
        return float3(0.0);
    }

    // Convert the penetration depth into a visibility weight. Rays that barely
    // dip into the planet's silhouette remain more visible after refraction,
    // while deeper rays are strongly suppressed.
    float refractedVisibility = 1.0 - saturate(shadowPenetration / max(maxRefractionReach, 1e-4));

    // Place the virtual refracted path at a heuristic altitude above the solid
    // surface. Deeper penetration pushes the path a bit higher, but we clamp it
    // to remain inside the lower half of the atmospheric shell.
    float effectiveAltitude = clamp(
        0.02 * shellThickness + shadowPenetration * 1.5,
        0.02 * shellThickness,
        0.5 * shellThickness
    );

    // Convert that altitude back to a radial distance from the planet center.
    float effectiveImpact = clamp(
        brdf.planetRadius + effectiveAltitude,
        brdf.planetRadius + 1e-4,
        brdf.atmosphereRadius - 1e-4
    );

    // Build a radial direction for the tangent point. In the degenerate case
    // where the closest point is numerically at the center line, fall back to a
    // stable perpendicular direction so the basis remains well-defined.
    float3 radialDirection = length(radialOffset) > 1e-4 ? normalize(radialOffset) : getStablePerpendicular(direction);
    float3 tangentPoint = sphereCenter + radialDirection * effectiveImpact;

    // A straight line at this impact parameter cuts a chord through the
    // atmosphere. We integrate over the full chord to approximate the total
    // extinction along the refracted path.
    float halfChordLength = sqrt(max(brdf.atmosphereRadius * brdf.atmosphereRadius - effectiveImpact * effectiveImpact, 0.0));
    if (halfChordLength <= 1e-4)
    {
        return float3(0.0);
    }

    float3 virtualRayOrigin = tangentPoint - direction * halfChordLength;

    // March through the virtual atmospheric chord and accumulate Rayleigh and
    // Mie optical depth.
    float2 opticalDepth = opticalDepth(
        brdf,
        sphereCenter,
        virtualRayOrigin,
        direction,
        0.0,
        2.0 * halfChordLength,
        REFRACTION_STEPS
    );

    // Convert integrated optical depth to wavelength-dependent transmittance,
    // then scale it by a small empirical intensity term so the effect reads as
    // subtle limb leakage rather than direct unoccluded sunlight.
    float3 transmittance = exp(-brdf.getExtinctionCoefficient(opticalDepth.x, opticalDepth.y));
    float refractedIntensity = 0.18 * refractedVisibility * refractedVisibility;
    return transmittance * refractedIntensity;
}

public bool isPlanetOccludingLight(float3 samplePoint, float3 lightDirection, float3 sphereCenter, float planetRadius, float maxDistance = float.maxValue)
{
    float2 hitRange = intersectSphere(samplePoint + normalize(lightDirection) * 1e-3, normalize(lightDirection), sphereCenter, planetRadius);
    return hitRange.x < maxDistance && hitRange.y > 0.0;
}

public float3 getAtmosphereLightTransmittance(
    AtmosphereBRDF brdf,
    float3 sphereCenter,
    float3 samplePoint,
    float3 lightDirection,
    float maxDistance = float.maxValue)
{
    const uint LIGHT_STEPS = 8;
    float3 direction = normalize(lightDirection);
    if (isPlanetOccludingLight(samplePoint, direction, sphereCenter, brdf.planetRadius, maxDistance))
    {
        return getApproximateRefractedLightTransmittance(brdf, sphereCenter, samplePoint, direction, maxDistance);
    }

    float2 shellRange = intersectSphere(samplePoint + direction * 1e-3, direction, sphereCenter, brdf.atmosphereRadius);
    float tStart = max(shellRange.x, 0.0);
    float tEnd = min(shellRange.y, maxDistance);
    if (tEnd <= tStart)
    {
        return float3(1.0);
    }

    float2 opticalDepth = opticalDepth(
        brdf,
        sphereCenter,
        samplePoint,
        direction,
        tStart,
        tEnd,
        LIGHT_STEPS
    );

    return exp(-brdf.getExtinctionCoefficient(opticalDepth.x, opticalDepth.y));
}

public float3 getApproximateMultipleScattering(
    AtmosphereBRDF brdf,
    float normalizedHeight,
    float rayleighStep,
    float mieStep,
    float3 lightTransmittance,
    float3 viewTransmittance)
{
    const float MULTI_SCATTER_STRENGTH = 0.08;
    float localDensity = max(rayleighStep + mieStep, 0.0);
    if (localDensity <= 1e-6)
    {
        return float3(0.0);
    }

    float lowerAtmosphereWeight = saturate(1.0 - normalizedHeight);
    float3 trappedLight = saturate(float3(1.0) - lightTransmittance);
    float3 partialLight = trappedLight * saturate(lightTransmittance + 0.15);
    float viewVisibility = dot(viewTransmittance, float3(0.33333334));
    float horizonDamping = smoothstep(0.15, 0.65, viewVisibility);
    float3 ambientTint =
        brdf.rayleighScatteringColor * (0.45 + 0.25 * lowerAtmosphereWeight) +
        brdf.mieScatteringColor * (0.05 + 0.15 * lowerAtmosphereWeight);

    return ambientTint * localDensity * partialLight * lowerAtmosphereWeight *
        horizonDamping * MULTI_SCATTER_STRENGTH;
}

public FilteredLightResult filterLightThroughAtmospheres(float3 hitPoint, float3 lightDirection, float lightDistance, RendererUniform uniforms, uint excludedMaterialId = 0xffffffff)
{
    FilteredLightResult result = FilteredLightResult(float3(1.0), false);
    float3 direction = normalize(lightDirection);
    float maxDistance = lightDistance;

    for (uint materialId = 0; materialId < uniforms.materialCount; ++materialId)
    {
        if (materialId == excludedMaterialId)
        {
            continue;
        }
        PhysicsBasedMaterial material = uniforms.physicsBasedMaterialBuf[materialId];
        if (material.atmosphereEnabled == 0 || material.atmosphereThickness <= 0.0)
        {
            continue;
        }

        float3 sphereCenter = material.atmosphereCenter;
        float outerRadius = material.atmospherePlanetRadius + material.atmosphereThickness;
        float2 atmosphereRange = intersectSphere(hitPoint + direction * 1e-3, direction, sphereCenter, outerRadius);
        float tEntry = max(atmosphereRange.x, 0.0);
        float tExit = min(atmosphereRange.y, maxDistance);
        if (tExit <= tEntry)
        {
            continue;
        }

        AtmosphereBRDF brdf = AtmosphereBRDF(
            material.albedo.uniformData,
            material.atmosphereRayleighScatteringColor,
            material.atmosphereMieScatteringColor,
            material.atmosphereAbsorptionColor,
            material.atmosphereRayleighDensityFalloff,
            material.atmosphereMieDensityFalloff,
            material.atmosphereRayleighStrength,
            material.atmosphereMieStrength,
            material.atmospherePhaseG,
            material.atmospherePlanetRadius,
            outerRadius
        );

        float2 planetRange = intersectSphere(hitPoint + direction * 1e-3, direction, sphereCenter, material.atmospherePlanetRadius);
        if (planetRange.y > 0.0 && planetRange.x < maxDistance)
        {
            float3 refractedTransmittance = getApproximateRefractedLightTransmittance(
                brdf,
                sphereCenter,
                hitPoint,
                direction,
                maxDistance
            );
            if (max(max(refractedTransmittance.x, refractedTransmittance.y), refractedTransmittance.z) <= 1e-4)
            {
                return FilteredLightResult(float3(0.0), true);
            }
            result.transmittance *= refractedTransmittance;
            continue;
        }

        if (tExit <= tEntry)
        {
            continue;
        }
        const uint LIGHT_FILTER_STEPS = 12;
        float2 opticalDepth = opticalDepth(
            brdf,
            sphereCenter,
            hitPoint,
            direction,
            tEntry,
            tExit,
            LIGHT_FILTER_STEPS
        );

        result.transmittance *= exp(-brdf.getExtinctionCoefficient(opticalDepth.x, opticalDepth.y));
    }

    return result;
}

public AtmosphereBRDF makeAtmosphereBRDF(PhysicsBasedParameters parameters)
{
    return AtmosphereBRDF(
        parameters.albedo,
        parameters.atmosphereRayleighScatteringColor,
        parameters.atmosphereMieScatteringColor,
        parameters.atmosphereAbsorptionColor,
        parameters.atmosphereRayleighDensityFalloff,
        parameters.atmosphereMieDensityFalloff,
        parameters.atmosphereRayleighStrength,
        parameters.atmosphereMieStrength,
        parameters.atmospherePhaseG,
        parameters.atmospherePlanetRadius,
        parameters.atmospherePlanetRadius + parameters.atmosphereThickness
    );
}

public float3 getAtmosphereDirectIllumination(PhysicsBasedParameters parameters, RendererUniform uniforms, float3 rayOut, float3 normal, float3 hitPoint, inout RNG rng)
{
    float3 outgoingRadiance = float3(0.0);
    const float ATMOSPHERE_BRIGHTNESS_SCALE = 0.5;
    const float EPSILON = 1e-2;
    AtmosphereBRDF brdf = makeAtmosphereBRDF(parameters);

    if (uniforms.pointLightCount == 0 && uniforms.directionalLightCount == 0 && uniforms.rectangularLightCount == 0)
    {
        LightSample lightSample = FlashLight().sample(rayOut);
        float3 irradiance = lightSample.radiance / lightSample.pdf;
        float3 atmosphereFr = brdf.evaluate(rayOut, normal, lightSample.lightDirection);
        float lightVisibility = 0.25 + 0.75 * abs(dot(normalize(normal), normalize(lightSample.lightDirection)));
        return atmosphereFr * irradiance * lightVisibility * ATMOSPHERE_BRIGHTNESS_SCALE;
    }

    for (int i = 0; i < uniforms.pointLightCount; i++)
    {
        LightSample lightSample = uniforms.pointLightBuf[i].sample(hitPoint);
        bool occluded = isLightOccluded(lightSample, hitPoint, uniforms);
        if (occluded)
        {
            continue;
        }
        FilteredLightResult filteredLight = filterLightThroughAtmospheres(
            hitPoint,
            lightSample.lightDirection,
            lightSample.lightDistance,
            uniforms
        );
        if (filteredLight.occluded)
        {
            continue;
        }
        float fallOffFactor = lightSample.lightDistance * lightSample.lightDistance;
        float3 irradiance = (lightSample.radiance / ((fallOffFactor + EPSILON) * lightSample.pdf)) * filteredLight.transmittance;
        float3 atmosphereFr = brdf.evaluate(rayOut, normal, lightSample.lightDirection);
        float lightVisibility = 0.25 + 0.75 * abs(dot(normalize(normal), normalize(lightSample.lightDirection)));
        outgoingRadiance += atmosphereFr * irradiance * lightVisibility * ATMOSPHERE_BRIGHTNESS_SCALE;
    }

    for (int i = 0; i < uniforms.directionalLightCount; i++)
    {
        LightSample lightSample = uniforms.directionalLightBuf[i].sample();
        bool occluded = isLightOccluded(lightSample, hitPoint, uniforms);
        if (occluded)
        {
            continue;
        }
        FilteredLightResult filteredLight = filterLightThroughAtmospheres(
            hitPoint,
            lightSample.lightDirection,
            lightSample.lightDistance,
            uniforms
        );
        if (filteredLight.occluded)
        {
            continue;
        }
        float3 irradiance = (lightSample.radiance / lightSample.pdf) * filteredLight.transmittance;
        float3 atmosphereFr = brdf.evaluate(rayOut, normal, lightSample.lightDirection);
        float lightVisibility = 0.25 + 0.75 * abs(dot(normalize(normal), normalize(lightSample.lightDirection)));
        outgoingRadiance += atmosphereFr * irradiance * lightVisibility * ATMOSPHERE_BRIGHTNESS_SCALE;
    }

    for (int i = 0; i < uniforms.rectangularLightCount; i++)
    {
        float3 accumulatedContribution = float3(0.0);
        int validSamples = 0;

        for (int j = 0; j < uniforms.numRectangularLightSamples; j++)
        {
            float2 randomEpsilon = rng.next_2d();
            LightSample lightSample = uniforms.rectangularLightBuf[i].sample(hitPoint, randomEpsilon);
            bool occluded = isLightOccluded(lightSample, hitPoint, uniforms);
            if (occluded)
            {
                continue;
            }
            FilteredLightResult filteredLight = filterLightThroughAtmospheres(
                hitPoint,
                lightSample.lightDirection,
                lightSample.lightDistance,
                uniforms
            );
            if (filteredLight.occluded)
            {
                continue;
            }
            float3 irradiance = (lightSample.radiance / lightSample.pdf) * filteredLight.transmittance;
            float3 atmosphereFr = brdf.evaluate(rayOut, normal, lightSample.lightDirection);
            float lightVisibility = 0.25 + 0.75 * abs(dot(normalize(normal), normalize(lightSample.lightDirection)));
            accumulatedContribution += atmosphereFr * irradiance * lightVisibility * ATMOSPHERE_BRIGHTNESS_SCALE;
            validSamples++;
        }

        if (validSamples > 0)
        {
            outgoingRadiance += accumulatedContribution / validSamples;
        }
    }

    return outgoingRadiance;
}

public AtmosphereMarchResult calculateLight(
    uint materialId,
    PhysicsBasedParameters parameters,
    RendererUniform uniforms,
    float3 rayOrigin,
    float3 rayDirection,
    float tStart,
    float tEnd)
{
    const uint VIEW_STEPS = 32;
    AtmosphereBRDF brdf = makeAtmosphereBRDF(parameters);
    float3 sphereCenter = parameters.atmosphereCenter;
    if (tEnd <= tStart)
    {
        return AtmosphereMarchResult(float3(0.0), float3(1.0));
    }

    float shellThickness = max(brdf.getShellThickness(), 1e-4);
    float segmentLength = tEnd - tStart;
    float stepLength = segmentLength / VIEW_STEPS;
    float3 transmittance = float3(1.0);
    float3 inscatteredRadiance = float3(0.0);
    float3 viewDirection = -normalize(rayDirection);

    [MaxIters(VIEW_STEPS)]
    for (uint i = 0; i < VIEW_STEPS; ++i)
    {
        float t = tStart + (float(i) + 0.5) * stepLength;
        float3 samplePoint = rayOrigin + rayDirection * t;
        float2 densities = densityAtPoint(brdf, sphereCenter, samplePoint);
        if (densities.x <= 1e-5 && densities.y <= 1e-5)
        {
            continue;
        }

        float rayleighStep = densities.x * (stepLength / shellThickness);
        float mieStep = densities.y * (stepLength / shellThickness);
        float radius = length(samplePoint - sphereCenter);
        float normalizedHeight = saturate((radius - brdf.planetRadius) / shellThickness);
        float3 stepScattering = float3(0.0);

        if (uniforms.pointLightCount == 0 && uniforms.directionalLightCount == 0 && uniforms.rectangularLightCount == 0)
        {
            LightSample lightSample = FlashLight().sample(viewDirection);
            bool occluded = isLightOccluded(lightSample, samplePoint, uniforms);
            if (occluded)
            {
                continue;
            }
            FilteredLightResult filteredLight = filterLightThroughAtmospheres(
                samplePoint,
                lightSample.lightDirection,
                lightSample.lightDistance,
                uniforms,
                materialId
            );
            if (filteredLight.occluded)
            {
                continue;
            }
            float3 lightTransmittance = getAtmosphereLightTransmittance(brdf, sphereCenter, samplePoint, lightSample.lightDirection, lightSample.lightDistance);
            float3 scatteringCoefficient = brdf.getScatteringCoefficient(
                dot(viewDirection, normalize(lightSample.lightDirection)),
                rayleighStep,
                mieStep
            );
            stepScattering += scatteringCoefficient * lightTransmittance * filteredLight.transmittance * lightSample.radiance;
            stepScattering += getApproximateMultipleScattering(
                brdf,
                normalizedHeight,
                rayleighStep,
                mieStep,
                lightTransmittance * filteredLight.transmittance,
                transmittance
            ) * lightSample.radiance;
        }

        for (int lightIdx = 0; lightIdx < uniforms.pointLightCount; ++lightIdx)
        {
            LightSample lightSample = uniforms.pointLightBuf[lightIdx].sample(samplePoint);
            bool occluded = isLightOccluded(lightSample, samplePoint, uniforms);
            if (occluded)
            {
                continue;
            }
            FilteredLightResult filteredLight = filterLightThroughAtmospheres(
                samplePoint,
                lightSample.lightDirection,
                lightSample.lightDistance,
                uniforms,
                materialId
            );
            if (filteredLight.occluded)
            {
                continue;
            }
            float distanceFalloff = max(lightSample.lightDistance * lightSample.lightDistance, 1e-4);
            float3 lightTransmittance = getAtmosphereLightTransmittance(brdf, sphereCenter, samplePoint, lightSample.lightDirection, lightSample.lightDistance);
            float3 scatteringCoefficient = brdf.getScatteringCoefficient(
                dot(viewDirection, normalize(lightSample.lightDirection)),
                rayleighStep,
                mieStep
            );
            stepScattering += scatteringCoefficient * lightTransmittance * filteredLight.transmittance * (lightSample.radiance / distanceFalloff);
            stepScattering += getApproximateMultipleScattering(
                brdf,
                normalizedHeight,
                rayleighStep,
                mieStep,
                lightTransmittance * filteredLight.transmittance,
                transmittance
            ) * (lightSample.radiance / distanceFalloff);
        }

        for (int lightIdx = 0; lightIdx < uniforms.directionalLightCount; ++lightIdx)
        {
            LightSample lightSample = uniforms.directionalLightBuf[lightIdx].sample();
            bool occluded = isLightOccluded(lightSample, samplePoint, uniforms);
            if (occluded)
            {
                continue;
            }
            FilteredLightResult filteredLight = filterLightThroughAtmospheres(
                samplePoint,
                lightSample.lightDirection,
                lightSample.lightDistance,
                uniforms,
                materialId
            );
            if (filteredLight.occluded)
            {
                continue;
            }
            float3 lightTransmittance = getAtmosphereLightTransmittance(brdf, sphereCenter, samplePoint, lightSample.lightDirection);
            float3 scatteringCoefficient = brdf.getScatteringCoefficient(
                dot(viewDirection, normalize(lightSample.lightDirection)),
                rayleighStep,
                mieStep
            );
            stepScattering += scatteringCoefficient * lightTransmittance * filteredLight.transmittance * lightSample.radiance;
            stepScattering += getApproximateMultipleScattering(
                brdf,
                normalizedHeight,
                rayleighStep,
                mieStep,
                lightTransmittance * filteredLight.transmittance,
                transmittance
            ) * lightSample.radiance;
        }

        for (int lightIdx = 0; lightIdx < uniforms.rectangularLightCount; ++lightIdx)
        {
            float3 accumulatedRectangular = float3(0.0);
            int validSamples = 0;
            for (int sampleIdx = 0; sampleIdx < uniforms.numRectangularLightSamples; ++sampleIdx)
            {
                float2 epsilon = float2(
                    frac(sin(float(i * 37 + sampleIdx * 17 + lightIdx * 13) * 12.9898) * 43758.5453),
                    frac(sin(float(i * 53 + sampleIdx * 23 + lightIdx * 19) * 78.233) * 24634.6345)
                );
                LightSample lightSample = uniforms.rectangularLightBuf[lightIdx].sample(samplePoint, epsilon);
                bool occluded = isLightOccluded(lightSample, samplePoint, uniforms);
                if (occluded)
                {
                    continue;
                }
                FilteredLightResult filteredLight = filterLightThroughAtmospheres(
                    samplePoint,
                    lightSample.lightDirection,
                    lightSample.lightDistance,
                    uniforms,
                    materialId
                );
                if (filteredLight.occluded)
                {
                    continue;
                }
                float3 lightTransmittance = getAtmosphereLightTransmittance(brdf, sphereCenter, samplePoint, lightSample.lightDirection, lightSample.lightDistance);
                float3 scatteringCoefficient = brdf.getScatteringCoefficient(
                    dot(viewDirection, normalize(lightSample.lightDirection)),
                    rayleighStep,
                    mieStep
                );
                accumulatedRectangular += scatteringCoefficient * lightTransmittance * filteredLight.transmittance * (lightSample.radiance / max(lightSample.pdf, 1e-4));
                accumulatedRectangular += getApproximateMultipleScattering(
                    brdf,
                    normalizedHeight,
                    rayleighStep,
                    mieStep,
                    lightTransmittance * filteredLight.transmittance,
                    transmittance
                ) * (lightSample.radiance / max(lightSample.pdf, 1e-4));
                validSamples++;
            }

            if (validSamples > 0)
            {
                stepScattering += accumulatedRectangular / validSamples;
            }
        }

        inscatteredRadiance += transmittance * stepScattering;
        transmittance *= exp(-brdf.getExtinctionCoefficient(rayleighStep, mieStep));
    }

    return AtmosphereMarchResult(inscatteredRadiance, transmittance);
}

public float3 getDirectIllumination(PhysicsBasedParameters parameters, RendererUniform uniforms, float3 rayOut, float3 normal, float3 hitPoint, inout RNG rng)
{
    float3 outgoingRadiance = float3(0.0);
    float3 fr = float3(0.0);
    bool lambertianScattering = false; // True only for Lambertian BRDF and Retroreflective Lambertian BRDF
    float3 retroreflectiveLambertianFr = float3(0.0);
    bool isAtmosphere = false;

    if (parameters.brdfType == 0)
    {
        LambertianBRDF brdf = LambertianBRDF(parameters.albedo, uniforms.useCosineWeightedSampling);
        fr = brdf.evaluate();
        lambertianScattering = true;
    }
    else if (parameters.brdfType == 1)
    {
        MirrorBRDF brdf = MirrorBRDF(parameters.albedo);
        // Mirror BRDF has nearly 0 chance of hitting a light source, unless the
        // light source lies along the path of ray direction bounced off it.
        // Remember that mirror bounces the ray in only one unique direction.
        // Check brdf/mirror.slang for more details.
        // Hence, we can simply return 0 for the BRDF value.
        fr = float3(0.0);
    }
    else if (parameters.brdfType == 2)
    {
        GlassBRDF brdf = GlassBRDF(parameters.albedo, parameters.ior);
        // Glass BRDF passes light through with minimal direct illumination contribution
        fr = float3(0.0);
    }
    else if (parameters.brdfType == 3)
    {
        RetroreflectiveBRDF brdf = RetroreflectiveBRDF(parameters.albedo);
        retroreflectiveLambertianFr = brdf.evaluate(); // Used only with Flash light
        fr = float3(0.0);
    }
    else if (parameters.brdfType == 4)
    {
        RetroreflectiveLambertianBRDF brdf = RetroreflectiveLambertianBRDF(parameters.albedo);
        retroreflectiveLambertianFr = brdf.evaluate(); // Used only with Flash light
        fr = float3(0.0);
        lambertianScattering = true;
    }
    else if (parameters.brdfType == 5)
    {
        isAtmosphere = true;
    }

    // If there are no light sources, we simply use the flash light source.
    if (uniforms.pointLightCount == 0 && uniforms.directionalLightCount == 0 && uniforms.rectangularLightCount == 0)
    {
        if (isAtmosphere)
        {
            return getAtmosphereDirectIllumination(parameters, uniforms, rayOut, normal, hitPoint, rng);
        }

        // For the combination of Retroreflective (Lambertian or not) BRDF and Flash light,
        if (parameters.brdfType == 3 || parameters.brdfType == 4)
        {
            fr = retroreflectiveLambertianFr;
        }
        float3 flashLightContribution = evaluateFlashLightContribution(hitPoint, fr, normal, uniforms, rayOut, lambertianScattering);
        outgoingRadiance = flashLightContribution;
    }
    else
    {
        if (isAtmosphere)
        {
            return getAtmosphereDirectIllumination(parameters, uniforms, rayOut, normal, hitPoint, rng);
        }

        float3 pointLightContribution = evaluatePointLightContribution(uniforms.pointLightBuf, uniforms.pointLightCount, hitPoint, fr, normal, uniforms, lambertianScattering);
        float3 directionalLightContribution = evaluateDirectionalLightContribution(uniforms.directionalLightBuf, uniforms.directionalLightCount, hitPoint, fr, normal, uniforms, lambertianScattering);
        float3 rectangularLightContribution = evaluateRectangularLightContribution(uniforms.rectangularLightBuf, uniforms.rectangularLightCount, hitPoint, fr, normal, uniforms, rng, lambertianScattering);
        outgoingRadiance = pointLightContribution + directionalLightContribution + rectangularLightContribution;
    }

    return outgoingRadiance;
}

public SampleBRDFResult sampleBRDF(PhysicsBasedParameters parameters, float3 rayOut, float3 normal, RendererUniform uniforms, inout RNG rng)
{
    float3 fr = float3(0.0);
    float3 sampledDirection = float3(0.0);
    float sampledPdf = 0.0;
    if (parameters.brdfType == 0)
    {
        // Sample Lambertian BRDF
        LambertianBRDF brdf = LambertianBRDF(parameters.albedo, uniforms.useCosineWeightedSampling);
        fr = brdf.evaluate();
        sampledDirection = brdf.sample(rayOut, normal, rng.next_2d());
        float lambertianPdf = brdf.pdf(sampledDirection, rayOut, normal);
        // Account for the probability of choosing lambertian in the overall PDF
        sampledPdf = lambertianPdf;
    }
    else if (parameters.brdfType == 1)
    {
        MirrorBRDF brdf = MirrorBRDF(parameters.albedo);
        // Note that BRDF evaluates to mirror's albedo here, unlike in
        // `getIllumination` where it evaluates to 0.
        // This is because we are sampling the BRDF for the next bounce for
        // indiret illumination, and not for the direct illumination.
        fr = brdf.evaluate();
        sampledDirection = brdf.sample(rayOut, normal);
        sampledPdf = brdf.pdf(sampledDirection, rayOut, normal);
    }
    else if (parameters.brdfType == 2)
    {
        if (uniforms.enableFresnelEffect)
        {
            // Extra Credit: Fresnel effect for Glass BRDF
            // TODO: Student implementation starts here.

            GlassBRDF brdf = GlassBRDF(parameters.albedo, parameters.ior);
            fr = brdf.evaluate();

            float3 wo = normalize(rayOut);
            float3 n = normalize(normal);
            float cosTheta = abs(dot(wo, n));

            bool entering = dot(wo, n) > 0.0;
            float n1 = entering ? 1.0 : parameters.ior;
            float n2 = entering ? parameters.ior : 1.0;
            float eta = n1 / n2;

            float sin2ThetaT = eta * eta * (1.0 - cosTheta * cosTheta);
            bool totalInternalReflection = sin2ThetaT > 1.0;

            float r0 = (n1 - n2) / (n1 + n2);
            r0 = r0 * r0;
            float fresnelReflectance = r0 + (1.0 - r0) * pow(1.0 - cosTheta, 5.0);
            fresnelReflectance = clamp(fresnelReflectance, 0.0, 1.0);

            if (totalInternalReflection || rng.next_1d() < fresnelReflectance)
            {
                MirrorBRDF mirror = MirrorBRDF(parameters.albedo);
                sampledDirection = mirror.sample(rayOut, normal);
                sampledPdf = totalInternalReflection ? 1.0 : fresnelReflectance;
            }
            else
            {
                sampledDirection = brdf.sample(rayOut, normal);
                sampledPdf = 1.0 - fresnelReflectance;
            }

            // TODO: Student implementation ends here.
        }
        else
        {
            // Naive Glass BRDF sampling without Fresnel effect.
            GlassBRDF brdf = GlassBRDF(parameters.albedo, parameters.ior);
            fr = brdf.evaluate();
            sampledDirection = brdf.sample(rayOut, normal);
            float glassPdf = brdf.pdf(sampledDirection, rayOut, normal);
            // Account for the probability of choosing glass in the overall PDF
            sampledPdf = glassPdf;
        }
    }
    else if (parameters.brdfType == 3)
    {
        RetroreflectiveBRDF brdf = RetroreflectiveBRDF(parameters.albedo);
        fr = brdf.evaluate();
        sampledDirection = brdf.sample(rayOut);
        sampledPdf = brdf.pdf(sampledDirection, rayOut, normal);
    }
    else if (parameters.brdfType == 4)
    {
        RetroreflectiveLambertianBRDF brdf = RetroreflectiveLambertianBRDF(parameters.albedo);
        fr = brdf.evaluate();
        sampledDirection = brdf.sample(rayOut);
        sampledPdf = brdf.pdf(sampledDirection, rayOut, normal);
    }
    else if (parameters.brdfType == 5)
    {
        AtmosphereBRDF brdf = AtmosphereBRDF(
            parameters.albedo,
            parameters.atmosphereRayleighScatteringColor,
            parameters.atmosphereMieScatteringColor,
            parameters.atmosphereAbsorptionColor,
            parameters.atmosphereRayleighDensityFalloff,
            parameters.atmosphereMieDensityFalloff,
            parameters.atmosphereRayleighStrength,
            parameters.atmosphereMieStrength,
            parameters.atmospherePhaseG,
            parameters.atmospherePlanetRadius,
            parameters.atmospherePlanetRadius + parameters.atmosphereThickness
        );
        sampledDirection = brdf.sample(rayOut, normal);
        fr = brdf.evaluate(rayOut, normal, sampledDirection);
        sampledPdf = brdf.pdf(sampledDirection, rayOut, normal);
    }

    return SampleBRDFResult(fr, sampledDirection, sampledPdf);
}
