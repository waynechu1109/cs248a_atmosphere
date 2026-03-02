implementing "../renderer.slang";

import math;
import texture;

public static uint MAX_STEPS = 1024;
public static float EPSILON = 1e-6;

public static float STEP_SIZE = 0.01;
public static float DENSITY_SCALE = 5.0;

/**
 * Perform ray marching through the volume and accumulate color.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param tMax The maximum t value for ray marching (usually the nearest primitive intersection t).
 * @param primColor The color of the nearest primitive intersection.
 * @return The accumulated color after ray marching through the volume.
 */
public float4 volumeSample(float2 uv, RendererUniform uniforms, float tMax, float4 primColor)
{
    // TODO: Student implementation starts here.

    Ray rayWorld = uniforms.camera.generateRay(uv);
    rayWorld.tRange = float2(0.0, tMax);

    Ray rayLocal = rayWorld.transform(uniforms.volume.invModelMatrix);

    return accumulateColor<1024>(
        rayLocal,
        uniforms.volume.bound,
        MAX_STEPS,
        STEP_SIZE,
        DENSITY_SCALE,
        primColor,
        uniforms.volume.tex);
    
    // TODO: Student implementation ends here.
}

/**
 * Backward pass for volumetric ray marching at a single uv.
 * @param uv The normalized uv coordinate from (0, 0) in the bottom-left to (1, 1) in the top-right.
 * @param uniforms The renderer uniform containing camera, sampling, and scene information.
 * @param outGrad The output gradient w.r.t. the sampled color.
 */
public void volumeSampleBwd(float2 uv, RendererUniform uniforms, float4 outGrad)
{
    // TODO: Student implementation starts here.

    uint2 pixel = uint2(uv * float2(uniforms.camera.canvasSize));
    RNG rng = RNG(pixel, (pixel.y * uniforms.camera.canvasSize.x + pixel.x) * 10000 + uniforms.seed);
    Optional<RayMeshSampleResult> rayMeshSampleRes = meshSample(uv, uniforms, rng);
    float4 primColor = uniforms.ambientColor;
    if (rayMeshSampleRes.hasValue)
    {
        primColor = rayMeshSampleRes.value.color;
    }
    float tMax = rayMeshSampleRes.hasValue ? rayMeshSampleRes.value.t : float.maxValue;

    Ray rayWorld = uniforms.camera.generateRay(uv);
    rayWorld.tRange = float2(0.0, tMax);

    Ray rayLocal = rayWorld.transform(uniforms.volume.invModelMatrix);

    DifferentialPtrPair<DiffTexture3D<float, 4>> volumeTexPair = {
        uniforms.volume.tex,
        uniforms.volume.dTex
    };

    bwd_diff(accumulateColor<1024>)(
        rayLocal,
        uniforms.volume.bound,
        MAX_STEPS,
        STEP_SIZE,
        DENSITY_SCALE,
        primColor,
        volumeTexPair,
        outGrad     // 代表 loss 對 accumulateColor 輸出的梯度
    );

    // TODO: Student implementation ends here.
}

/**
 * Ray march through the volume and accumulate color.
 * @param ray The ray in local space.
 * @param volumeBound The bounding box of the volume in local space.
 * @param maxSteps The maximum number of marching steps.
 * @param stepSize The size of each marching step.
 * @param densityScale The scale factor for the volume density (alpha channel).
 * @param backgroundColor The background color behind the volume
                          (this can be the ambient color if there's no primitive behind the volume, or the color of the primitive hit if there's one).
 * @param data The 3D texture storing the volume data (rgba).
 * @return The accumulated color after ray marching through the volume.
 */
[Differentiable]
public float4 accumulateColor<let N : uint>(
    no_diff Ray ray,
    no_diff BoundingBox volumeBound,
    no_diff uint maxSteps,
    no_diff float stepSize,
    no_diff float densityScale,
    no_diff float4 backgroundColor,
    DiffTexture3D<float, 4> data)
{
    // TODO: Student implementation starts here.

    float2 tRangeBox = volumeBound.hit(ray);
    if (tRangeBox.x > tRangeBox.y)
    {
        return backgroundColor;
    }

    float tMin = max(ray.tRange.x, tRangeBox.x);
    float tMax = min(ray.tRange.y, tRangeBox.y);
    if (tMin > tMax)
    {
        return backgroundColor;
    }

    float dirLen = length(ray.direction);
    float stepT = stepSize / max(dirLen, EPSILON);

    float transmittance = 1.0;
    float3 radiance = float3(0.0, 0.0, 0.0);  // start acculumate radiance from 0
    float t = tMin;

    float3 extent = volumeBound.pMax - volumeBound.pMin;
    float3 invExtent = 1.0 / max(extent, float3(EPSILON, EPSILON, EPSILON));

    [MaxIters(N)]
    for (uint i = 0; i < maxSteps && t <= tMax; ++i)
    {
        float3 pLocal = ray.origin + t * ray.direction;
        float3 uvw = (pLocal - volumeBound.pMin) * invExtent;

        float density = 0.0;

        float3 emission = float3(0.0, 0.0, 0.0);
        if (all(uvw >= 0.0) && all(uvw <= 1.0))
        {
            float4 sample = sampleTrilinear(data, uvw);
            density = clamp(sample.w, 0.0, 1.0);
            emission = max(sample.xyz, 0.0);
        }

        float a = density * densityScale * stepSize;
        float stepAtten = max(1.0 - a, 0.0);
        radiance += transmittance * emission * a;
        transmittance *= stepAtten;

        // if (transmittance <= EPSILON)
        // {
        //     transmittance = 0.0;
        //     break;
        // }

        t += stepT;
    }

    // return float4(radiance + backgroundColor.xyz * transmittance, backgroundColor.w);
    float outAlpha = 1.0 - transmittance + backgroundColor.w * transmittance;
    return float4(radiance + backgroundColor.xyz * transmittance, outAlpha);

    // TODO: Student implementation ends here.
}
