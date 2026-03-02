implementing "../renderer.slang";

import light;

public struct RayMeshIntersectionResult
{
    public RayHitResult closestHit;
    public Triangle hitTriangle;
}

public Optional<RectangularLight.LightHitResult> rayRectangularLightIntersection(Ray ray, RendererUniform uniforms)
{
    Optional<RectangularLight.LightHitResult> closestLightHitResult = Optional<RectangularLight.LightHitResult>();
    for (int i = 0; i < uniforms.rectangularLightCount; i++)
    {
        Optional<RectangularLight.LightHitResult> lightHitResult = uniforms.rectangularLightBuf[i].hit(ray);
        if (lightHitResult.hasValue)
        {
            if (!closestLightHitResult.hasValue || lightHitResult.value.t < closestLightHitResult.value.t)
            {
                closestLightHitResult = lightHitResult;
            }
        }
    }
    return closestLightHitResult;
}

public Optional<RayMeshIntersectionResult> rayMeshIntersection(Ray ray, RendererUniform uniforms, bool occlusionTest = false)
{
    Optional<RayHitResult> result = Optional<RayHitResult>();
    Optional<RayMeshIntersectionResult> allResult = Optional<RayMeshIntersectionResult>();
    Optional<Triangle> hitTriangle = Optional<Triangle>();

    // IMPORTANT NOTE: Check the modified signature of `rayMeshIntersection` function in ray_mesh_intersection.slang.
    // Can you modify your existing code to make use of this flag? This is helpful in `isLightOccluded` function.

    // TODO: Student implementation starts here.

    RayHitResult initHit;
    initHit.t = float.maxValue;
    initHit.normal = float3(0.0, 0.0, 0.0);

    result = Optional<RayHitResult>(initHit);

    // 1. ray-triangle intersection
    if (uniforms.useBVH)
    {
        // use BVH
        Optional<BVH<Triangle>.HitResult> bvhHit = uniforms.bvh.hit(ray);

        if (bvhHit.hasValue)
        {
            result = Optional<RayHitResult>(bvhHit.value.rayHitResult);
            hitTriangle = Optional<Triangle>(bvhHit.value.primitive);
        }
    }
    else
    {
        // original brute-force loop
        // iterate through all triangles
        for (int i = 0; i < uniforms.triangleCount; i++)
        {
            Triangle tri = uniforms.triangleBuf[i];
            Optional<RayHitResult> hit = tri.hit(ray);

            if (hit.hasValue)
            {
                if (hit.value.t < result.value.t)
                {
                    result = hit;
                    hitTriangle = Optional<Triangle>(tri);
                    if (occlusionTest)
                    {
                        break;
                    }
                }
            }
        }
    }

    // TODO: Student implementation ends here.

    if (result.hasValue && hitTriangle.hasValue)
    {
        allResult = Optional<RayMeshIntersectionResult>(RayMeshIntersectionResult(result.value, hitTriangle.value));
    }

    return allResult;
}
